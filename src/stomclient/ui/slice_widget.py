"""Thin 2D slice view: window/level drag, measurement drawing, mask overlay."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .. import slice_renderer as sr
from ..app_controller import AppController
from ..coords import widget_to_image
from .qt_image import ndarray_to_qimage


class SliceWidget(QWidget):
    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller
        self.setMinimumSize(256, 256)
        self.measure_mode = False
        self._drag_start: tuple[float, float] | None = None   # measure, image coords
        self._drag_end: tuple[float, float] | None = None
        self._wl_anchor: tuple[float, float] | None = None     # window/level drag, widget px

    def set_measure_mode(self, on: bool) -> None:
        self.measure_mode = on
        self._drag_start = self._drag_end = None
        self.update()

    def _image_size(self) -> tuple[int, int]:
        c = self._c
        if c.volume is None:
            return (1, 1)
        z, y, x = c.volume.shape
        return {sr.AXIAL: (x, y), sr.CORONAL: (x, z), sr.SAGITTAL: (y, z)}[c.plane]

    def render_image(self) -> QImage:
        c = self._c
        if c.volume is None:
            return QImage(1, 1, QImage.Format.Format_RGB888)
        gray = sr.apply_window_level(
            sr.slice_array(c.volume.voxels, c.plane, c.index),
            c.window_center, c.window_width,
        )
        if c.mask is not None:
            mask_slice = sr.slice_array(c.mask.labels, c.plane, c.index)
            rgb = sr.composite_overlay(gray, mask_slice, c.mask.label_map, alpha=0.5)
        else:
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
        return ndarray_to_qimage(rgb)

    def wheelEvent(self, event) -> None:
        step = 1 if event.angleDelta().y() > 0 else -1
        self._c.set_index(self._c.index + step)
        self.update()

    def mousePressEvent(self, event) -> None:
        pos = (event.position().x(), event.position().y())
        if self.measure_mode:
            mapped = widget_to_image(pos, (self.width(), self.height()), self._image_size())
            if mapped is not None:
                self._drag_start = mapped
                self._drag_end = mapped
        else:
            self._wl_anchor = pos
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = (event.position().x(), event.position().y())
        if self.measure_mode and self._drag_start is not None:
            mapped = widget_to_image(pos, (self.width(), self.height()), self._image_size())
            if mapped is not None:
                self._drag_end = mapped
        elif self._wl_anchor is not None:
            dx = pos[0] - self._wl_anchor[0]
            dy = pos[1] - self._wl_anchor[1]
            self._wl_anchor = pos
            self._c.set_window_level(self._c.window_center + dy, self._c.window_width + dx)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self.measure_mode and self._drag_start and self._drag_end:
            self._c.add_measurement(self._drag_start, self._drag_end)
            self._drag_start = self._drag_end = None
        self._wl_anchor = None
        self.update()

    def _scale(self) -> float:
        iw, ih = self._image_size()
        return min(self.width() / iw, self.height() / ih)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        img = self.render_image()
        scale = self._scale()
        target = img.scaled(
            int(img.width() * scale), int(img.height() * scale),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawImage(0, 0, target)

        pen = QPen(Qt.GlobalColor.yellow)
        pen.setWidth(1)
        painter.setPen(pen)
        for m in self._c.measurements:
            self._draw_line(painter, m.p0, m.p1, scale, f"{m.length_mm:.1f} mm")
        if self._drag_start and self._drag_end:
            self._draw_line(painter, self._drag_start, self._drag_end, scale, "")
        painter.end()

    def _draw_line(self, painter, p0, p1, scale, label) -> None:
        a = QPointF(p0[0] * scale, p0[1] * scale)
        b = QPointF(p1[0] * scale, p1[1] * scale)
        painter.drawLine(a, b)
        if label:
            painter.drawText(b, label)
