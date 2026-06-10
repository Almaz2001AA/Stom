"""Thin 2D slice view: builds a QImage from controller state and paints it."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

from .. import slice_renderer as sr
from ..app_controller import AppController
from .qt_image import ndarray_to_qimage


class SliceWidget(QWidget):
    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller
        self.setMinimumSize(256, 256)

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

    def wheelEvent(self, event) -> None:  # scroll changes slice index
        step = 1 if event.angleDelta().y() > 0 else -1
        self._c.set_index(self._c.index + step)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        img = self.render_image()
        target = img.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawImage(0, 0, target)
        painter.end()
