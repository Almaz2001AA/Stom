"""Main window: left panel + slice view + toolbar wiring. Thin over AppController."""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from stomcore.dicom_loader import DicomError, DicomLoader

from .. import slice_renderer as sr
from ..app_controller import AppController, State
from ..cloud_client import CloudError
from .slice_widget import SliceWidget


class _SubmitWorker(QObject):
    """Runs the blocking submit() off the UI thread."""

    done = Signal()
    failed = Signal(str)

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller

    def run(self) -> None:
        try:
            self._c.submit()
            self.done.emit()
        except (CloudError, RuntimeError) as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller
        self.setWindowTitle("Stom — CBCT Viewer")

        self.slice_widget = SliceWidget(controller)
        self._status = QLabel("No study")
        self._plane = QComboBox()
        self._plane.addItems(list(sr.PLANES))
        self._plane.currentTextChanged.connect(self._on_plane)

        open_btn = QPushButton("Open DICOM…")
        open_btn.clicked.connect(self._on_open)
        segment_btn = QPushButton("Upload & Segment")
        segment_btn.clicked.connect(self._on_segment)

        left = QVBoxLayout()
        left.addWidget(open_btn)
        left.addWidget(segment_btn)
        left.addWidget(self._plane)
        left.addWidget(self._status)
        left.addStretch(1)
        left_panel = QWidget()
        left_panel.setLayout(left)

        root = QHBoxLayout()
        root.addWidget(left_panel, 0)
        root.addWidget(self.slice_widget, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._on_poll_tick)
        self._thread: QThread | None = None

    def _refresh(self) -> None:
        self._status.setText(self._c.state.value)
        self.slice_widget.update()

    def _on_plane(self, plane: str) -> None:
        if self._c.volume is not None:
            self._c.set_plane(plane)
            self._refresh()

    def _on_open(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open DICOM series")
        if not directory:
            return
        try:
            volume = DicomLoader.load(directory)
        except DicomError as exc:
            QMessageBox.critical(self, "DICOM error", str(exc))
            return
        self._c.load_volume(volume)
        self._refresh()

    def _on_segment(self) -> None:
        if self._c.volume is None:
            QMessageBox.information(self, "No study", "Open a DICOM series first.")
            return
        self._thread = QThread(self)
        worker = _SubmitWorker(self._c)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.done.connect(self._on_submitted)
        worker.failed.connect(self._on_submit_failed)
        worker.done.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
        self._worker = worker  # keep ref
        self._thread.start()
        self._refresh()

    def _on_submitted(self) -> None:
        self._refresh()
        self._poll_timer.start()

    def _on_submit_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Cloud error", message)
        self._refresh()

    def _on_poll_tick(self) -> None:
        try:
            terminal = self._c.poll()
        except CloudError as exc:
            self._poll_timer.stop()
            QMessageBox.critical(self, "Cloud error", str(exc))
            return
        if terminal:
            self._poll_timer.stop()
            if self._c.state == State.FAILED:
                QMessageBox.warning(self, "Segmentation failed", self._c.error or "")
        self._refresh()
