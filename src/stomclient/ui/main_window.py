"""Main window: panel (mask list, tools), slice view, cloud roundtrip wiring."""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from stomcore.dicom_loader import DicomError, DicomLoader
from stomcore.mask_io import save_mask_nifti

from .. import slice_renderer as sr
from ..app_controller import AppController, State
from ..cloud_client import CloudClient, CloudError
from ..config import load, save
from .settings_dialog import SettingsDialog
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
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _InstallWorker(QObject):
    """Downloads + installs the engine-pack off the UI thread."""
    progress = Signal(int, int)  # (bytes_done, bytes_total)
    done = Signal(object)        # the ready engine
    failed = Signal(str)

    def __init__(self, provision_fn) -> None:
        super().__init__()
        self._provision = provision_fn

    def run(self) -> None:
        try:
            engine = self._provision(progress=lambda d, t: self.progress.emit(d, t))
            self.done.emit(engine)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller
        self._config = load()
        self.setWindowTitle("Stom — CBCT Viewer")

        self.slice_widget = SliceWidget(controller)
        self._status = QLabel("No study")
        self._plane = QComboBox()
        self._plane.addItems(list(sr.PLANES))
        self._plane.currentTextChanged.connect(self._on_plane)

        settings_btn = QPushButton("Settings…")
        settings_btn.clicked.connect(self._on_settings)
        open_btn = QPushButton("Open DICOM…")
        open_btn.clicked.connect(self._on_open)
        self._segment_btn = QPushButton("Upload & Segment")
        self._segment_btn.clicked.connect(self._on_segment)
        # Local (on-device) segmentation: only offered when an engine is wired,
        # so there is no network round-trip for the large study upload.
        self._local_chk = QCheckBox("Local (on-device)")
        self._local_chk.setEnabled(self._c.local_available)
        self._local_chk.setToolTip(
            "Segment on this PC — no server upload."
            if self._c.local_available
            else "Local engine not installed."
        )
        self._local_chk.toggled.connect(self._on_local_toggled)
        # First-run install of the engine-pack: shown only until an engine is
        # wired, so the checkbox above can actually be enabled on a slim client.
        self._install_btn = QPushButton("Install local engine…")
        self._install_btn.setToolTip("Download the on-device segmentation engine (~0.5 GB).")
        self._install_btn.setVisible(not self._c.local_available)
        self._install_btn.clicked.connect(self._on_install_engine)
        self._measure_btn = QPushButton("Measure")
        self._measure_btn.setCheckable(True)
        self._measure_btn.toggled.connect(self.slice_widget.set_measure_mode)
        clear_btn = QPushButton("Clear measurements")
        clear_btn.clicked.connect(self._on_clear_measurements)
        png_btn = QPushButton("Save PNG…")
        png_btn.clicked.connect(self._on_save_png)
        mask_btn = QPushButton("Save Mask…")
        mask_btn.clicked.connect(self._on_save_mask)

        self.mask_list = QListWidget()
        self.mask_list.itemChanged.connect(self._on_mask_item_changed)

        left = QVBoxLayout()
        for w in (settings_btn, open_btn, self._segment_btn, self._local_chk,
                  self._install_btn, self._plane, self._measure_btn,
                  clear_btn, png_btn, mask_btn, QLabel("Masks:"), self.mask_list,
                  self._status):
            left.addWidget(w)
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
        self._install_thread: QThread | None = None
        self._install_progress: QProgressDialog | None = None

    def refresh(self) -> None:
        self._status.setText(self._c.state.value)
        self._rebuild_mask_list()
        self.slice_widget.update()

    def _rebuild_mask_list(self) -> None:
        self.mask_list.blockSignals(True)
        self.mask_list.clear()
        if self._c.mask is not None:
            for label_id, info in sorted(self._c.mask.label_map.items()):
                item = QListWidgetItem(f"{label_id}: {info.name}")
                item.setData(Qt.ItemDataRole.UserRole, label_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if info.visible else Qt.CheckState.Unchecked
                )
                self.mask_list.addItem(item)
        self.mask_list.blockSignals(False)

    def _on_mask_item_changed(self, item: QListWidgetItem) -> None:
        label_id = item.data(Qt.ItemDataRole.UserRole)
        self._c.set_label_visible(label_id, item.checkState() == Qt.CheckState.Checked)
        self.slice_widget.update()

    def _on_plane(self, plane: str) -> None:
        if self._c.volume is not None:
            self._c.set_plane(plane)
            self.refresh()

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
        self.refresh()

    def _on_clear_measurements(self) -> None:
        self._c.clear_measurements()
        self.slice_widget.update()

    def _on_save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "slice.png", "PNG (*.png)")
        if path:
            self.slice_widget.render_image().save(path, "PNG")

    def _on_save_mask(self) -> None:
        if self._c.mask is None:
            QMessageBox.information(self, "No mask", "No mask to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save mask", "mask.nii.gz", "NIfTI (*.nii.gz)")
        if path:
            labels_path = path.replace(".nii.gz", "").rstrip(".") + "_labels.json"
            save_mask_nifti(self._c.mask, path, labels_path)

    def _on_settings(self) -> None:
        dialog = SettingsDialog(self._config, self)
        if dialog.exec():
            self._apply_config(dialog.values())

    def _apply_config(self, config) -> None:
        save(config)
        self._config = config
        cloud = CloudClient(config.server_url, config.token) if config.server_url else None
        self._c.set_cloud_client(cloud)

    def _on_local_toggled(self, checked: bool) -> None:
        self._c.set_local_mode(checked)
        self._segment_btn.setText("Segment (local)" if checked else "Upload & Segment")

    def _on_install_engine(self) -> None:
        if self._install_thread is not None and self._install_thread.isRunning():
            return
        from ..local_engine import provision_local_engine

        self._install_btn.setEnabled(False)
        self._install_progress = QProgressDialog(
            "Downloading on-device engine…", None, 0, 100, self
        )
        self._install_progress.setWindowTitle("Install local engine")
        self._install_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._install_progress.setAutoClose(True)
        self._install_progress.setMinimumDuration(0)
        self._install_progress.setValue(0)

        self._install_thread = QThread(self)
        worker = _InstallWorker(provision_local_engine)
        worker.moveToThread(self._install_thread)
        self._install_thread.started.connect(worker.run)
        worker.progress.connect(self._on_install_progress)
        worker.done.connect(self._on_install_done)
        worker.failed.connect(self._on_install_failed)
        worker.done.connect(self._install_thread.quit)
        worker.failed.connect(self._install_thread.quit)
        self._install_worker = worker
        self._install_thread.start()

    def _on_install_progress(self, done: int, total: int) -> None:
        if total > 0:
            self._install_progress.setValue(min(100, done * 100 // total))

    def _on_install_done(self, engine: object) -> None:
        if self._install_progress is not None:
            self._install_progress.reset()
        self._install_btn.setEnabled(True)
        self._install_btn.setVisible(False)
        self._c.set_engine(engine)
        self._local_chk.setEnabled(True)
        self._local_chk.setToolTip("Segment on this PC — no server upload.")
        self._local_chk.setChecked(True)
        QMessageBox.information(self, "Local engine", "On-device engine installed.")

    def _on_install_failed(self, message: str) -> None:
        if self._install_progress is not None:
            self._install_progress.reset()
        self._install_btn.setEnabled(True)
        QMessageBox.critical(self, "Install failed", message)

    def _on_segment(self) -> None:
        if not self._c.local and not self._config.server_url:
            QMessageBox.information(self, "No server", "Open Settings and set the server URL.")
            return
        if self._c.volume is None:
            QMessageBox.information(self, "No study", "Open a DICOM series first.")
            return
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = QThread(self)
        worker = _SubmitWorker(self._c)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.done.connect(self._on_submitted)
        worker.failed.connect(self._on_submit_failed)
        worker.done.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
        self._worker = worker
        self._thread.start()
        self.refresh()

    def _on_submitted(self) -> None:
        self.refresh()
        self._poll_timer.start()

    def _on_submit_failed(self, message: str) -> None:
        title = "Segmentation error" if self._c.local else "Cloud error"
        QMessageBox.critical(self, title, message)
        self.refresh()

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
        self.refresh()
