"""Main window: panel (mask list, tools), slice view, cloud roundtrip wiring."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
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
from . import strings as S
from .settings_dialog import SettingsDialog
from .slice_widget import SliceWidget


class _SubmitWorker(QObject):
    """Runs the blocking submit() off the UI thread."""
    done = Signal()
    failed = Signal(str)
    progress = Signal(int, int)  # (steps_done, steps_total)

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller

    def run(self) -> None:
        try:
            # The callback fires on this worker thread; emitting a queued signal
            # marshals the update to the UI thread.
            self._c.submit(progress=lambda done, total: self.progress.emit(done, total))
            self.done.emit()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _InstallWorker(QObject):
    """Downloads + installs (or updates) the engine-pack off the UI thread."""
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


class _FnWorker(QObject):
    """Runs a no-arg function off the UI thread and emits its result.

    Used for the fail-soft startup update checks (engine + client), which do
    network I/O and must not block window construction.
    """
    done = Signal(object)

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self._fn())
        except Exception:  # noqa: BLE001 - update checks never surface errors
            self.done.emit(None)


class _DownloadWorker(QObject):
    """Downloads the client installer off the UI thread."""
    progress = Signal(int, int)
    done = Signal(str)   # path to the downloaded installer
    failed = Signal(str)

    def __init__(self, url: str, dest: Path) -> None:
        super().__init__()
        self._url = url
        self._dest = dest

    def run(self) -> None:
        from ..updates import download_installer

        try:
            path = download_installer(
                self._url, self._dest,
                progress=lambda d, t: self.progress.emit(d, t),
            )
            self.done.emit(str(path))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


def _section(title: str) -> QLabel:
    """A bold section header for the left panel."""
    label = QLabel(title)
    font = label.font()
    font.setBold(True)
    label.setFont(font)
    return label


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    return line


class MainWindow(QMainWindow):
    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller
        self._config = load()
        self.setWindowTitle(S.WINDOW_TITLE)

        self.slice_widget = SliceWidget(controller)
        self._status = QLabel(S.STATUS[State.EMPTY])
        self._plane = QComboBox()
        for plane in sr.PLANES:
            self._plane.addItem(S.PLANE.get(plane, plane), plane)
        self._plane.currentIndexChanged.connect(self._on_plane)

        settings_btn = QPushButton(S.BTN["settings"])
        settings_btn.clicked.connect(self._on_settings)
        open_btn = QPushButton(S.BTN["open"])
        open_btn.clicked.connect(self._on_open)
        self._segment_btn = QPushButton(S.BTN["segment_cloud"])
        self._segment_btn.clicked.connect(self._on_segment)
        # Local (on-device) segmentation: only offered when an engine is wired,
        # so there is no network round-trip for the large study upload.
        self._local_chk = QCheckBox(S.BTN["local_checkbox"])
        self._local_chk.setEnabled(self._c.local_available)
        self._local_chk.setToolTip(
            S.TIP["local_on"] if self._c.local_available else S.TIP["local_off"]
        )
        self._local_chk.toggled.connect(self._on_local_toggled)
        # First-run install of the engine-pack: shown only until an engine is
        # wired, so the checkbox above can actually be enabled on a slim client.
        self._install_btn = QPushButton(S.BTN["install_engine"])
        self._install_btn.setToolTip(S.TIP["install_engine"])
        self._install_btn.setVisible(not self._c.local_available)
        self._install_btn.clicked.connect(lambda: self._start_engine_install(clean=False))
        # Engine update: hidden until the startup check finds a newer engine-pack.
        self._update_btn = QPushButton(S.BTN["update_engine"])
        self._update_btn.setToolTip(S.TIP["update_engine"])
        self._update_btn.setVisible(False)
        self._update_btn.clicked.connect(lambda: self._start_engine_install(clean=True))
        # Prominent, persistent warning banner across the top of the window: a
        # plain side button is too easy to miss, so an outdated engine (which
        # silently disables the live progress %) must announce itself loudly and
        # stay up until the update is actually applied.
        self._update_banner = self._build_update_banner()
        self._update_banner.setVisible(False)
        self._measure_btn = QPushButton(S.BTN["measure"])
        self._measure_btn.setCheckable(True)
        self._measure_btn.toggled.connect(self.slice_widget.set_measure_mode)
        clear_btn = QPushButton(S.BTN["clear_measure"])
        clear_btn.clicked.connect(self._on_clear_measurements)
        png_btn = QPushButton(S.BTN["save_png"])
        png_btn.clicked.connect(self._on_save_png)
        mask_btn = QPushButton(S.BTN["save_mask"])
        mask_btn.clicked.connect(self._on_save_mask)

        self.mask_list = QListWidget()
        self.mask_list.itemChanged.connect(self._on_mask_item_changed)

        left = QVBoxLayout()
        left.addWidget(settings_btn)
        left.addWidget(_separator())
        left.addWidget(_section(S.SECTION["study"]))
        left.addWidget(open_btn)
        left.addWidget(_section(S.SECTION["segmentation"]))
        for w in (self._local_chk, self._segment_btn, self._install_btn, self._update_btn):
            left.addWidget(w)
        left.addWidget(_section(S.SECTION["tools"]))
        for w in (self._plane, self._measure_btn, clear_btn):
            left.addWidget(w)
        left.addWidget(_section(S.SECTION["export"]))
        for w in (png_btn, mask_btn):
            left.addWidget(w)
        left.addWidget(_section(S.SECTION["masks"]))
        left.addWidget(self.mask_list)
        left.addWidget(_separator())
        left.addWidget(self._status)
        left.addStretch(1)
        left_panel = QWidget()
        left_panel.setLayout(left)

        body = QHBoxLayout()
        body.addWidget(left_panel, 0)
        body.addWidget(self.slice_widget, 1)
        root = QVBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._update_banner)  # full-width, above the panel + view
        root.addLayout(body, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._on_poll_tick)
        self._thread: QThread | None = None
        self._install_thread: QThread | None = None
        self._install_progress: QProgressDialog | None = None
        self._check_threads: list[QThread] = []
        self._dl_thread: QThread | None = None
        self._dl_progress: QProgressDialog | None = None

    def refresh(self) -> None:
        self._status.setText(S.STATUS.get(self._c.state, self._c.state.value))
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

    def _on_plane(self, _index: int) -> None:
        plane = self._plane.currentData()
        if plane is not None and self._c.volume is not None:
            self._c.set_plane(plane)
            self.refresh()

    def _on_open(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, S.BTN["open"])
        if not directory:
            return
        try:
            volume = DicomLoader.load(directory)
        except DicomError as exc:
            QMessageBox.critical(self, S.MSG["dicom_error_title"], str(exc))
            return
        self._c.load_volume(volume)
        self.refresh()

    def _on_clear_measurements(self) -> None:
        self._c.clear_measurements()
        self.slice_widget.update()

    def _on_save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, S.BTN["save_png"], "slice.png", "PNG (*.png)")
        if path:
            self.slice_widget.render_image().save(path, "PNG")

    def _on_save_mask(self) -> None:
        if self._c.mask is None:
            QMessageBox.information(self, S.MSG["no_mask_title"], S.MSG["no_mask_body"])
            return
        path, _ = QFileDialog.getSaveFileName(self, S.BTN["save_mask"], "mask.nii.gz", "NIfTI (*.nii.gz)")
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
        self._segment_btn.setText(
            S.BTN["segment_local"] if checked else S.BTN["segment_cloud"]
        )

    # --- engine install / update --------------------------------------------

    def _build_update_banner(self) -> QWidget:
        """A loud, full-width warning bar urging the user to update the engine."""
        banner = QFrame()
        banner.setObjectName("updateBanner")
        banner.setStyleSheet(
            "#updateBanner { background: #FFF3CD; border: 1px solid #FFE69C; }"
            "#updateBanner QLabel { color: #664D03; }"
        )
        label = QLabel(S.MSG["engine_update_banner"])
        label.setWordWrap(True)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        button = QPushButton(S.BTN["update_engine"])
        button.clicked.connect(lambda: self._start_engine_install(clean=True))
        layout = QHBoxLayout(banner)
        layout.addWidget(label, 1)
        layout.addWidget(button, 0)
        return banner

    def _set_engine_update_available(self, available: bool) -> None:
        """Show/hide both the side button and the top banner together."""
        self._update_btn.setVisible(available)
        self._update_banner.setVisible(available)

    def _start_engine_install(self, *, clean: bool) -> None:
        if self._install_thread is not None and self._install_thread.isRunning():
            return
        from ..local_engine import provision_local_engine

        def _provision(progress):
            return provision_local_engine(progress=progress, clean=clean)

        self._install_btn.setEnabled(False)
        self._update_btn.setEnabled(False)
        self._install_progress = QProgressDialog(
            S.MSG["update_progress"] if clean else S.MSG["install_progress"],
            None, 0, 100, self,
        )
        self._install_progress.setWindowTitle(
            S.MSG["update_title"] if clean else S.MSG["install_title"]
        )
        self._install_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._install_progress.setAutoClose(True)
        self._install_progress.setMinimumDuration(0)
        self._install_progress.setValue(0)

        # Remembered so the (UI-thread) done handler knows install vs update.
        # Must NOT ride on a context-less lambda: connecting ``done`` to a bare
        # lambda would run the slot in the *worker* thread, and creating the
        # success QMessageBox there yields a blank, frozen "(Не отвечает)" window.
        self._installing_clean = clean
        self._install_thread = QThread(self)
        worker = _InstallWorker(_provision)
        worker.moveToThread(self._install_thread)
        self._install_thread.started.connect(worker.run)
        worker.progress.connect(self._on_install_progress)
        worker.done.connect(self._on_install_done)
        worker.failed.connect(self._on_install_failed)
        worker.done.connect(self._install_thread.quit)
        worker.failed.connect(self._install_thread.quit)
        self._install_worker = worker
        self._install_thread.start()

    # Backwards-compatible alias for the original first-run entry point.
    def _on_install_engine(self) -> None:
        self._start_engine_install(clean=False)

    def _on_install_progress(self, done: int, total: int) -> None:
        if total > 0:
            self._install_progress.setValue(min(100, done * 100 // total))

    def _on_install_done(self, engine: object, *, updated: bool | None = None) -> None:
        # ``updated`` arrives explicitly from tests; from the worker signal it is
        # None, so fall back to the flag stashed by _start_engine_install.
        if updated is None:
            updated = getattr(self, "_installing_clean", False)
        if self._install_progress is not None:
            self._install_progress.reset()
        self._install_btn.setEnabled(True)
        self._install_btn.setVisible(False)
        self._update_btn.setEnabled(True)
        self._set_engine_update_available(False)
        self._c.set_engine(engine)
        self._local_chk.setEnabled(True)
        self._local_chk.setToolTip(S.TIP["local_on"])
        self._local_chk.setChecked(True)
        QMessageBox.information(
            self, S.MSG["install_done_title"],
            S.MSG["update_done_body"] if updated else S.MSG["install_done_body"],
        )

    def _on_install_failed(self, message: str) -> None:
        if self._install_progress is not None:
            self._install_progress.reset()
        self._install_btn.setEnabled(True)
        self._update_btn.setEnabled(True)
        QMessageBox.critical(self, S.MSG["install_failed_title"], message)

    # --- startup update checks ----------------------------------------------

    def check_for_updates(self) -> None:
        """Kick off fail-soft engine + client update checks in the background.

        Called from the app entry point after the window is shown — never from
        ``__init__`` so headless construction (tests) does no network I/O.
        """
        from ..local_engine import engine_update_available
        from ..updates import check_for_client_update

        self._run_check(engine_update_available, self._on_engine_update_checked)
        self._run_check(check_for_client_update, self._on_client_update_checked)

    def _run_check(self, fn, slot) -> None:
        thread = QThread(self)
        worker = _FnWorker(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(slot)
        worker.done.connect(thread.quit)
        # Keep references alive until the thread finishes. Use a bound method
        # (not a context-less lambda) so the cleanup is marshalled to the UI
        # thread rather than run in the worker thread that emits ``finished``.
        thread.finished.connect(self._on_check_thread_finished)
        self._check_threads.append(thread)
        self._check_worker = worker
        thread.start()

    def _on_check_thread_finished(self) -> None:
        thread = self.sender()
        if thread in self._check_threads:
            self._check_threads.remove(thread)

    def _on_engine_update_checked(self, available: object) -> None:
        if not available:
            return
        self._set_engine_update_available(True)
        if QMessageBox.question(
            self, S.MSG["engine_update_title"], S.MSG["engine_update_body"]
        ) == QMessageBox.StandardButton.Yes:
            self._start_engine_install(clean=True)

    def _on_client_update_checked(self, release: object) -> None:
        if not release:
            return
        if QMessageBox.question(
            self, S.MSG["client_update_title"],
            S.MSG["client_update_body"].format(version=release.get("version", "")),
        ) == QMessageBox.StandardButton.Yes:
            self._start_client_update(release)

    def _start_client_update(self, release: dict) -> None:
        if self._dl_thread is not None and self._dl_thread.isRunning():
            return
        import tempfile

        dest = Path(tempfile.gettempdir()) / "StomClientSetup.exe"
        self._dl_progress = QProgressDialog(
            S.MSG["client_update_progress"], None, 0, 100, self
        )
        self._dl_progress.setWindowTitle(S.MSG["client_update_title"])
        self._dl_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._dl_progress.setMinimumDuration(0)
        self._dl_progress.setValue(0)

        self._dl_thread = QThread(self)
        worker = _DownloadWorker(release["url"], dest)
        worker.moveToThread(self._dl_thread)
        self._dl_thread.started.connect(worker.run)
        worker.progress.connect(self._on_dl_progress)
        worker.done.connect(self._on_client_downloaded)
        worker.failed.connect(self._on_client_dl_failed)
        worker.done.connect(self._dl_thread.quit)
        worker.failed.connect(self._dl_thread.quit)
        self._dl_worker = worker
        self._dl_thread.start()

    def _on_dl_progress(self, done: int, total: int) -> None:
        if self._dl_progress is not None and total > 0:
            self._dl_progress.setValue(min(100, done * 100 // total))

    def _on_client_downloaded(self, path: str) -> None:
        from ..updates import launch_installer

        if self._dl_progress is not None:
            self._dl_progress.reset()
        QMessageBox.information(
            self, S.MSG["client_update_ready_title"], S.MSG["client_update_ready_body"]
        )
        launch_installer(Path(path))
        self.close()

    def _on_client_dl_failed(self, message: str) -> None:
        if self._dl_progress is not None:
            self._dl_progress.reset()
        QMessageBox.critical(self, S.MSG["client_update_failed_title"], message)

    # --- segmentation --------------------------------------------------------

    def _on_segment(self) -> None:
        if not self._c.local and not self._config.server_url:
            QMessageBox.information(self, S.MSG["no_server_title"], S.MSG["no_server_body"])
            return
        if self._c.volume is None:
            QMessageBox.information(self, S.MSG["no_study_title"], S.MSG["no_study_body"])
            return
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = QThread(self)
        worker = _SubmitWorker(self._c)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(self._on_seg_progress)
        worker.done.connect(self._on_submitted)
        worker.failed.connect(self._on_submit_failed)
        worker.done.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
        self._worker = worker
        self._thread.start()
        self.refresh()

    def _on_seg_progress(self, done: int, total: int) -> None:
        """Show live inference progress as a percentage in the status label."""
        if total > 0:
            pct = min(100, done * 100 // total)
            self._status.setText(S.STATUS_SEG_PROGRESS.format(pct=pct))

    def _on_submitted(self) -> None:
        self.refresh()
        self._poll_timer.start()

    def _on_submit_failed(self, message: str) -> None:
        if self._c.local:
            self._show_local_engine_error(message)
        else:
            QMessageBox.critical(self, S.MSG["cloud_error_title"], message)
        self.refresh()

    def _show_local_engine_error(self, message: str) -> None:
        """Friendly error for a failed local run: short text + collapsible details.

        If the failure looks like the pre-freeze_support engine-pack (the bug that
        hung/crashed inference), point the user at "Обновить движок…".
        """
        outdated = "freeze_support" in message or "bootstrapping phase" in message
        short = S.MSG["seg_error_title"]
        if outdated:
            short += S.MSG["engine_outdated_hint"]
            self._set_engine_update_available(True)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(S.MSG["seg_error_title"])
        box.setText(short)
        box.setDetailedText(message)
        box.exec()

    def _on_poll_tick(self) -> None:
        try:
            terminal = self._c.poll()
        except CloudError as exc:
            self._poll_timer.stop()
            QMessageBox.critical(self, S.MSG["cloud_error_title"], str(exc))
            return
        if terminal:
            self._poll_timer.stop()
            if self._c.state == State.FAILED:
                QMessageBox.warning(self, S.MSG["seg_failed_title"], self._c.error or "")
        self.refresh()
