import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from stomcore.geometry import Geometry
from stomcore.volume import Volume
from stomclient.app_controller import AppController


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_ndarray_to_qimage_dimensions(qapp):
    from stomclient.ui.qt_image import ndarray_to_qimage

    rgb = np.zeros((5, 7, 3), dtype=np.uint8)
    img = ndarray_to_qimage(rgb)
    assert img.width() == 7
    assert img.height() == 5


def test_slice_widget_renders_loaded_volume(qapp):
    from stomclient.ui.slice_widget import SliceWidget

    controller = AppController(cloud_client=None)
    controller.load_volume(
        Volume(np.zeros((4, 5, 6), dtype=np.int16), Geometry.identity((0.3, 0.3, 0.3)))
    )
    widget = SliceWidget(controller)
    img = widget.render_image()
    assert img.width() == 6   # axial -> x columns
    assert img.height() == 5  # axial -> y rows


def test_settings_dialog_values_roundtrip(qapp):
    from stomclient.config import ClientConfig
    from stomclient.ui.settings_dialog import SettingsDialog

    dialog = SettingsDialog(ClientConfig(server_url="https://api", token="tok"))
    cfg = dialog.values()
    assert cfg.server_url == "https://api"
    assert cfg.token == "tok"


def test_main_window_builds_with_controller(qapp):
    from stomclient.app_controller import AppController
    from stomclient.ui.main_window import MainWindow

    window = MainWindow(AppController(cloud_client=None))
    assert "Stom" in window.windowTitle()
    assert window.slice_widget is not None


def test_slice_widget_measure_mode_toggle(qapp):
    from stomclient.ui.slice_widget import SliceWidget

    controller = AppController(cloud_client=None)
    controller.load_volume(
        Volume(np.zeros((4, 5, 6), dtype=np.int16), Geometry.identity((0.3, 0.3, 0.3)))
    )
    widget = SliceWidget(controller)
    widget.set_measure_mode(True)
    assert widget.measure_mode is True


def test_main_window_mask_list_populates(qapp):
    from stomcore.mask import LabelInfo, SegmentationMask
    from stomclient.ui.main_window import MainWindow

    controller = AppController(cloud_client=None)
    geo = Geometry.identity((0.3, 0.3, 0.3))
    controller.load_volume(Volume(np.zeros((4, 5, 6), dtype=np.int16), geo))
    controller.mask = SegmentationMask(
        np.zeros((4, 5, 6), dtype=np.uint16), geo,
        {1: LabelInfo(1, "tooth", (255, 0, 0), True),
         2: LabelInfo(2, "canal", (0, 255, 0), True)},
    )
    window = MainWindow(controller)
    window.refresh()
    assert window.mask_list.count() == 2


def test_save_stl_guard_when_no_mask(qapp, monkeypatch):
    from stomclient.ui.main_window import MainWindow

    seen = {}
    monkeypatch.setattr("stomclient.ui.main_window.QMessageBox.information",
                        lambda *a, **k: seen.setdefault("info", True))
    # Must NOT reach the directory picker when there is no mask.
    monkeypatch.setattr("stomclient.ui.main_window.QFileDialog.getExistingDirectory",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("picked dir")))
    window = MainWindow(AppController(cloud_client=None))
    window._on_save_stl()
    assert seen.get("info") is True


def test_stl_export_worker_writes_one_file_per_visible_label(qapp, tmp_path):
    from stomcore.mask import LabelInfo, SegmentationMask
    from stomclient.ui.main_window import _StlExportWorker

    labels = np.zeros((5, 5, 5), dtype=np.uint8)
    labels[1, 1, 1] = 3
    labels[3, 3, 3] = 4
    geo = Geometry.identity((0.3, 0.3, 0.3))
    mask = SegmentationMask(labels, geo,
                            {3: LabelInfo(3, "Upper Teeth", (1, 1, 1)),
                             4: LabelInfo(4, "Lower Teeth", (2, 2, 2))})
    worker = _StlExportWorker(mask, tmp_path, [3, 4], 12)  # smoothing on
    result = {}
    worker.done.connect(lambda count, folder: result.update(count=count, folder=folder))
    worker.run()  # synchronous: signals deliver directly on this thread
    assert result["count"] == 2
    assert sorted(p.name for p in tmp_path.glob("*.stl")) == [
        "03_Upper_Teeth.stl", "04_Lower_Teeth.stl"]


def test_stl_smoothing_checkbox_defaults_on(qapp):
    from stomclient.ui.main_window import MainWindow

    window = MainWindow(AppController(cloud_client=None))
    assert window._stl_smooth_chk.isChecked() is True


def test_stl_teeth_only_checkbox_defaults_off(qapp):
    from stomclient.ui.main_window import MainWindow

    window = MainWindow(AppController(cloud_client=None))
    assert window._stl_teeth_only_chk.isChecked() is False


def _mixed_tf2_mask():
    from stomcore.mask import LabelInfo, SegmentationMask

    geo = Geometry.identity((0.3, 0.3, 0.3))
    labels = np.zeros((6, 6, 6), dtype=np.uint16)
    labels[0, 0, 0] = 1    # Lower Jawbone (anatomy)
    labels[2, 2, 2] = 16   # Upper Right First Molar (tooth)
    labels[4, 4, 4] = 38   # Lower Left Third Molar (tooth)
    label_map = {
        1: LabelInfo(1, "Lower Jawbone", (1, 1, 1), True),
        16: LabelInfo(16, "Upper Right First Molar", (2, 2, 2), True),
        38: LabelInfo(38, "Lower Left Third Molar", (3, 3, 3), True),
    }
    return SegmentationMask(labels, geo, label_map)


def test_stl_teeth_only_filters_to_fdi_teeth(qapp):
    from stomclient.ui.main_window import MainWindow

    controller = AppController(cloud_client=None)
    controller.mask = _mixed_tf2_mask()
    window = MainWindow(controller)

    window._stl_teeth_only_chk.setChecked(False)
    assert window._selected_stl_label_ids() == [1, 16, 38]  # all visible structures

    window._stl_teeth_only_chk.setChecked(True)
    assert window._selected_stl_label_ids() == [16, 38]  # anatomy (1) dropped


def test_main_window_segment_guard_when_thread_running(qapp):
    from stomclient.ui.main_window import MainWindow

    controller = AppController(cloud_client=None)
    controller.load_volume(
        Volume(np.zeros((4, 5, 6), dtype=np.int16), Geometry.identity((0.3, 0.3, 0.3)))
    )
    window = MainWindow(controller)
    # Configure a server so _on_segment passes the no-server guard (which would
    # otherwise pop a modal QMessageBox and block headlessly) and reaches the
    # thread-running guard under test.
    from stomclient.config import ClientConfig

    window._config = ClientConfig(server_url="http://x", token="t")

    class _FakeRunningThread:
        def isRunning(self):
            return True

    window._thread = _FakeRunningThread()
    window._worker = "sentinel"
    window._on_segment()  # must early-return because a thread is "running"
    assert window._worker == "sentinel"  # worker not replaced


def test_install_button_offered_then_enables_local(qapp, monkeypatch):
    from stomclient.ui.main_window import MainWindow

    # The success path pops an info box; no-op it so the test stays headless.
    monkeypatch.setattr("stomclient.ui.main_window.QMessageBox.information",
                        lambda *a, **k: None)
    controller = AppController(cloud_client=None)  # slim first run: no engine
    window = MainWindow(controller)
    # Install offered, local checkbox disabled until an engine is wired.
    assert window._install_btn.isHidden() is False
    assert window._local_chk.isEnabled() is False

    class _Eng:
        def segment(self, volume):  # pragma: no cover - not exercised here
            return None

    window._on_install_done(_Eng())  # simulate a finished download
    assert controller.local_available is True
    assert window._local_chk.isEnabled() is True
    assert window._install_btn.isHidden() is True  # no longer offered
    assert controller.local is True                # auto-selected local mode


def test_install_done_uses_stored_clean_flag(qapp, monkeypatch):
    """The worker's ``done`` signal calls ``_on_install_done(engine)`` with no
    ``updated`` kwarg, so the handler must read the install-vs-update mode from
    the flag stashed by ``_start_engine_install`` — not default to install.

    This guards the fix that replaced a context-less ``lambda`` (which ran the
    slot, and built its QMessageBox, in the worker thread → blank frozen window)
    with a bound-method connection that marshals to the UI thread.
    """
    from stomclient.ui import strings as S
    from stomclient.ui.main_window import MainWindow

    bodies = []
    monkeypatch.setattr("stomclient.ui.main_window.QMessageBox.information",
                        lambda parent, title, body, *a, **k: bodies.append(body))
    controller = AppController(cloud_client=None)
    window = MainWindow(controller)
    window._set_engine_update_available(True)     # show the update banner
    assert window._update_banner.isHidden() is False

    class _Eng:
        pass

    window._installing_clean = True               # as _start_engine_install(clean=True) would
    window._on_install_done(_Eng())               # signal arrives with no updated kwarg
    assert window._update_banner.isHidden() is True  # update path taken
    assert bodies == [S.MSG["update_done_body"]]      # update (not install) message


def test_main_window_apply_config_sets_cloud(qapp, monkeypatch):
    from stomclient.config import ClientConfig
    from stomclient.ui.main_window import MainWindow

    monkeypatch.setattr("stomclient.ui.main_window.save", lambda *a, **k: None)
    controller = AppController(cloud_client=None)
    window = MainWindow(controller)
    window._apply_config(ClientConfig(server_url="http://x", token="t"))
    assert window._config.server_url == "http://x"
    assert controller._cloud is not None


def test_status_label_is_russian(qapp):
    from stomclient.ui import strings as S
    from stomclient.ui.main_window import MainWindow

    controller = AppController(cloud_client=None)
    window = MainWindow(controller)
    window.refresh()
    assert window._status.text() == S.STATUS[controller.state]
    assert window._status.text() == "Нет исследования"


def test_plane_combo_maps_russian_label_to_plane_id(qapp):
    from stomclient.ui.main_window import MainWindow

    controller = AppController(cloud_client=None)
    controller.load_volume(
        Volume(np.zeros((4, 5, 6), dtype=np.int16), Geometry.identity((0.3, 0.3, 0.3)))
    )
    window = MainWindow(controller)
    # Combo shows Russian labels but carries the renderer's plane id as data.
    window._plane.setCurrentIndex(window._plane.findText("Корональная"))
    assert controller.plane == "coronal"


def test_engine_update_check_reveals_update_button(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from stomclient.ui.main_window import MainWindow

    # User declines the offered update; the button must remain for later.
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.No)
    window = MainWindow(AppController(cloud_client=None))
    assert window._update_btn.isHidden() is True
    window._on_engine_update_checked(True)
    assert window._update_btn.isHidden() is False


def test_engine_update_check_no_op_when_current(qapp):
    from stomclient.ui.main_window import MainWindow

    window = MainWindow(AppController(cloud_client=None))
    window._on_engine_update_checked(False)
    assert window._update_btn.isHidden() is True


def test_engine_update_shows_prominent_banner(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from stomclient.ui.main_window import MainWindow

    # User declines the modal prompt; the persistent banner must stay up so the
    # outdated engine (no progress %) cannot be silently ignored.
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.No)
    window = MainWindow(AppController(cloud_client=None))
    assert window._update_banner.isHidden() is True
    window._on_engine_update_checked(True)
    assert window._update_banner.isHidden() is False
    assert window._update_btn.isHidden() is False


def test_engine_update_banner_hidden_after_install(qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from stomclient.ui.main_window import MainWindow

    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.No)
    monkeypatch.setattr("stomclient.ui.main_window.QMessageBox.information",
                        lambda *a, **k: None)
    window = MainWindow(AppController(cloud_client=None))
    window._on_engine_update_checked(True)
    assert window._update_banner.isHidden() is False

    class _Eng:
        pass

    window._on_install_done(_Eng(), updated=True)  # finished update clears it
    assert window._update_banner.isHidden() is True
    assert window._update_btn.isHidden() is True


def test_local_error_detects_outdated_engine(qapp, monkeypatch):
    from stomclient.ui.main_window import MainWindow

    # The error dialog is modal; intercept exec so the test stays headless.
    monkeypatch.setattr("stomclient.ui.main_window.QMessageBox.exec",
                        lambda self: None, raising=False)
    controller = AppController(cloud_client=None)
    window = MainWindow(controller)
    window._show_local_engine_error(
        "local engine failed: RuntimeError ... freeze_support()"
    )
    # A freeze_support failure means a stale pack -> offer the update button.
    assert window._update_btn.isHidden() is False
