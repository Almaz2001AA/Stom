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


def test_main_window_apply_config_sets_cloud(qapp, monkeypatch):
    from stomclient.config import ClientConfig
    from stomclient.ui.main_window import MainWindow

    monkeypatch.setattr("stomclient.ui.main_window.save", lambda *a, **k: None)
    controller = AppController(cloud_client=None)
    window = MainWindow(controller)
    window._apply_config(ClientConfig(server_url="http://x", token="t"))
    assert window._config.server_url == "http://x"
    assert controller._cloud is not None
