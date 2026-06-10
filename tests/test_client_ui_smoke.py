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
