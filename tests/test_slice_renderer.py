import numpy as np

from stomcore.geometry import Geometry
from stomcore.mask import LabelInfo, SegmentationMask
from stomcore.volume import Volume
from stomclient import slice_renderer as sr


def _volume():
    voxels = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)  # [z, y, x]
    return Volume(voxels, Geometry.identity(spacing=(1, 1, 1)))


def test_slice_count_per_plane():
    v = _volume()
    assert sr.slice_count(v, sr.AXIAL) == 2
    assert sr.slice_count(v, sr.CORONAL) == 3
    assert sr.slice_count(v, sr.SAGITTAL) == 4


def test_slice_array_axial_shape_is_y_by_x():
    v = _volume()
    assert sr.slice_array(v.voxels, sr.AXIAL, 0).shape == (3, 4)
    assert sr.slice_array(v.voxels, sr.CORONAL, 0).shape == (2, 4)
    assert sr.slice_array(v.voxels, sr.SAGITTAL, 0).shape == (2, 3)


def test_apply_window_level_maps_to_uint8_range():
    s = np.array([[0, 50, 100]], dtype=np.int16)
    out = sr.apply_window_level(s, center=50, width=100)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0
    assert out[0, 2] == 255
    assert 120 <= out[0, 1] <= 135  # midpoint


def test_composite_overlay_colors_visible_label_only():
    gray = np.zeros((1, 2), dtype=np.uint8)
    mask_slice = np.array([[1, 2]], dtype=np.uint16)
    label_map = {
        1: LabelInfo(1, "a", (255, 0, 0), visible=True),
        2: LabelInfo(2, "b", (0, 255, 0), visible=False),
    }
    out = sr.composite_overlay(gray, mask_slice, label_map, alpha=1.0)
    assert tuple(out[0, 0]) == (255, 0, 0)   # visible label painted
    assert tuple(out[0, 1]) == (0, 0, 0)     # hidden label untouched


def test_default_window_level_spans_data_range():
    v = _volume()  # values 0..23
    center, width = sr.default_window_level(v)
    assert width >= 1
    assert center == 11.5
