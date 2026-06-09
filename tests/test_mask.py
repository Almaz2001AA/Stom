import numpy as np
import pytest

from stomcore.geometry import Geometry
from stomcore.mask import LabelInfo, SegmentationMask
from stomcore.volume import Volume


def _label_map():
    return {
        11: LabelInfo(label_id=11, name="tooth-11", color=(255, 0, 0)),
        12: LabelInfo(label_id=12, name="tooth-12", color=(0, 255, 0), visible=False),
    }


def _labels(shape=(4, 3, 2)):
    arr = np.zeros(shape, dtype=np.uint16)
    arr[0, 0, 0] = 11
    arr[1, 1, 1] = 12
    return arr


def test_labelinfo_defaults_to_visible():
    info = LabelInfo(label_id=11, name="tooth-11", color=(255, 0, 0))
    assert info.visible is True


def test_mask_exposes_geometry_and_label_map():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    mask = SegmentationMask(_labels(), geo, _label_map())
    assert mask.geometry is geo
    assert mask.label_map[12].visible is False


def test_rejects_non_3d_labels():
    geo = Geometry.identity(spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        SegmentationMask(np.zeros((4, 4), dtype=np.uint16), geo, {})


def test_present_labels_excludes_background():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    mask = SegmentationMask(_labels(), geo, _label_map())
    assert mask.present_labels() == {11, 12}


def test_compatible_with_volume_when_shape_and_geometry_match():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    vol = Volume(np.zeros((4, 3, 2), dtype=np.int16), geo)
    mask = SegmentationMask(_labels((4, 3, 2)), geo, _label_map())
    assert mask.is_compatible_with(vol) is True


def test_incompatible_with_volume_when_shape_differs():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    vol = Volume(np.zeros((4, 3, 2), dtype=np.int16), geo)
    mask = SegmentationMask(_labels((2, 3, 4)), geo, _label_map())
    assert mask.is_compatible_with(vol) is False


def test_incompatible_with_volume_when_geometry_differs():
    vol = Volume(np.zeros((4, 3, 2), dtype=np.int16), Geometry.identity(spacing=(0.3, 0.3, 0.3)))
    mask = SegmentationMask(_labels((4, 3, 2)), Geometry.identity(spacing=(0.4, 0.3, 0.3)), _label_map())
    assert mask.is_compatible_with(vol) is False
