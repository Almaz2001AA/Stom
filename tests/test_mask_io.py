import numpy as np

from stomcore.geometry import Geometry
from stomcore.mask import LabelInfo, SegmentationMask
from stomcore.mask_io import load_mask_nifti, save_mask_nifti


def _mask():
    geo = Geometry(spacing=(0.3, 0.4, 0.5), origin=(1.0, 2.0, 3.0),
                   direction=(0, -1, 0, 1, 0, 0, 0, 0, 1))  # non-identity
    labels = np.zeros((5, 4, 3), dtype=np.uint16)
    labels[0, 0, 0] = 2
    labels[1, 1, 1] = 5
    label_map = {
        2: LabelInfo(2, "mandible", (200, 170, 130)),
        5: LabelInfo(5, "mandibular-canal", (220, 80, 80), visible=False),
    }
    return SegmentationMask(labels, geo, label_map)


def test_save_then_load_round_trips_mask(tmp_path):
    mask = _mask()
    nifti = tmp_path / "mask.nii.gz"
    labels = tmp_path / "mask_labels.json"
    save_mask_nifti(mask, nifti, labels)
    assert nifti.exists() and labels.exists()

    restored = load_mask_nifti(nifti, labels)
    assert restored.shape == mask.shape
    np.testing.assert_array_equal(restored.labels, mask.labels)
    assert restored.geometry.is_compatible(mask.geometry)
    assert restored.present_labels() == {2, 5}
    assert restored.label_map[5].name == "mandibular-canal"
    assert restored.label_map[5].color == (220, 80, 80)
    assert restored.label_map[5].visible is False
