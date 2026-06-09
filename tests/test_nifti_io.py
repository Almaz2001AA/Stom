import numpy as np

from stomcore.geometry import Geometry
from stomcore.nifti_io import load_volume_nifti, save_volume_nifti
from stomcore.volume import Volume


def _volume():
    geo = Geometry(spacing=(0.3, 0.4, 0.5), origin=(1.0, 2.0, 3.0),
                   direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    vox = np.arange(5 * 4 * 3, dtype=np.int16).reshape(5, 4, 3)
    return Volume(vox, geo)


def test_save_then_load_round_trips_volume(tmp_path):
    vol = _volume()
    path = tmp_path / "study.nii.gz"
    save_volume_nifti(vol, path)
    assert path.exists()
    restored = load_volume_nifti(path)
    assert restored.shape == vol.shape
    np.testing.assert_array_equal(restored.voxels, vol.voxels)
    assert restored.geometry.is_compatible(vol.geometry)


def test_accepts_string_paths(tmp_path):
    vol = _volume()
    path = str(tmp_path / "study.nii.gz")
    save_volume_nifti(vol, path)
    restored = load_volume_nifti(path)
    assert restored.shape == vol.shape
