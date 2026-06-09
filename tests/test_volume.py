import numpy as np
import pytest

from stomcore.geometry import Geometry
from stomcore.volume import Volume


def _voxels(shape=(4, 3, 2)):
    return np.arange(np.prod(shape), dtype=np.int16).reshape(shape)


def test_volume_exposes_voxels_and_geometry():
    vox = _voxels()
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    vol = Volume(vox, geo)
    assert vol.geometry is geo
    np.testing.assert_array_equal(vol.voxels, vox)


def test_shape_is_zyx_voxel_shape():
    vol = Volume(_voxels((4, 3, 2)), Geometry.identity(spacing=(1.0, 1.0, 1.0)))
    assert vol.shape == (4, 3, 2)


def test_rejects_non_3d_voxels():
    geo = Geometry.identity(spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        Volume(np.zeros((4, 4)), geo)


def test_equality_compares_voxels_and_geometry():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    a = Volume(_voxels(), geo)
    b = Volume(_voxels(), geo)
    assert a == b


def test_inequality_when_voxels_differ():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    a = Volume(_voxels(), geo)
    diff = _voxels()
    diff[0, 0, 0] += 1
    b = Volume(diff, geo)
    assert a != b
