import numpy as np
import SimpleITK as sitk

from stomcore.geometry import Geometry
from stomcore.sitk_interop import (
    geometry_from_sitk,
    sitk_from_volume,
    volume_from_sitk,
)
from stomcore.volume import Volume


def _sitk_image():
    arr = np.arange(4 * 3 * 2, dtype=np.int16).reshape(4, 3, 2)  # [z, y, x]
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.3, 0.4, 0.5))  # (x, y, z)
    img.SetOrigin((1.0, 2.0, 3.0))
    return img


def test_geometry_from_sitk_reads_spacing_origin_direction():
    geo = geometry_from_sitk(_sitk_image())
    assert geo.spacing == (0.3, 0.4, 0.5)
    assert geo.origin == (1.0, 2.0, 3.0)
    assert len(geo.direction) == 9


def test_volume_from_sitk_preserves_voxels_in_zyx():
    vol = volume_from_sitk(_sitk_image())
    assert vol.shape == (4, 3, 2)
    assert vol.voxels[0, 0, 0] == 0
    assert vol.voxels[3, 2, 1] == 23


def test_round_trip_volume_through_sitk():
    geo = Geometry(spacing=(0.3, 0.4, 0.5), origin=(1.0, 2.0, 3.0),
                   direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    vox = np.arange(4 * 3 * 2, dtype=np.int16).reshape(4, 3, 2)
    vol = Volume(vox, geo)
    restored = volume_from_sitk(sitk_from_volume(vol))
    assert restored == vol
