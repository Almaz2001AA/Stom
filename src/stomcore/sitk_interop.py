"""The single bridge between stomcore types and SimpleITK images."""

from __future__ import annotations

import SimpleITK as sitk

from .geometry import Geometry
from .volume import Volume


def geometry_from_sitk(image: sitk.Image) -> Geometry:
    return Geometry(
        spacing=tuple(float(v) for v in image.GetSpacing()),
        origin=tuple(float(v) for v in image.GetOrigin()),
        direction=tuple(float(v) for v in image.GetDirection()),
    )


def volume_from_sitk(image: sitk.Image) -> Volume:
    voxels = sitk.GetArrayFromImage(image)  # [z, y, x]
    return Volume(voxels, geometry_from_sitk(image))


def sitk_from_volume(volume: Volume) -> sitk.Image:
    image = sitk.GetImageFromArray(volume.voxels)  # consumes [z, y, x]
    image.SetSpacing(volume.geometry.spacing)
    image.SetOrigin(volume.geometry.origin)
    image.SetDirection(volume.geometry.direction)
    return image
