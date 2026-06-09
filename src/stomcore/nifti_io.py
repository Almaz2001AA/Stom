"""Read/write Volume as NIfTI (.nii.gz) — the client<->cloud exchange format."""

from __future__ import annotations

import os

import SimpleITK as sitk

from .sitk_interop import sitk_from_volume, volume_from_sitk
from .volume import Volume


def save_volume_nifti(volume: Volume, path: str | os.PathLike) -> None:
    sitk.WriteImage(sitk_from_volume(volume), str(path), useCompression=True)


def load_volume_nifti(path: str | os.PathLike) -> Volume:
    return volume_from_sitk(sitk.ReadImage(str(path)))
