"""Bridge Volume/SegmentationMask to .nii.gz bytes via stomcore I/O."""

from __future__ import annotations

import tempfile
from pathlib import Path

from stomcore.mask import SegmentationMask
from stomcore.mask_io import load_mask_nifti, save_mask_nifti
from stomcore.nifti_io import save_volume_nifti
from stomcore.volume import Volume


def volume_to_nifti_bytes(volume: Volume) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "volume.nii.gz"
        save_volume_nifti(volume, path)
        return path.read_bytes()


def mask_to_bytes(mask: SegmentationMask) -> tuple[bytes, bytes]:
    with tempfile.TemporaryDirectory() as tmp:
        nifti = Path(tmp) / "mask.nii.gz"
        labels = Path(tmp) / "labels.json"
        save_mask_nifti(mask, nifti, labels)
        return nifti.read_bytes(), labels.read_bytes()


def mask_from_bytes(mask_bytes: bytes, labels_bytes: bytes) -> SegmentationMask:
    with tempfile.TemporaryDirectory() as tmp:
        nifti = Path(tmp) / "mask.nii.gz"
        labels = Path(tmp) / "labels.json"
        nifti.write_bytes(mask_bytes)
        labels.write_bytes(labels_bytes)
        return load_mask_nifti(nifti, labels)
