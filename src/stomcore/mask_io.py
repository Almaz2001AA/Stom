"""Serialize SegmentationMask as a .nii.gz label volume + a JSON label sidecar."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .mask import LabelInfo, SegmentationMask
from .nifti_io import load_volume_nifti, save_volume_nifti
from .volume import Volume


def save_mask_nifti(
    mask: SegmentationMask,
    nifti_path: str | os.PathLike,
    labels_path: str | os.PathLike,
) -> None:
    save_volume_nifti(Volume(mask.labels, mask.geometry), nifti_path)
    payload = {
        str(info.label_id): {
            "name": info.name,
            "color": list(info.color),
            "visible": info.visible,
        }
        for info in mask.label_map.values()
    }
    Path(labels_path).write_text(json.dumps(payload, indent=2))


def load_mask_nifti(
    nifti_path: str | os.PathLike,
    labels_path: str | os.PathLike,
) -> SegmentationMask:
    volume = load_volume_nifti(nifti_path)
    raw = json.loads(Path(labels_path).read_text())
    label_map = {
        int(k): LabelInfo(
            label_id=int(k),
            name=v["name"],
            color=tuple(v["color"]),
            visible=v["visible"],
        )
        for k, v in raw.items()
    }
    return SegmentationMask(volume.voxels, volume.geometry, label_map)
