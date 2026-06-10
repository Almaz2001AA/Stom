"""Pure slice-rendering math: plane extraction, window/level, mask overlay."""

from __future__ import annotations

import numpy as np

from stomcore.mask import LabelInfo
from stomcore.volume import Volume

AXIAL = "axial"
CORONAL = "coronal"
SAGITTAL = "sagittal"
PLANES = (AXIAL, CORONAL, SAGITTAL)


def slice_count(volume: Volume, plane: str) -> int:
    z, y, x = volume.shape
    return {AXIAL: z, CORONAL: y, SAGITTAL: x}[plane]


def slice_array(array: np.ndarray, plane: str, index: int) -> np.ndarray:
    """Extract a 2D slice from a [z, y, x] array. Returns [row, col]."""
    if plane == AXIAL:
        return array[index, :, :]      # [y, x]
    if plane == CORONAL:
        return array[:, index, :]      # [z, x]
    if plane == SAGITTAL:
        return array[:, :, index]      # [z, y]
    raise ValueError(f"unknown plane: {plane}")


def apply_window_level(slice2d: np.ndarray, center: float, width: float) -> np.ndarray:
    """Map intensities to uint8 [0, 255] using window center/width."""
    width = max(float(width), 1.0)
    lo = center - width / 2.0
    clipped = np.clip(slice2d.astype(np.float64), lo, lo + width)
    scaled = (clipped - lo) / width * 255.0
    return scaled.astype(np.uint8)


def composite_overlay(
    gray_uint8: np.ndarray,
    mask_slice: np.ndarray,
    label_map: dict[int, LabelInfo],
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend visible mask labels over a grayscale slice. Returns [row, col, 3] uint8."""
    rgb = np.repeat(gray_uint8[:, :, None].astype(np.float64), 3, axis=2)
    for label_id, info in label_map.items():
        if not info.visible:
            continue
        sel = mask_slice == label_id
        if not sel.any():
            continue
        color = np.array(info.color, dtype=np.float64)
        rgb[sel] = (1.0 - alpha) * rgb[sel] + alpha * color
    return np.clip(rgb, 0, 255).astype(np.uint8)


def default_window_level(volume: Volume) -> tuple[float, float]:
    """Center/width spanning the volume's intensity range."""
    data = volume.voxels
    lo = float(data.min())
    hi = float(data.max())
    width = max(hi - lo, 1.0)
    return (lo + hi) / 2.0, width
