"""3D voxel volume bound to a spatial Geometry."""

from __future__ import annotations

import numpy as np

from .geometry import Geometry


class Volume:
    """Immutable-by-convention 3D volume.

    voxels: NumPy array indexed [z, y, x] (SimpleITK array order).
    geometry: spacing/origin/direction in (x, y, z) order.
    """

    def __init__(self, voxels: np.ndarray, geometry: Geometry) -> None:
        if voxels.ndim != 3:
            raise ValueError(f"voxels must be 3D [z, y, x], got ndim={voxels.ndim}")
        self._voxels = voxels
        self._geometry = geometry

    @property
    def voxels(self) -> np.ndarray:
        return self._voxels

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self._voxels.shape)  # type: ignore[return-value]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Volume):
            return NotImplemented
        return self._geometry == other._geometry and np.array_equal(
            self._voxels, other._voxels
        )
