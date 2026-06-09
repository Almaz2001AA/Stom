"""Spatial geometry shared by volumes and masks."""

from __future__ import annotations

import math
from dataclasses import dataclass

_IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Geometry:
    """Voxel-to-world mapping: spacing (mm), origin, direction cosines.

    spacing/origin are (x, y, z); direction is a row-major flat 3x3 matrix.
    """

    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.spacing) != 3:
            raise ValueError("spacing must have 3 components")
        if len(self.origin) != 3:
            raise ValueError("origin must have 3 components")
        if len(self.direction) != 9:
            raise ValueError("direction must have 9 components (flat 3x3)")

    @classmethod
    def identity(
        cls,
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "Geometry":
        return cls(spacing=spacing, origin=origin, direction=_IDENTITY_DIRECTION)

    def is_compatible(self, other: "Geometry", tol: float = 1e-4) -> bool:
        """True if spacing, origin and direction all match within tol."""
        return (
            _all_close(self.spacing, other.spacing, tol)
            and _all_close(self.origin, other.origin, tol)
            and _all_close(self.direction, other.direction, tol)
        )


def _all_close(a: tuple[float, ...], b: tuple[float, ...], tol: float) -> bool:
    return len(a) == len(b) and all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))
