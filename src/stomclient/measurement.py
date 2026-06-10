"""Linear measurements in millimetres on a displayed 2D plane."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from stomcore.geometry import Geometry

from . import slice_renderer as sr


def plane_spacing(geometry: Geometry, plane: str) -> tuple[float, float]:
    """Return (col_spacing_mm, row_spacing_mm) for the displayed plane.

    Geometry.spacing is (x, y, z). Slice rows/cols per plane:
      axial    -> col=x, row=y
      coronal  -> col=x, row=z
      sagittal -> col=y, row=z
    """
    sx, sy, sz = geometry.spacing
    if plane == sr.AXIAL:
        return sx, sy
    if plane == sr.CORONAL:
        return sx, sz
    if plane == sr.SAGITTAL:
        return sy, sz
    raise ValueError(f"unknown plane: {plane}")


@dataclass(frozen=True)
class LinearMeasurement:
    p0: tuple[float, float]   # (col, row) in pixels
    p1: tuple[float, float]
    plane: str
    geometry: Geometry

    @property
    def length_mm(self) -> float:
        col_s, row_s = plane_spacing(self.geometry, self.plane)
        dc = (self.p1[0] - self.p0[0]) * col_s
        dr = (self.p1[1] - self.p0[1]) * row_s
        return math.hypot(dc, dr)


@dataclass
class MeasurementSet:
    items: list[LinearMeasurement] = field(default_factory=list)

    def add(self, m: LinearMeasurement) -> None:
        self.items.append(m)

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
