"""Multi-label segmentation mask aligned to a Volume's geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Geometry
from .volume import Volume

BACKGROUND_LABEL = 0


@dataclass(frozen=True)
class LabelInfo:
    """Describes one label: its id, human name, RGB color and visibility."""

    label_id: int
    name: str
    color: tuple[int, int, int]
    visible: bool = True


class SegmentationMask:
    """Integer label volume [z, y, x] plus a label_id -> LabelInfo map.

    Label 0 is background and is never listed in label_map.
    """

    def __init__(
        self,
        labels: np.ndarray,
        geometry: Geometry,
        label_map: dict[int, LabelInfo],
    ) -> None:
        if labels.ndim != 3:
            raise ValueError(f"labels must be 3D [z, y, x], got ndim={labels.ndim}")
        self._labels = labels
        self._geometry = geometry
        self._label_map = dict(label_map)

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def label_map(self) -> dict[int, LabelInfo]:
        return self._label_map

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self._labels.shape)  # type: ignore[return-value]

    def present_labels(self) -> set[int]:
        """Set of label ids actually present in the volume, excluding background."""
        present = set(int(v) for v in np.unique(self._labels))
        present.discard(BACKGROUND_LABEL)
        return present

    def is_compatible_with(self, volume: Volume, tol: float = 1e-4) -> bool:
        """True if this mask matches the volume in shape and geometry."""
        return self.shape == volume.shape and self._geometry.is_compatible(
            volume.geometry, tol=tol
        )
