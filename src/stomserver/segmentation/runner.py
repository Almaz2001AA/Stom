"""Segmentation runner: interface, deterministic fake, and real nnU-Net runner."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from stomcore.volume import Volume


class SegmentationRunner(Protocol):
    def predict(self, volume: Volume) -> np.ndarray:
        """Return a label volume [z, y, x] matching the input volume shape."""
        ...


class FakeRunner:
    """Deterministic stand-in: labels a few fixed voxels. No model needed."""

    def predict(self, volume: Volume) -> np.ndarray:
        labels = np.zeros(volume.shape, dtype=np.uint16)
        flat = labels.reshape(-1)
        for i in range(min(5, flat.size)):
            flat[i] = i + 1
        return labels


class DentalSegmentatorRunner:
    """Real nnU-Net v2 runner. Body implemented in Task 14."""

    def __init__(self, model_dir: str) -> None:
        self._model_dir = model_dir

    def predict(self, volume: Volume) -> np.ndarray:
        raise NotImplementedError("DentalSegmentatorRunner.predict added in Task 14")
