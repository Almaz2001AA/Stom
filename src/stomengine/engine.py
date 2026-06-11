"""LocalEngine: turn a Volume into a SegmentationMask without the cloud.

A ``LocalEngine`` is the seam the desktop client uses for local segmentation.
``InProcessEngine`` runs a :class:`SegmentationRunner` in the current process
(used by the server worker, tests, and dev). A future ``SubprocessEngine`` will
shell out to the downloaded engine-pack so the slim frozen client need not bundle
torch/nnunet — both satisfy the same ``LocalEngine`` protocol.
"""

from __future__ import annotations

from typing import Protocol

from stomcore.mask import SegmentationMask
from stomcore.volume import Volume

from .labels import DENTALSEGMENTATOR_LABELS
from .runner import SegmentationRunner


class LocalEngine(Protocol):
    def segment(self, volume: Volume) -> SegmentationMask:
        """Return a :class:`SegmentationMask` for ``volume``.

        Implementations must return a mask whose geometry matches ``volume``;
        the caller is expected to verify this and reject a mismatch.
        """
        ...


class InProcessEngine:
    """Run a :class:`SegmentationRunner` in this process and build the mask."""

    def __init__(self, runner: SegmentationRunner) -> None:
        self._runner = runner

    def segment(self, volume: Volume) -> SegmentationMask:
        labels, geometry = self._runner.predict(volume)
        mask = SegmentationMask(labels, geometry, DENTALSEGMENTATOR_LABELS)
        if not mask.is_compatible_with(volume):
            raise ValueError(
                "predicted mask shape/geometry does not match input volume"
            )
        return mask
