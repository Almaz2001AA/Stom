"""Local segmentation engine: runner, labels, and the LocalEngine abstraction.

This package holds everything needed to run DentalSegmentator inference without
the cloud backend (no FastAPI/RQ/SQLAlchemy). It is importable by both the
server worker and the desktop client, so segmentation can run on the server or
locally on the user's machine.
"""

from .engine import InProcessEngine, LocalEngine, SubprocessEngine
from .labels import DENTALSEGMENTATOR_LABELS
from .runner import (
    DentalSegmentatorRunner,
    FakeRunner,
    SegmentationRunner,
    harmonize_to_model_domain,
    tta_enabled,
)

__all__ = [
    "DENTALSEGMENTATOR_LABELS",
    "DentalSegmentatorRunner",
    "FakeRunner",
    "InProcessEngine",
    "LocalEngine",
    "SegmentationRunner",
    "SubprocessEngine",
    "harmonize_to_model_domain",
    "tta_enabled",
]
