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
    ToothFairy2Runner,
    clamp_air_padding,
    harmonize_to_model_domain,
    tta_enabled,
)
from .tf2_labels import TOOTHFAIRY2_LABELS

__all__ = [
    "DENTALSEGMENTATOR_LABELS",
    "TOOTHFAIRY2_LABELS",
    "DentalSegmentatorRunner",
    "FakeRunner",
    "InProcessEngine",
    "LocalEngine",
    "SegmentationRunner",
    "SubprocessEngine",
    "ToothFairy2Runner",
    "clamp_air_padding",
    "harmonize_to_model_domain",
    "tta_enabled",
]
