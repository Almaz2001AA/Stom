"""Back-compat re-export: the runner now lives in :mod:`stomengine`.

Moved so the desktop client can run inference locally without importing the
cloud backend. Server code and existing imports keep working unchanged.
"""

from __future__ import annotations

from stomengine.runner import (  # noqa: F401
    MODEL_FG_MEAN,
    MODEL_FG_STD,
    DentalSegmentatorRunner,
    FakeRunner,
    SegmentationRunner,
    harmonize_to_model_domain,
    tta_enabled,
)

__all__ = [
    "MODEL_FG_MEAN",
    "MODEL_FG_STD",
    "DentalSegmentatorRunner",
    "FakeRunner",
    "SegmentationRunner",
    "harmonize_to_model_domain",
    "tta_enabled",
]
