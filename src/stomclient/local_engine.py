"""Build a LocalEngine for on-device segmentation, or None if unavailable.

When run from source or alongside the server, the weights and torch/nnunet are
present, so we run :class:`InProcessEngine` directly. The slim frozen client
ships without those; Phase B will instead return a subprocess-backed engine
pointing at the downloaded engine-pack. Either way the caller gets a
``LocalEngine`` (or ``None``), and importing this module never pulls in torch —
the heavy imports happen only when inference actually runs.
"""

from __future__ import annotations

import os
from pathlib import Path


def build_local_engine(model_dir: str | None = None):
    """Return a LocalEngine for on-device segmentation, or None if unavailable.

    Preference order:
    1. A downloaded engine-pack (slim production client) -> ``SubprocessEngine``.
    2. A from-source / server-side environment with weights + torch present ->
       ``InProcessEngine``.
    Otherwise ``None`` (local mode stays disabled until the pack is installed).
    """
    try:
        from .engine_pack import find_engine_exe

        exe = find_engine_exe()
        if exe is not None:
            from stomengine import SubprocessEngine

            return SubprocessEngine(str(exe))
    except Exception:  # noqa: BLE001 - fall through to in-process / disabled
        pass

    model_dir = model_dir or os.environ.get("STOM_MODEL_DIR")
    if not model_dir or not Path(model_dir).is_dir():
        return None
    try:
        from stomengine import DentalSegmentatorRunner, InProcessEngine
    except Exception:  # noqa: BLE001 - engine deps absent -> no local mode
        return None
    return InProcessEngine(DentalSegmentatorRunner(model_dir))
