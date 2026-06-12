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
from collections.abc import Callable
from pathlib import Path

ProgressCb = Callable[[int, int], None]  # (bytes_done, bytes_total)


def provision_local_engine(progress: ProgressCb | None = None, *, clean: bool = False):
    """Download + install the engine-pack and return a ready ``SubprocessEngine``.

    Used by the slim client's "Install local engine" (first run) and "Update
    engine" actions: fetch the release manifest, download + verify + extract the
    engine-pack, and point a subprocess engine at the unpacked binary. Pass
    ``clean=True`` for an update so the old pack's files are removed first. Raises
    on any failure (network, checksum, missing binary) for the caller to surface.
    """
    from stomengine import SubprocessEngine

    from . import engine_pack

    manifest = engine_pack.fetch_manifest()
    exe = engine_pack.provision(manifest, progress=progress, clean=clean)
    return SubprocessEngine(str(exe))


def engine_update_available() -> bool:
    """Whether a newer engine-pack than the installed one is published.

    Network-tolerant: any failure (offline, GitHub down) returns False so startup
    never blocks or errors on the update check.
    """
    try:
        from . import engine_pack

        return engine_pack.engine_update_available(engine_pack.fetch_manifest())
    except Exception:  # noqa: BLE001 - update check must never break startup
        return False


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
