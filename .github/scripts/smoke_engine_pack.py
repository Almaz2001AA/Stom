"""Smoke test the published engine-pack on a clean Windows VM.

Reproduces the slim client's first-local-run path end-to-end against the REAL
release artifacts: fetch the manifest, download + SHA-256 verify + extract the
engine-pack, then run DentalSegmentator inference through SubprocessEngine on a
small synthetic volume. Exits non-zero on any failure so CI fails loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from stomclient.engine_pack import fetch_manifest, find_engine_exe, provision
from stomcore.geometry import Geometry
from stomcore.volume import Volume
from stomengine import SubprocessEngine


def main() -> int:
    root = Path("engine").resolve()

    print("1/4 fetching manifest from releases/latest ...")
    manifest = fetch_manifest()
    print(f"    version={manifest['version']} sha256={manifest['sha256'][:12]}...")

    print("2/4 downloading + verifying + extracting engine-pack (~524MB) ...")

    def _progress(done: int, total: int) -> None:
        if total:
            print(f"    {done * 100 // total:3d}% ({done >> 20}/{total >> 20} MB)", end="\r")

    exe = provision(manifest, root=root, progress=_progress)
    print(f"\n    extracted engine exe: {exe}")
    assert find_engine_exe(root) is not None, "engine exe not found after extract"

    print("3/4 building synthetic volume (32^3) ...")
    geo = Geometry.identity(spacing=(0.4, 0.4, 0.4))
    vol = Volume(np.zeros((32, 32, 32), dtype=np.int16), geo)

    print("4/4 running inference via SubprocessEngine (TTA off) ...")
    engine = SubprocessEngine(str(exe))
    mask = engine.segment(vol)

    assert mask.is_compatible_with(vol), "mask geometry incompatible with input"
    labels = np.asarray(mask.labels)
    print(f"    OK: mask shape={labels.shape} dtype={labels.dtype} "
          f"unique_labels={sorted(np.unique(labels).tolist())[:10]}")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
