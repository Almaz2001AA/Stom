"""LocalEngine: turn a Volume into a SegmentationMask without the cloud.

A ``LocalEngine`` is the seam the desktop client uses for local segmentation.
``InProcessEngine`` runs a :class:`SegmentationRunner` in the current process
(used by the server worker, tests, and dev). A future ``SubprocessEngine`` will
shell out to the downloaded engine-pack so the slim frozen client need not bundle
torch/nnunet — both satisfy the same ``LocalEngine`` protocol.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from stomcore.mask import SegmentationMask
from stomcore.mask_io import load_mask_nifti
from stomcore.nifti_io import save_volume_nifti
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


class SubprocessEngine:
    """Run inference by shelling out to the engine-pack ``stom-engine`` binary.

    The slim frozen client does not bundle torch/nnU-Net; it downloads the
    engine-pack on first use and points this engine at its executable. We pass
    the volume as a temp NIfTI and read the mask + labels back from the
    subprocess's output directory.
    """

    def __init__(
        self,
        exe: str | os.PathLike | list[str],
        *,
        model_dir: str | None = None,
        run=subprocess.run,
    ) -> None:
        # Accept a bare path or a full argv prefix (e.g. ["python", "-m", ...]).
        self._cmd = [str(exe)] if isinstance(exe, (str, os.PathLike)) else list(exe)
        self._model_dir = model_dir
        self._run = run

    def segment(self, volume: Volume) -> SegmentationMask:
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "input.nii.gz"
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()
            save_volume_nifti(volume, in_path)

            env = dict(os.environ)
            if self._model_dir:
                env["STOM_MODEL_DIR"] = self._model_dir

            proc = self._run(
                [*self._cmd, "predict", str(in_path), str(out_dir)],
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or "").strip() or f"exit code {proc.returncode}"
                raise RuntimeError(f"local engine failed: {detail}")

            mask = load_mask_nifti(out_dir / "mask.nii.gz", out_dir / "mask_labels.json")

        if not mask.is_compatible_with(volume):
            raise ValueError("engine mask shape/geometry does not match input volume")
        return mask
