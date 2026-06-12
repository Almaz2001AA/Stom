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

# Windows: run the console engine without a window so it is not tied to the GUI's
# transient console (whose teardown delivers a CTRL_CLOSE to any child process —
# the cause of exit 0xC000013A) and so no console flashes during inference.
_CREATE_NO_WINDOW = 0x08000000

# Friendly hints for Windows NTSTATUS exit codes the engine can surface with no
# stderr of its own, so an opaque number becomes an actionable message.
_WIN_EXIT_HINTS = {
    3221225786: "the engine was terminated by a console-close/Ctrl+C event "
                "(0xC000013A) — typically a multiprocessing/console issue",
    3221225477: "memory access violation (0xC0000005) — possible out-of-memory "
                "on a large volume",
    3221225781: "a required DLL was not found (0xC0000135) — the engine-pack may "
                "be incomplete; reinstall/update the engine",
}


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


def _failure_detail(proc) -> str:
    """Build the most informative message available from a failed engine run.

    Prefers the engine's own stderr; otherwise falls back to stdout, then to a
    decoded exit code with a hint for known Windows crash codes.
    """
    stderr = (proc.stderr or "").strip()
    if stderr:
        return stderr
    code = proc.returncode
    parts = [f"exit code {code}"]
    if code in _WIN_EXIT_HINTS:
        parts.append(_WIN_EXIT_HINTS[code])
    stdout = (proc.stdout or "").strip()
    if stdout:
        parts.append(f"output tail: {stdout[-500:]}")
    return " — ".join(parts)


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
        timeout: float | None = None,
        run=subprocess.run,
    ) -> None:
        # Accept a bare path or a full argv prefix (e.g. ["python", "-m", ...]).
        self._cmd = [str(exe)] if isinstance(exe, (str, os.PathLike)) else list(exe)
        self._model_dir = model_dir
        # Cap inference wall-time so a wedged engine surfaces as an error instead
        # of hanging the caller forever (None = wait indefinitely).
        self._timeout = timeout
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

            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = _CREATE_NO_WINDOW

            try:
                proc = self._run(
                    [*self._cmd, "predict", str(in_path), str(out_dir)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    **kwargs,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"local engine timed out after {self._timeout}s"
                ) from exc
            if proc.returncode != 0:
                raise RuntimeError(f"local engine failed: {_failure_detail(proc)}")

            mask = load_mask_nifti(out_dir / "mask.nii.gz", out_dir / "mask_labels.json")

        if not mask.is_compatible_with(volume):
            raise ValueError("engine mask shape/geometry does not match input volume")
        return mask
