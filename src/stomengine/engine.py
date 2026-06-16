"""LocalEngine: turn a Volume into a SegmentationMask without the cloud.

A ``LocalEngine`` is the seam the desktop client uses for local segmentation.
``InProcessEngine`` runs a :class:`SegmentationRunner` in the current process
(used by the server worker, tests, and dev). A future ``SubprocessEngine`` will
shell out to the downloaded engine-pack so the slim frozen client need not bundle
torch/nnunet — both satisfy the same ``LocalEngine`` protocol.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Protocol

from stomcore.mask import SegmentationMask
from stomcore.mask_io import load_mask_nifti
from stomcore.nifti_io import save_volume_nifti
from stomcore.volume import Volume

from .labels import DENTALSEGMENTATOR_LABELS
from .runner import ProgressCb, SegmentationRunner

# Windows: run the console engine without a window so it is not tied to the GUI's
# transient console (whose teardown delivers a CTRL_CLOSE to any child process —
# the cause of exit 0xC000013A) and so no console flashes during inference.
_CREATE_NO_WINDOW = 0x08000000

# The engine-pack subprocess prints one of these per tile so the parent can show
# a live percentage (see stomengine.cli). Parsed out of stdout; other lines are
# kept for error reporting.
_PROGRESS_RE = re.compile(r"^PROGRESS\s+(\d+)\s+(\d+)\s*$")

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
    def segment(
        self, volume: Volume, *, progress: ProgressCb | None = None
    ) -> SegmentationMask:
        """Return a :class:`SegmentationMask` for ``volume``.

        Implementations must return a mask whose geometry matches ``volume``;
        the caller is expected to verify this and reject a mismatch. ``progress``,
        if given, is called with ``(steps_done, steps_total)`` as inference runs.
        """
        ...


class InProcessEngine:
    """Run a :class:`SegmentationRunner` in this process and build the mask."""

    def __init__(self, runner: SegmentationRunner, *, labels=None) -> None:
        self._runner = runner
        # Which label map to attach to the mask; defaults to DentalSegmentator's
        # 5 classes. ToothFairy2's 49-class map is passed in by the CLI.
        self._labels = DENTALSEGMENTATOR_LABELS if labels is None else labels

    def segment(
        self, volume: Volume, *, progress: ProgressCb | None = None
    ) -> SegmentationMask:
        # Forward progress only when asked, so simpler runners (test doubles)
        # that take just ``volume`` keep working.
        if progress is None:
            labels, geometry = self._runner.predict(volume)
        else:
            labels, geometry = self._runner.predict(volume, progress=progress)
        mask = SegmentationMask(labels, geometry, self._labels)
        if not mask.is_compatible_with(volume):
            raise ValueError(
                "predicted mask shape/geometry does not match input volume"
            )
        return mask


def _stream_engine(cmd, *, env, timeout, on_progress=None, creationflags=0):
    """Run the engine binary, streaming stdout so progress is live.

    ``subprocess.run(capture_output=True)`` blocks until the process exits, so
    the parent learns nothing until inference is done. We instead drain the
    child's output on a thread, forwarding ``PROGRESS <done> <total>`` lines to
    ``on_progress`` and keeping the rest for error reporting. stderr is merged
    into stdout so a single reader cannot deadlock on a full pipe. Returns a
    :class:`subprocess.CompletedProcess`; raises ``TimeoutExpired`` past
    ``timeout`` after killing the child.
    """
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, creationflags=creationflags,
    )
    captured: list[str] = []

    def _drain() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            match = _PROGRESS_RE.match(line.strip())
            if match and on_progress is not None:
                with contextlib.suppress(Exception):
                    on_progress(int(match.group(1)), int(match.group(2)))
            else:
                captured.append(line)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    reader.join(timeout=5)
    return subprocess.CompletedProcess(
        cmd, proc.returncode, stdout="".join(captured), stderr=""
    )


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
        run=_stream_engine,
    ) -> None:
        # Accept a bare path or a full argv prefix (e.g. ["python", "-m", ...]).
        self._cmd = [str(exe)] if isinstance(exe, (str, os.PathLike)) else list(exe)
        self._model_dir = model_dir
        # Cap inference wall-time so a wedged engine surfaces as an error instead
        # of hanging the caller forever (None = wait indefinitely).
        self._timeout = timeout
        self._run = run

    def segment(
        self, volume: Volume, *, progress: ProgressCb | None = None
    ) -> SegmentationMask:
        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "input.nii.gz"
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()
            save_volume_nifti(volume, in_path)

            env = dict(os.environ)
            if self._model_dir:
                env["STOM_MODEL_DIR"] = self._model_dir

            creationflags = _CREATE_NO_WINDOW if os.name == "nt" else 0

            try:
                proc = self._run(
                    [*self._cmd, "predict", str(in_path), str(out_dir)],
                    env=env,
                    timeout=self._timeout,
                    on_progress=progress,
                    creationflags=creationflags,
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
