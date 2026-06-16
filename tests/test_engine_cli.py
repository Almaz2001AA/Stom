import json
import sys
from collections import namedtuple

import numpy as np

from stomcore.geometry import Geometry
from stomcore.mask_io import load_mask_nifti
from stomcore.nifti_io import save_volume_nifti
from stomcore.volume import Volume
from stomengine import FakeRunner, InProcessEngine, SubprocessEngine
from stomengine.cli import resolve_model_dir, run_predict

_Proc = namedtuple("_Proc", "returncode stdout stderr")


def _volume(geo=None):
    geo = geo or Geometry.identity((0.3, 0.3, 0.3))
    return Volume(np.zeros((4, 5, 6), dtype=np.int16), geo)


# --- CLI core ---------------------------------------------------------------

def test_run_predict_writes_mask_and_labels(tmp_path):
    in_path = tmp_path / "in.nii.gz"
    save_volume_nifti(_volume(), in_path)
    out = tmp_path / "out"
    run_predict(in_path, out, InProcessEngine(FakeRunner()))
    assert (out / "mask.nii.gz").is_file()
    labels = json.loads((out / "mask_labels.json").read_text())
    assert "5" in labels  # DentalSegmentator has 5 structures


def test_resolve_model_dir_honors_env(monkeypatch):
    monkeypatch.setenv("STOM_MODEL_DIR", "/custom/model")
    assert resolve_model_dir() == "/custom/model"


# --- SubprocessEngine -------------------------------------------------------

def test_subprocess_engine_reads_back_mask(tmp_path):
    """Fake `run` segments with FakeRunner, writing into the out dir from argv."""

    def fake_run(cmd, env=None, timeout=None, on_progress=None, creationflags=0):
        # cmd == [*prefix, "predict", in_path, out_dir]
        in_path, out_dir = cmd[-2], cmd[-1]
        run_predict(in_path, out_dir, InProcessEngine(FakeRunner()))
        return _Proc(0, "", "")

    engine = SubprocessEngine("stom-engine", run=fake_run)
    mask = engine.segment(_volume())
    assert mask.is_compatible_with(_volume())


def test_subprocess_engine_forwards_progress():
    """A progress callback passed to segment() reaches the run seam."""
    seen = []

    def fake_run(cmd, env=None, timeout=None, on_progress=None, creationflags=0):
        # Simulate the engine reporting two tiles, as the streaming reader would.
        on_progress(1, 2)
        on_progress(2, 2)
        run_predict(cmd[-2], cmd[-1], InProcessEngine(FakeRunner()))
        return _Proc(0, "", "")

    engine = SubprocessEngine("stom-engine", run=fake_run)
    engine.segment(_volume(), progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 2), (2, 2)]


def test_subprocess_engine_raises_on_nonzero_exit():
    def fake_run(cmd, env=None, timeout=None, on_progress=None, creationflags=0):
        return _Proc(1, "", "boom: model not found")

    engine = SubprocessEngine("stom-engine", run=fake_run)
    try:
        engine.segment(_volume())
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "boom: model not found" in str(exc)


def test_subprocess_engine_decodes_windows_crash_code():
    # No stderr, just a Windows NTSTATUS exit code -> decode it into a hint
    # instead of surfacing an opaque number (0xC000013A = console-close kill).
    def fake_run(cmd, env=None, timeout=None, on_progress=None, creationflags=0):
        return _Proc(3221225786, "preprocessing\npredicting\n", "")

    engine = SubprocessEngine("stom-engine", run=fake_run)
    try:
        engine.segment(_volume())
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        msg = str(exc)
        assert "0xC000013A" in msg          # the decoded code
        assert "exit code 3221225786" in msg  # and the raw number
        assert "output tail" in msg          # stdout surfaced when stderr is empty


def test_subprocess_engine_raises_on_timeout():
    import subprocess

    def fake_run(cmd, env=None, timeout=None, on_progress=None, creationflags=0):
        raise subprocess.TimeoutExpired(cmd, timeout)

    engine = SubprocessEngine("stom-engine", timeout=5, run=fake_run)
    try:
        engine.segment(_volume())
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "timed out after 5s" in str(exc)


def test_subprocess_engine_passes_model_dir(tmp_path):
    seen = {}

    def fake_run(cmd, env=None, timeout=None, on_progress=None, creationflags=0):
        seen["model_dir"] = (env or {}).get("STOM_MODEL_DIR")
        run_predict(cmd[-2], cmd[-1], InProcessEngine(FakeRunner()))
        return _Proc(0, "", "")

    SubprocessEngine("stom-engine", model_dir="/weights/here", run=fake_run).segment(_volume())
    assert seen["model_dir"] == "/weights/here"


def test_subprocess_engine_passes_selected_model(tmp_path):
    seen = {}

    def fake_run(cmd, env=None, timeout=None, on_progress=None, creationflags=0):
        seen["model"] = (env or {}).get("STOM_MODEL")
        run_predict(cmd[-2], cmd[-1], InProcessEngine(FakeRunner()))
        return _Proc(0, "", "")

    engine = SubprocessEngine("stom-engine", run=fake_run)
    engine.segment(_volume())
    assert seen["model"] is None  # default: no STOM_MODEL -> DentalSegmentator

    engine.set_model("toothfairy2")
    engine.segment(_volume())
    assert seen["model"] == "toothfairy2"


# --- real subprocess round-trip (no model needed; FakeRunner via env) -------

def test_real_subprocess_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("STOM_ENGINE_FAKE", "1")
    engine = SubprocessEngine([sys.executable, "-m", "stomengine"])
    seen = []
    mask = engine.segment(_volume(), progress=lambda d, t: seen.append((d, t)))
    assert mask.is_compatible_with(_volume())
    # and the masked voxels survive the NIfTI round-trip through the subprocess
    assert int(mask.labels.max()) >= 1
    # the streaming reader parsed the engine's PROGRESS line off real stdout
    assert seen == [(1, 1)]


def test_stream_engine_parses_progress_and_captures_rest():
    """_stream_engine forwards PROGRESS lines and keeps other output for errors."""
    from stomengine.engine import _stream_engine

    script = (
        "import sys\n"
        "print('starting'); sys.stdout.flush()\n"
        "print('PROGRESS 3 10'); sys.stdout.flush()\n"
        "print('PROGRESS 10 10'); sys.stdout.flush()\n"
        "print('done')\n"
    )
    seen = []
    proc = _stream_engine(
        [sys.executable, "-c", script], env=None, timeout=30,
        on_progress=lambda d, t: seen.append((d, t)),
    )
    assert proc.returncode == 0
    assert seen == [(3, 10), (10, 10)]
    # PROGRESS lines are consumed; ordinary output is retained for diagnostics.
    assert "starting" in proc.stdout and "done" in proc.stdout
    assert "PROGRESS" not in proc.stdout
