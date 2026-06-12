from stomclient.local_engine import (
    build_local_engine,
    engine_update_available,
    provision_local_engine,
)
from stomengine.engine import InProcessEngine, SubprocessEngine


def test_returns_none_without_model_dir(monkeypatch):
    monkeypatch.delenv("STOM_MODEL_DIR", raising=False)
    assert build_local_engine() is None


def test_returns_none_for_missing_dir():
    assert build_local_engine("/no/such/model/dir") is None


def test_builds_inprocess_engine_when_dir_exists(tmp_path):
    engine = build_local_engine(str(tmp_path))
    assert isinstance(engine, InProcessEngine)


def test_env_var_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("STOM_MODEL_DIR", str(tmp_path))
    assert isinstance(build_local_engine(), InProcessEngine)


def test_provision_local_engine_fetches_then_builds_subprocess(tmp_path, monkeypatch):
    from stomclient import engine_pack

    exe = tmp_path / "stom-engine"
    exe.write_text("")
    seen = {}

    monkeypatch.setattr(
        engine_pack, "fetch_manifest",
        lambda: {"url": "http://x/p.zip", "sha256": "abc", "version": "v9"},
    )

    def fake_provision(manifest, *, progress=None, clean=False):
        seen["manifest"] = manifest
        seen["progress"] = progress
        seen["clean"] = clean
        return exe

    monkeypatch.setattr(engine_pack, "provision", fake_provision)

    cb = lambda done, total: None  # noqa: E731
    engine = provision_local_engine(progress=cb)

    assert isinstance(engine, SubprocessEngine)
    assert str(exe) in engine._cmd          # subprocess points at the unpacked binary
    assert seen["manifest"]["version"] == "v9"
    assert seen["progress"] is cb           # progress wired through to the download
    assert seen["clean"] is False           # first-run install does not wipe


def test_provision_local_engine_clean_passes_through(tmp_path, monkeypatch):
    from stomclient import engine_pack

    exe = tmp_path / "stom-engine"
    exe.write_text("")
    seen = {}
    monkeypatch.setattr(engine_pack, "fetch_manifest", lambda: {"version": "v9"})

    def fake_provision(manifest, *, progress=None, clean=False):
        seen["clean"] = clean
        return exe

    monkeypatch.setattr(engine_pack, "provision", fake_provision)
    provision_local_engine(clean=True)
    assert seen["clean"] is True            # update wipes the old pack


def test_engine_update_available_is_network_tolerant(monkeypatch):
    from stomclient import engine_pack

    def _boom():
        raise OSError("offline")

    monkeypatch.setattr(engine_pack, "fetch_manifest", _boom)
    assert engine_update_available() is False   # never raises on a failed check
