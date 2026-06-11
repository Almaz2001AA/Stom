from stomclient.local_engine import build_local_engine
from stomengine.engine import InProcessEngine


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
