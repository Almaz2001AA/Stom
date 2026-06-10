import stomserver
from stomserver.config import load_config


def test_defaults(monkeypatch):
    for var in ["STOM_DB_URL", "STOM_STORAGE_DIR", "STOM_REDIS_URL",
                "STOM_MODEL_DIR", "STOM_MAX_UPLOAD_BYTES", "STOM_JOB_TIMEOUT_SECONDS"]:
        monkeypatch.delenv(var, raising=False)
    cfg = load_config()
    assert cfg.db_url == "sqlite:///stom.db"
    assert cfg.storage_dir == "./storage"
    assert cfg.redis_url == "redis://localhost:6379/0"
    assert cfg.model_dir == "./models"
    assert cfg.max_upload_bytes == 500 * 1024 * 1024
    assert cfg.job_timeout_seconds == 60 * 60


def test_env_override(monkeypatch):
    monkeypatch.setenv("STOM_DB_URL", "sqlite:///other.db")
    monkeypatch.setenv("STOM_MAX_UPLOAD_BYTES", "1024")
    cfg = load_config()
    assert cfg.db_url == "sqlite:///other.db"
    assert cfg.max_upload_bytes == 1024
