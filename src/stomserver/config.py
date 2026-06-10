"""Backend configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_MAX_UPLOAD = 500 * 1024 * 1024  # 500 MB
_DEFAULT_JOB_TIMEOUT = 60 * 60  # 1 hour; jobs running longer are reaped as stale


@dataclass(frozen=True)
class Config:
    db_url: str
    storage_dir: str
    redis_url: str
    model_dir: str
    max_upload_bytes: int
    job_timeout_seconds: int


def load_config() -> Config:
    return Config(
        db_url=os.environ.get("STOM_DB_URL", "sqlite:///stom.db"),
        storage_dir=os.environ.get("STOM_STORAGE_DIR", "./storage"),
        redis_url=os.environ.get("STOM_REDIS_URL", "redis://localhost:6379/0"),
        model_dir=os.environ.get("STOM_MODEL_DIR", "./models"),
        max_upload_bytes=int(
            os.environ.get("STOM_MAX_UPLOAD_BYTES", str(_DEFAULT_MAX_UPLOAD))
        ),
        job_timeout_seconds=int(
            os.environ.get("STOM_JOB_TIMEOUT_SECONDS", str(_DEFAULT_JOB_TIMEOUT))
        ),
    )
