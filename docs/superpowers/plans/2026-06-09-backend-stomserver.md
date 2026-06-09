# Backend (`stomserver`) Implementation Plan — Plan 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the cloud backend `stomserver` — a FastAPI API + RQ worker that ingests a CBCT NIfTI volume, runs real DentalSegmentator (nnU-Net v2) segmentation, and serves the resulting mask — with bearer-token auth and per-account isolation, plus a `stomcore.mask_io` addition for mask serialization.

**Architecture:** New package `stomserver` depending on the existing `stomcore`. The API server only stores data and enqueues jobs; a separate RQ worker performs segmentation and writes results. External services sit behind interfaces (`Storage`, a job queue, SQLAlchemy ORM) so dev runs on local disk + SQLite + Redis (with `fakeredis` in tests) and prod swaps via config. The heavy nnU-Net call is hidden behind a `SegmentationRunner` interface with a `FakeRunner` so the whole pipeline is testable without weights or GPU.

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy 2.0, RQ + Redis (`fakeredis` in tests), httpx/TestClient, pytest, SimpleITK (via `stomcore`), nnU-Net v2 (real runner only).

---

## Environment notes (already set up — do not redo)

- venv at `/opt/almaz/test/Stom/.venv`; run tests as `.venv/bin/python -m pytest`. Do NOT create venvs.
- These are already installed in the venv: `fastapi`, `uvicorn[standard]`, `sqlalchemy>=2.0`, `rq`, `redis`, `fakeredis`, `httpx`, `python-multipart`, plus `numpy`, `SimpleITK`, `pytest` (from Plan 1). `nnunetv2` is NOT installed (Task 14 handles it; everything else uses `FakeRunner`).
- `stomcore` is installed editable and exposes `Volume`, `Geometry`, `SegmentationMask`, `LabelInfo`, `load_volume_nifti`, `save_volume_nifti`, `DicomLoader`, `DicomError`.
- git repo on branch `master`, remote `origin` configured. Commit after each task; do not push unless asked.
- `tests/conftest.py` already exists with a `_write_dicom_series` helper and DICOM fixtures from Plan 1 — append new fixtures, do not break existing ones.

## File Structure

```
src/stomcore/
  mask_io.py                 # NEW: save/load SegmentationMask as .nii.gz + label JSON sidecar
src/stomserver/
  __init__.py
  config.py                  # Config dataclass + load_config() from env
  db/
    __init__.py
    models.py                # Base, Account, ApiToken, Study, Job
    session.py               # make_engine / make_session_factory / create_all
  storage/
    __init__.py
    base.py                  # Storage ABC + StorageKeyError
    local.py                 # LocalFileStorage
  queue/
    __init__.py
    base.py                  # JobQueue protocol
    rq_queue.py              # RqJobQueue
  auth.py                    # hash_token() + get_current_account dependency
  segmentation/
    __init__.py
    labels.py                # DENTALSEGMENTATOR_LABELS
    runner.py                # SegmentationRunner protocol, FakeRunner, DentalSegmentatorRunner
    worker.py                # run_segmentation (RQ entry) + _run_segmentation (testable core)
  api/
    __init__.py
    schemas.py               # Pydantic response models
    deps.py                  # get_db / get_storage / get_queue (read app.state)
    errors.py                # exception handlers -> {detail, code}
    routes_studies.py
    routes_jobs.py
    app.py                   # create_app() factory
scripts/
  create_account.py          # admin CLI: create Account + issue token
  download_weights.py        # download DentalSegmentator weights from Zenodo
NOTICE                       # attribution (CC-BY) for DentalSegmentator + nnU-Net
tests/
  conftest.py                # (append) app/client/db/storage fixtures
  test_mask_io.py
  test_storage_local.py
  test_db_models.py
  test_auth.py
  test_labels.py
  test_api_health.py
  test_api_studies.py
  test_api_jobs.py
  test_worker.py
  test_api_masks.py
  test_integration_e2e.py
  test_create_account_script.py
  test_runner_real.py        # @pytest.mark.slow
```

---

## Task 1: `stomcore.mask_io` — mask serialization (closes Plan 1 follow-up)

Add mask `.nii.gz` + label-map JSON I/O to the shared core. Reuses `nifti_io` (DRY). The round-trip test uses a **non-identity direction** matrix (closes the test gap flagged in the Plan 1 review).

**Files:**
- Create: `src/stomcore/mask_io.py`
- Test: `tests/test_mask_io.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mask_io.py`:

```python
import numpy as np

from stomcore.geometry import Geometry
from stomcore.mask import LabelInfo, SegmentationMask
from stomcore.mask_io import load_mask_nifti, save_mask_nifti


def _mask():
    geo = Geometry(spacing=(0.3, 0.4, 0.5), origin=(1.0, 2.0, 3.0),
                   direction=(0, -1, 0, 1, 0, 0, 0, 0, 1))  # non-identity
    labels = np.zeros((5, 4, 3), dtype=np.uint16)
    labels[0, 0, 0] = 2
    labels[1, 1, 1] = 5
    label_map = {
        2: LabelInfo(2, "mandible", (200, 170, 130)),
        5: LabelInfo(5, "mandibular-canal", (220, 80, 80), visible=False),
    }
    return SegmentationMask(labels, geo, label_map)


def test_save_then_load_round_trips_mask(tmp_path):
    mask = _mask()
    nifti = tmp_path / "mask.nii.gz"
    labels = tmp_path / "mask_labels.json"
    save_mask_nifti(mask, nifti, labels)
    assert nifti.exists() and labels.exists()

    restored = load_mask_nifti(nifti, labels)
    assert restored.shape == mask.shape
    np.testing.assert_array_equal(restored.labels, mask.labels)
    assert restored.geometry.is_compatible(mask.geometry)
    assert restored.present_labels() == {2, 5}
    assert restored.label_map[5].name == "mandibular-canal"
    assert restored.label_map[5].color == (220, 80, 80)
    assert restored.label_map[5].visible is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mask_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stomcore.mask_io'`

- [ ] **Step 3: Implement**

Create `src/stomcore/mask_io.py`:

```python
"""Serialize SegmentationMask as a .nii.gz label volume + a JSON label sidecar."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .mask import LabelInfo, SegmentationMask
from .nifti_io import load_volume_nifti, save_volume_nifti
from .volume import Volume


def save_mask_nifti(
    mask: SegmentationMask,
    nifti_path: str | os.PathLike,
    labels_path: str | os.PathLike,
) -> None:
    save_volume_nifti(Volume(mask.labels, mask.geometry), nifti_path)
    payload = {
        str(info.label_id): {
            "name": info.name,
            "color": list(info.color),
            "visible": info.visible,
        }
        for info in mask.label_map.values()
    }
    Path(labels_path).write_text(json.dumps(payload, indent=2))


def load_mask_nifti(
    nifti_path: str | os.PathLike,
    labels_path: str | os.PathLike,
) -> SegmentationMask:
    volume = load_volume_nifti(nifti_path)
    raw = json.loads(Path(labels_path).read_text())
    label_map = {
        int(k): LabelInfo(
            label_id=int(k),
            name=v["name"],
            color=tuple(v["color"]),
            visible=v["visible"],
        )
        for k, v in raw.items()
    }
    return SegmentationMask(volume.voxels, volume.geometry, label_map)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_mask_io.py -v`
Expected: `1 passed`

- [ ] **Step 5: Re-export from stomcore and commit**

Edit `src/stomcore/__init__.py` — add to the imports and `__all__`:

```python
from .mask_io import load_mask_nifti, save_mask_nifti
```

Add `"load_mask_nifti"` and `"save_mask_nifti"` to the `__all__` list.

Run: `.venv/bin/python -m pytest -q` (expect all green, 34 passed).

```bash
git add src/stomcore/mask_io.py src/stomcore/__init__.py tests/test_mask_io.py
git commit -m "feat: add stomcore.mask_io for mask + label serialization

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `stomserver` scaffolding + config

Create the package, register it in `pyproject.toml`, add the `slow` pytest marker, and add `config.py`.

**Files:**
- Modify: `pyproject.toml`
- Create: `src/stomserver/__init__.py`, `src/stomserver/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Update `pyproject.toml`**

In `pyproject.toml`, add a `server` optional-dependencies group and register the `slow` marker. Add these entries (merge into existing sections; do not remove the existing `dev` extra or `stom-dicom2nifti` script):

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
server = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "sqlalchemy>=2.0",
    "rq>=1.16",
    "redis>=5.0",
    "fakeredis>=2.21",
    "httpx>=0.27",
    "python-multipart>=0.0.9",
]
nnunet = ["nnunetv2>=2.5"]
```

Under `[tool.pytest.ini_options]` add:

```toml
markers = ["slow: tests that need the real model/weights (skipped by default unless run explicitly)"]
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_config.py`:

```python
import stomserver
from stomserver.config import load_config


def test_defaults(monkeypatch):
    for var in ["STOM_DB_URL", "STOM_STORAGE_DIR", "STOM_REDIS_URL",
                "STOM_MODEL_DIR", "STOM_MAX_UPLOAD_BYTES"]:
        monkeypatch.delenv(var, raising=False)
    cfg = load_config()
    assert cfg.db_url == "sqlite:///stom.db"
    assert cfg.storage_dir == "./storage"
    assert cfg.redis_url == "redis://localhost:6379/0"
    assert cfg.model_dir == "./models"
    assert cfg.max_upload_bytes == 500 * 1024 * 1024


def test_env_override(monkeypatch):
    monkeypatch.setenv("STOM_DB_URL", "sqlite:///other.db")
    monkeypatch.setenv("STOM_MAX_UPLOAD_BYTES", "1024")
    cfg = load_config()
    assert cfg.db_url == "sqlite:///other.db"
    assert cfg.max_upload_bytes == 1024
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stomserver'`

- [ ] **Step 4: Implement**

Create `src/stomserver/__init__.py`:

```python
"""Cloud backend for CBCT segmentation."""

__version__ = "0.1.0"
```

Create `src/stomserver/config.py`:

```python
"""Backend configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_MAX_UPLOAD = 500 * 1024 * 1024  # 500 MB


@dataclass(frozen=True)
class Config:
    db_url: str
    storage_dir: str
    redis_url: str
    model_dir: str
    max_upload_bytes: int


def load_config() -> Config:
    return Config(
        db_url=os.environ.get("STOM_DB_URL", "sqlite:///stom.db"),
        storage_dir=os.environ.get("STOM_STORAGE_DIR", "./storage"),
        redis_url=os.environ.get("STOM_REDIS_URL", "redis://localhost:6379/0"),
        model_dir=os.environ.get("STOM_MODEL_DIR", "./models"),
        max_upload_bytes=int(
            os.environ.get("STOM_MAX_UPLOAD_BYTES", str(_DEFAULT_MAX_UPLOAD))
        ),
    )
```

- [ ] **Step 5: Install editable, run, commit**

Run:
```bash
cd /opt/almaz/test/Stom
.venv/bin/python -m pip install -e ".[dev,server]"
.venv/bin/python -m pytest tests/test_config.py -v
```
Expected: `2 passed`.

```bash
git add pyproject.toml src/stomserver/__init__.py src/stomserver/config.py tests/test_config.py
git commit -m "chore: scaffold stomserver package with config

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Storage layer

`Storage` interface + `LocalFileStorage` with path-traversal protection.

**Files:**
- Create: `src/stomserver/storage/__init__.py`, `src/stomserver/storage/base.py`, `src/stomserver/storage/local.py`
- Test: `tests/test_storage_local.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_storage_local.py`:

```python
import pytest

from stomserver.storage.base import StorageKeyError
from stomserver.storage.local import LocalFileStorage


def test_put_get_roundtrip(tmp_path):
    s = LocalFileStorage(str(tmp_path))
    s.put("acct/1/volume.nii.gz", b"hello")
    assert s.get("acct/1/volume.nii.gz") == b"hello"


def test_exists(tmp_path):
    s = LocalFileStorage(str(tmp_path))
    assert s.exists("missing") is False
    s.put("k", b"x")
    assert s.exists("k") is True


def test_get_missing_raises(tmp_path):
    s = LocalFileStorage(str(tmp_path))
    with pytest.raises(StorageKeyError):
        s.get("nope")


def test_delete(tmp_path):
    s = LocalFileStorage(str(tmp_path))
    s.put("k", b"x")
    s.delete("k")
    assert s.exists("k") is False
    s.delete("k")  # idempotent, no error


def test_rejects_path_traversal(tmp_path):
    s = LocalFileStorage(str(tmp_path))
    with pytest.raises(ValueError):
        s.put("../escape", b"x")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_storage_local.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stomserver.storage'`

- [ ] **Step 3: Implement**

Create `src/stomserver/storage/__init__.py` (empty):

```python
```

Create `src/stomserver/storage/base.py`:

```python
"""Storage interface for binary objects (volumes, masks)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class StorageKeyError(KeyError):
    """Raised when a requested storage key does not exist."""


class Storage(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...
```

Create `src/stomserver/storage/local.py`:

```python
"""Filesystem-backed Storage for local development."""

from __future__ import annotations

from pathlib import Path

from .base import Storage, StorageKeyError


class LocalFileStorage(Storage):
    def __init__(self, root: str) -> None:
        self._root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if path != self._root and self._root not in path.parents:
            raise ValueError(f"invalid storage key (path traversal): {key}")
        return path

    def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise StorageKeyError(key)
        return path.read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.is_file():
            path.unlink()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_storage_local.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomserver/storage/ tests/test_storage_local.py
git commit -m "feat: add Storage interface and LocalFileStorage

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Database models + session

SQLAlchemy 2.0 models (`Account`, `ApiToken`, `Study`, `Job`) and session helpers.

**Files:**
- Create: `src/stomserver/db/__init__.py`, `src/stomserver/db/models.py`, `src/stomserver/db/session.py`
- Test: `tests/test_db_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_db_models.py`:

```python
from stomserver.db.models import Account, ApiToken, Job, Study
from stomserver.db.session import create_all, make_engine, make_session_factory


def _session():
    engine = make_engine("sqlite://")  # in-memory
    create_all(engine)
    return make_session_factory(engine)()


def test_create_and_query_account_token():
    db = _session()
    acct = Account(name="Clinic A")
    db.add(acct)
    db.flush()
    db.add(ApiToken(token_hash="abc", account_id=acct.id))
    db.commit()

    found = db.query(ApiToken).filter_by(token_hash="abc").one()
    assert found.account_id == acct.id


def test_study_and_job_defaults():
    db = _session()
    acct = Account(name="A")
    db.add(acct)
    db.flush()
    study = Study(account_id=acct.id, original_filename="s.nii.gz",
                  storage_key="A/studies/1/volume.nii.gz",
                  shape="[8, 16, 16]", spacing="[0.3, 0.3, 0.3]")
    db.add(study)
    db.flush()
    job = Job(study_id=study.id, account_id=acct.id, model_name="dentalsegmentator")
    db.add(job)
    db.commit()

    assert job.status == "queued"
    assert job.error is None
    assert job.mask_storage_key is None
    assert job.created_at is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_db_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stomserver.db'`

- [ ] **Step 3: Implement**

Create `src/stomserver/db/__init__.py` (empty):

```python
```

Create `src/stomserver/db/models.py`:

```python
"""SQLAlchemy ORM models for accounts, tokens, studies and jobs."""

from __future__ import annotations

import datetime

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime.datetime] = mapped_column(default=_utcnow)


class ApiToken(Base):
    __tablename__ = "api_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"))
    created_at: Mapped[datetime.datetime] = mapped_column(default=_utcnow)


class Study(Base):
    __tablename__ = "studies"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(512))
    shape: Mapped[str] = mapped_column(String(64))      # JSON list, e.g. "[8, 16, 16]"
    spacing: Mapped[str] = mapped_column(String(64))    # JSON list
    created_at: Mapped[datetime.datetime] = mapped_column(default=_utcnow)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    study_id: Mapped[int] = mapped_column(ForeignKey("studies.id"), index=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    mask_storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    model_name: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime.datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(default=_utcnow, onupdate=_utcnow)
```

Create `src/stomserver/db/session.py`:

```python
"""Engine and session factory helpers."""

from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


def make_engine(db_url: str) -> Engine:
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_all(engine: Engine) -> None:
    Base.metadata.create_all(engine)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_db_models.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomserver/db/ tests/test_db_models.py
git commit -m "feat: add SQLAlchemy models and session helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Token hashing + label map

Two small, pure pieces used widely downstream: `hash_token` (shared by the admin script and the auth dependency) and the DentalSegmentator label map.

**Files:**
- Create: `src/stomserver/auth.py` (hash function only in this task; the FastAPI dependency is added in Task 7)
- Create: `src/stomserver/segmentation/__init__.py`, `src/stomserver/segmentation/labels.py`
- Test: `tests/test_auth.py`, `tests/test_labels.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_auth.py`:

```python
from stomserver.auth import hash_token


def test_hash_is_deterministic_and_hex64():
    h1 = hash_token("secret-token")
    h2 = hash_token("secret-token")
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_different_tokens_differ():
    assert hash_token("a") != hash_token("b")
```

Create `tests/test_labels.py`:

```python
from stomcore.mask import LabelInfo
from stomserver.segmentation.labels import DENTALSEGMENTATOR_LABELS


def test_label_map_has_five_structures():
    assert set(DENTALSEGMENTATOR_LABELS.keys()) == {1, 2, 3, 4, 5}


def test_labels_are_labelinfo_with_matching_ids():
    for label_id, info in DENTALSEGMENTATOR_LABELS.items():
        assert isinstance(info, LabelInfo)
        assert info.label_id == label_id
        assert len(info.color) == 3
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_auth.py tests/test_labels.py -v`
Expected: FAIL — `ModuleNotFoundError` for `stomserver.auth` / `stomserver.segmentation.labels`

- [ ] **Step 3: Implement**

Create `src/stomserver/auth.py`:

```python
"""Token hashing and (later) the request auth dependency."""

from __future__ import annotations

import hashlib


def hash_token(token: str) -> str:
    """SHA-256 hex digest of a raw token. Only the hash is ever stored."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
```

Create `src/stomserver/segmentation/__init__.py` (empty):

```python
```

Create `src/stomserver/segmentation/labels.py`:

```python
"""Fixed label map for the DentalSegmentator model output.

NOTE: label ids/order must match the model's dataset.json. Verify against the
downloaded weights in Task 14 and adjust names/ids if the model differs.
"""

from __future__ import annotations

from stomcore.mask import LabelInfo

DENTALSEGMENTATOR_LABELS: dict[int, LabelInfo] = {
    1: LabelInfo(1, "maxilla-upper-skull", (230, 200, 160)),
    2: LabelInfo(2, "mandible", (200, 170, 130)),
    3: LabelInfo(3, "upper-teeth", (255, 255, 240)),
    4: LabelInfo(4, "lower-teeth", (245, 245, 230)),
    5: LabelInfo(5, "mandibular-canal", (220, 80, 80)),
}
```

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_auth.py tests/test_labels.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomserver/auth.py src/stomserver/segmentation/__init__.py src/stomserver/segmentation/labels.py tests/test_auth.py tests/test_labels.py
git commit -m "feat: add token hashing and DentalSegmentator label map

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Queue abstraction

A tiny `JobQueue` protocol plus the RQ implementation. Tests and the app use the protocol; a `FakeQueue` for tests is defined in `conftest.py` in Task 7.

**Files:**
- Create: `src/stomserver/queue/__init__.py`, `src/stomserver/queue/base.py`, `src/stomserver/queue/rq_queue.py`
- Test: `tests/test_queue_rq.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_queue_rq.py`:

```python
import fakeredis

from stomserver.queue.rq_queue import RqJobQueue


def test_enqueue_segmentation_pushes_job(monkeypatch):
    fake = fakeredis.FakeStrictRedis()
    q = RqJobQueue(fake, queue_name="test-seg")
    q.enqueue_segmentation(42)

    import rq
    registry = rq.Queue("test-seg", connection=fake)
    assert registry.count == 1
    enqueued = registry.jobs[0]
    assert enqueued.args == (42,)
    assert enqueued.func_name == "stomserver.segmentation.worker.run_segmentation"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_queue_rq.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stomserver.queue'`

- [ ] **Step 3: Implement**

Create `src/stomserver/queue/__init__.py` (empty):

```python
```

Create `src/stomserver/queue/base.py`:

```python
"""Job queue interface."""

from __future__ import annotations

from typing import Protocol


class JobQueue(Protocol):
    def enqueue_segmentation(self, job_id: int) -> None: ...
```

Create `src/stomserver/queue/rq_queue.py`:

```python
"""RQ-backed job queue."""

from __future__ import annotations

import rq

_WORKER_FUNC = "stomserver.segmentation.worker.run_segmentation"


class RqJobQueue:
    def __init__(self, redis_conn, queue_name: str = "segmentation") -> None:
        self._queue = rq.Queue(queue_name, connection=redis_conn)

    def enqueue_segmentation(self, job_id: int) -> None:
        self._queue.enqueue(_WORKER_FUNC, job_id)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_queue_rq.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomserver/queue/ tests/test_queue_rq.py
git commit -m "feat: add JobQueue protocol and RQ implementation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: FastAPI app factory, deps, errors, auth dependency, healthz

Assemble the app with dependency-injected DB/storage/queue read from `app.state`, a global error handler returning `{detail, code}`, the `get_current_account` auth dependency, and a `/healthz` route. Also add shared test fixtures (`conftest.py`).

**Files:**
- Create: `src/stomserver/api/__init__.py`, `src/stomserver/api/deps.py`, `src/stomserver/api/errors.py`, `src/stomserver/api/app.py`, `src/stomserver/api/schemas.py`
- Modify: `src/stomserver/auth.py` (append the dependency)
- Modify: `tests/conftest.py` (append fixtures)
- Test: `tests/test_api_health.py`

- [ ] **Step 1: Append shared fixtures to `tests/conftest.py`**

Append to `tests/conftest.py`:

```python
import numpy as np
import pytest
from fastapi.testclient import TestClient

from stomcore.geometry import Geometry
from stomcore.nifti_io import save_volume_nifti
from stomcore.volume import Volume
from stomserver.api.app import create_app
from stomserver.auth import hash_token
from stomserver.db.models import Account, ApiToken
from stomserver.db.session import create_all, make_engine, make_session_factory
from stomserver.storage.local import LocalFileStorage


class FakeQueue:
    """Records enqueued job ids; optionally runs them synchronously."""

    def __init__(self):
        self.enqueued: list[int] = []
        self._sync_handler = None

    def set_sync_handler(self, handler):
        self._sync_handler = handler

    def enqueue_segmentation(self, job_id: int) -> None:
        self.enqueued.append(job_id)
        if self._sync_handler is not None:
            self._sync_handler(job_id)


@pytest.fixture
def db_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    create_all(engine)
    return make_session_factory(engine)


@pytest.fixture
def storage(tmp_path):
    return LocalFileStorage(str(tmp_path / "storage"))


@pytest.fixture
def queue():
    return FakeQueue()


@pytest.fixture
def account_token(db_factory):
    db = db_factory()
    acct = Account(name="Clinic A")
    db.add(acct)
    db.flush()
    raw_token = "test-token-A"
    db.add(ApiToken(token_hash=hash_token(raw_token), account_id=acct.id))
    db.commit()
    return acct.id, raw_token


@pytest.fixture
def client(db_factory, storage, queue):
    app = create_app(session_factory=db_factory, storage=storage, queue=queue)
    return TestClient(app)


def _nifti_bytes(tmp_path, shape=(8, 16, 16), name="vol.nii.gz"):
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    vol = Volume(np.arange(int(np.prod(shape)), dtype=np.int16).reshape(shape), geo)
    path = tmp_path / name
    save_volume_nifti(vol, path)
    return path.read_bytes()


@pytest.fixture
def nifti_bytes(tmp_path):
    return _nifti_bytes(tmp_path)
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_api_health.py`:

```python
def test_healthz_no_auth(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api_health.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stomserver.api'`

- [ ] **Step 4: Implement**

Create `src/stomserver/api/__init__.py` (empty):

```python
```

Create `src/stomserver/api/schemas.py`:

```python
"""Pydantic response models."""

from __future__ import annotations

from pydantic import BaseModel


class StudyCreated(BaseModel):
    study_id: int
    shape: list[int]
    spacing: list[float]


class JobStatus(BaseModel):
    job_id: int
    status: str
    error: str | None = None
```

Create `src/stomserver/api/deps.py`:

```python
"""Request-scoped dependencies, backed by objects on app.state."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from ..storage.base import Storage


def get_db(request: Request) -> Iterator[Session]:
    session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_queue(request: Request):
    return request.app.state.queue
```

Create `src/stomserver/api/errors.py`:

```python
"""Uniform JSON error responses: {detail, code}."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.status_code},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": "validation error", "code": 422},
        )
```

Append to `src/stomserver/auth.py`:

```python
from fastapi import Depends, Header, HTTPException

from .api.deps import get_db
from .db.models import Account, ApiToken


def get_current_account(
    authorization: str | None = Header(default=None),
    db=Depends(get_db),
) -> Account:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization[len("Bearer "):]
    row = db.query(ApiToken).filter_by(token_hash=hash_token(token)).first()
    if row is None:
        raise HTTPException(status_code=401, detail="invalid token")
    account = db.get(Account, row.account_id)
    if account is None:
        raise HTTPException(status_code=401, detail="invalid token")
    return account
```

Create `src/stomserver/api/app.py`:

```python
"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from ..config import Config, load_config
from ..queue.rq_queue import RqJobQueue
from ..storage.base import Storage
from ..storage.local import LocalFileStorage
from .errors import install_error_handlers


def create_app(
    config: Config | None = None,
    *,
    session_factory=None,
    storage: Storage | None = None,
    queue=None,
) -> FastAPI:
    cfg = config or load_config()
    app = FastAPI(title="stomserver")

    if session_factory is None:
        from ..db.session import create_all, make_engine, make_session_factory

        engine = make_engine(cfg.db_url)
        create_all(engine)
        session_factory = make_session_factory(engine)
    app.state.session_factory = session_factory
    app.state.storage = storage or LocalFileStorage(cfg.storage_dir)

    if queue is None:
        import redis

        queue = RqJobQueue(redis.Redis.from_url(cfg.redis_url))
    app.state.queue = queue
    app.state.config = cfg

    install_error_handlers(app)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    from .routes_jobs import router as jobs_router
    from .routes_studies import router as studies_router

    app.include_router(studies_router)
    app.include_router(jobs_router)
    return app
```

> NOTE: `routes_studies` and `routes_jobs` are created in Tasks 8–10. To keep this task runnable on its own, create **stub routers now** and fill them in later tasks.

Create `src/stomserver/api/routes_studies.py`:

```python
"""Study routes (filled in Tasks 8-9, masks in Task 10)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
```

Create `src/stomserver/api/routes_jobs.py`:

```python
"""Job routes (filled in Task 9)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_api_health.py -v`
Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add src/stomserver/api/ src/stomserver/auth.py tests/conftest.py tests/test_api_health.py
git commit -m "feat: add FastAPI app factory, deps, error handlers, auth dependency

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `POST /studies` — upload + validate + store

Accept a `.nii.gz` upload, validate it via `stomcore.load_volume_nifti`, store it, and create a `Study`. Enforce auth.

**Files:**
- Modify: `src/stomserver/api/routes_studies.py`
- Test: `tests/test_api_studies.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api_studies.py`:

```python
def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_upload_requires_auth(client, nifti_bytes):
    r = client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)})
    assert r.status_code == 401
    assert r.json()["code"] == 401


def test_upload_success(client, account_token, nifti_bytes):
    _, token = account_token
    r = client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)},
                    headers=_auth(token))
    assert r.status_code == 201
    body = r.json()
    assert body["shape"] == [8, 16, 16]
    assert body["spacing"] == [0.3, 0.3, 0.3]
    assert isinstance(body["study_id"], int)


def test_upload_rejects_bad_nifti(client, account_token):
    _, token = account_token
    r = client.post("/studies", files={"file": ("bad.nii.gz", b"not a nifti")},
                    headers=_auth(token))
    assert r.status_code == 400
    assert r.json()["code"] == 400
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api_studies.py -v`
Expected: FAIL — 404/405 (route not implemented yet)

- [ ] **Step 3: Implement**

Replace `src/stomserver/api/routes_studies.py` with:

```python
"""Study routes: upload (this task), masks (Task 10)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..auth import get_current_account
from ..db.models import Account, Study
from ..storage.base import Storage
from .deps import get_db, get_storage
from .schemas import StudyCreated

router = APIRouter()


def _study_key(account_id: int, study_id: int, name: str) -> str:
    return f"{account_id}/studies/{study_id}/{name}"


@router.post("/studies", response_model=StudyCreated, status_code=201)
def create_study(
    file: UploadFile = File(...),
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> StudyCreated:
    raw = file.file.read()

    # Validate by parsing through stomcore; write to a temp file first.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "upload.nii.gz"
        tmp_path.write_bytes(raw)
        try:
            from stomcore.nifti_io import load_volume_nifti

            volume = load_volume_nifti(tmp_path)
        except Exception as exc:  # noqa: BLE001 - any read failure is a bad upload
            raise HTTPException(status_code=400, detail=f"invalid NIfTI: {exc}") from exc

    study = Study(
        account_id=account.id,
        original_filename=file.filename or "upload.nii.gz",
        storage_key="",  # set after we know the id
        shape=json.dumps(list(volume.shape)),
        spacing=json.dumps([float(s) for s in volume.geometry.spacing]),
    )
    db.add(study)
    db.flush()  # assigns study.id

    key = _study_key(account.id, study.id, "volume.nii.gz")
    study.storage_key = key
    storage.put(key, raw)
    db.commit()

    return StudyCreated(
        study_id=study.id,
        shape=list(volume.shape),
        spacing=[float(s) for s in volume.geometry.spacing],
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_api_studies.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomserver/api/routes_studies.py tests/test_api_studies.py
git commit -m "feat: add POST /studies upload with validation and storage

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: `POST /studies/{id}/segment` + `GET /jobs/{id}`

Enqueue a segmentation job (with account isolation) and expose job status.

**Files:**
- Modify: `src/stomserver/api/routes_studies.py` (add segment endpoint)
- Modify: `src/stomserver/api/routes_jobs.py`
- Test: `tests/test_api_jobs.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api_jobs.py`:

```python
def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _upload(client, token, nifti_bytes):
    r = client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)},
                    headers=_auth(token))
    return r.json()["study_id"]


def test_segment_enqueues_job(client, account_token, queue, nifti_bytes):
    _, token = account_token
    study_id = _upload(client, token, nifti_bytes)
    r = client.post(f"/studies/{study_id}/segment", headers=_auth(token))
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "queued"
    assert queue.enqueued == [job_id]


def test_segment_unknown_study_404(client, account_token):
    _, token = account_token
    r = client.post("/studies/999/segment", headers=_auth(token))
    assert r.status_code == 404


def test_job_status(client, account_token, nifti_bytes):
    _, token = account_token
    study_id = _upload(client, token, nifti_bytes)
    job_id = client.post(f"/studies/{study_id}/segment", headers=_auth(token)).json()["job_id"]
    r = client.get(f"/jobs/{job_id}", headers=_auth(token))
    assert r.status_code == 200
    assert r.json() == {"job_id": job_id, "status": "queued", "error": None}


def test_job_isolation_across_accounts(client, db_factory, account_token, nifti_bytes):
    from stomserver.auth import hash_token
    from stomserver.db.models import Account, ApiToken

    _, token_a = account_token
    study_id = _upload(client, token_a, nifti_bytes)
    job_id = client.post(f"/studies/{study_id}/segment", headers=_auth(token_a)).json()["job_id"]

    # Second account
    db = db_factory()
    acct_b = Account(name="Clinic B")
    db.add(acct_b)
    db.flush()
    db.add(ApiToken(token_hash=hash_token("token-B"), account_id=acct_b.id))
    db.commit()

    r = client.get(f"/jobs/{job_id}", headers=_auth("token-B"))
    assert r.status_code == 404
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api_jobs.py -v`
Expected: FAIL — 404/405 on segment/jobs routes

- [ ] **Step 3: Implement**

Append to `src/stomserver/api/routes_studies.py` (add imports `Job`, `get_queue`, `JobStatus` at top; then the endpoint):

At the top of the file, extend the imports:

```python
from ..db.models import Account, Job, Study
from .deps import get_db, get_queue, get_storage
from .schemas import JobStatus, StudyCreated
```

Append this endpoint to the file:

```python
@router.post("/studies/{study_id}/segment", response_model=JobStatus, status_code=202)
def segment_study(
    study_id: int,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
    queue=Depends(get_queue),
) -> JobStatus:
    study = db.query(Study).filter_by(id=study_id, account_id=account.id).first()
    if study is None:
        raise HTTPException(status_code=404, detail="study not found")

    job = Job(study_id=study.id, account_id=account.id, model_name="dentalsegmentator")
    db.add(job)
    db.commit()

    try:
        queue.enqueue_segmentation(job.id)
    except Exception as exc:  # noqa: BLE001 - queue/Redis down
        raise HTTPException(status_code=503, detail=f"queue unavailable: {exc}") from exc

    return JobStatus(job_id=job.id, status=job.status, error=job.error)
```

Replace `src/stomserver/api/routes_jobs.py` with:

```python
"""Job status route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_account
from ..db.models import Account, Job
from .deps import get_db
from .schemas import JobStatus

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(
    job_id: int,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
) -> JobStatus:
    job = db.query(Job).filter_by(id=job_id, account_id=account.id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(job_id=job.id, status=job.status, error=job.error)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_api_jobs.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomserver/api/routes_studies.py src/stomserver/api/routes_jobs.py tests/test_api_jobs.py
git commit -m "feat: add segment enqueue and job status endpoints with account isolation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Segmentation runner interface + FakeRunner + worker core

Define `SegmentationRunner`, a deterministic `FakeRunner`, and the testable worker core `_run_segmentation` (the RQ entrypoint `run_segmentation` wires real deps from env). The real `DentalSegmentatorRunner` body is added in Task 14; create it now raising `NotImplementedError` until then.

**Files:**
- Create: `src/stomserver/segmentation/runner.py`, `src/stomserver/segmentation/worker.py`
- Test: `tests/test_worker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_worker.py`:

```python
import numpy as np

from stomcore.mask_io import load_mask_nifti
from stomserver.db.models import Account, Job, Study
from stomserver.segmentation.runner import FakeRunner
from stomserver.segmentation.worker import _run_segmentation


def _seed_study_job(db_factory, storage, tmp_path):
    from stomcore.geometry import Geometry
    from stomcore.nifti_io import save_volume_nifti
    from stomcore.volume import Volume

    db = db_factory()
    acct = Account(name="A")
    db.add(acct)
    db.flush()
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    vol = Volume(np.zeros((6, 5, 4), dtype=np.int16), geo)
    vpath = tmp_path / "v.nii.gz"
    save_volume_nifti(vol, vpath)
    key = f"{acct.id}/studies/1/volume.nii.gz"
    storage.put(key, vpath.read_bytes())
    study = Study(account_id=acct.id, original_filename="v.nii.gz", storage_key=key,
                  shape="[6, 5, 4]", spacing="[0.3, 0.3, 0.3]")
    db.add(study)
    db.flush()
    job = Job(study_id=study.id, account_id=acct.id, model_name="dentalsegmentator")
    db.add(job)
    db.commit()
    return job.id, acct.id


def test_worker_success(db_factory, storage, tmp_path):
    job_id, acct_id = _seed_study_job(db_factory, storage, tmp_path)
    _run_segmentation(job_id, db_factory, storage, FakeRunner())

    db = db_factory()
    job = db.get(Job, job_id)
    assert job.status == "done"
    assert job.error is None
    assert job.mask_storage_key == f"{acct_id}/studies/1/mask.nii.gz"
    assert storage.exists(job.mask_storage_key)
    assert storage.exists(f"{acct_id}/studies/1/mask_labels.json")


def test_worker_marks_failed_on_runner_error(db_factory, storage, tmp_path):
    job_id, _ = _seed_study_job(db_factory, storage, tmp_path)

    class BoomRunner:
        def predict(self, volume):
            raise RuntimeError("inference exploded")

    _run_segmentation(job_id, db_factory, storage, BoomRunner())

    db = db_factory()
    job = db.get(Job, job_id)
    assert job.status == "failed"
    assert "inference exploded" in job.error


def test_fake_runner_returns_matching_shape():
    from stomcore.geometry import Geometry
    from stomcore.volume import Volume

    vol = Volume(np.zeros((6, 5, 4), dtype=np.int16), Geometry.identity(spacing=(1, 1, 1)))
    labels = FakeRunner().predict(vol)
    assert labels.shape == (6, 5, 4)
    assert set(np.unique(labels)).issubset({0, 1, 2, 3, 4, 5})
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stomserver.segmentation.runner'`

- [ ] **Step 3: Implement**

Create `src/stomserver/segmentation/runner.py`:

```python
"""Segmentation runner: interface, deterministic fake, and real nnU-Net runner."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from stomcore.volume import Volume


class SegmentationRunner(Protocol):
    def predict(self, volume: Volume) -> np.ndarray:
        """Return a label volume [z, y, x] matching the input volume shape."""
        ...


class FakeRunner:
    """Deterministic stand-in: labels a few fixed voxels. No model needed."""

    def predict(self, volume: Volume) -> np.ndarray:
        labels = np.zeros(volume.shape, dtype=np.uint16)
        flat = labels.reshape(-1)
        # Assign labels 1..5 to the first few voxels deterministically.
        for i in range(min(5, flat.size)):
            flat[i] = i + 1
        return labels


class DentalSegmentatorRunner:
    """Real nnU-Net v2 runner. Body implemented in Task 14."""

    def __init__(self, model_dir: str) -> None:
        self._model_dir = model_dir

    def predict(self, volume: Volume) -> np.ndarray:
        raise NotImplementedError("DentalSegmentatorRunner.predict added in Task 14")
```

Create `src/stomserver/segmentation/worker.py`:

```python
"""Segmentation worker: RQ entrypoint + testable core."""

from __future__ import annotations

import tempfile
from pathlib import Path

from stomcore.mask import SegmentationMask
from stomcore.mask_io import save_mask_nifti
from stomcore.nifti_io import load_volume_nifti

from ..db.models import Job, Study
from ..storage.base import Storage
from .labels import DENTALSEGMENTATOR_LABELS
from .runner import SegmentationRunner


def _mask_key(account_id: int, study_id: int, name: str) -> str:
    return f"{account_id}/studies/{study_id}/{name}"


def _run_segmentation(job_id: int, session_factory, storage: Storage,
                      runner: SegmentationRunner) -> None:
    """Testable core: load volume, run runner, save mask, update job."""
    db = session_factory()
    try:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.status = "running"
        db.commit()

        study = db.get(Study, job.study_id)
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / "volume.nii.gz"
            vpath.write_bytes(storage.get(study.storage_key))
            volume = load_volume_nifti(vpath)

            labels = runner.predict(volume)
            mask = SegmentationMask(labels, volume.geometry, DENTALSEGMENTATOR_LABELS)
            if not mask.is_compatible_with(volume):
                raise ValueError("predicted mask geometry does not match volume")

            mask_nifti = Path(tmp) / "mask.nii.gz"
            mask_labels = Path(tmp) / "mask_labels.json"
            save_mask_nifti(mask, mask_nifti, mask_labels)

            mask_key = _mask_key(job.account_id, study.id, "mask.nii.gz")
            labels_key = _mask_key(job.account_id, study.id, "mask_labels.json")
            storage.put(mask_key, mask_nifti.read_bytes())
            storage.put(labels_key, mask_labels.read_bytes())

        job.mask_storage_key = mask_key
        job.status = "done"
        job.error = None
        db.commit()
    except Exception as exc:  # noqa: BLE001 - any failure marks the job failed
        db.rollback()
        job = db.get(Job, job_id)
        if job is not None:
            job.status = "failed"
            job.error = str(exc)
            db.commit()
    finally:
        db.close()


def run_segmentation(job_id: int) -> None:
    """RQ entrypoint: build real dependencies from environment/config."""
    from ..config import load_config
    from ..db.session import make_engine, make_session_factory
    from ..storage.local import LocalFileStorage
    from .runner import DentalSegmentatorRunner

    cfg = load_config()
    engine = make_engine(cfg.db_url)
    session_factory = make_session_factory(engine)
    storage = LocalFileStorage(cfg.storage_dir)
    runner = DentalSegmentatorRunner(cfg.model_dir)
    _run_segmentation(job_id, session_factory, storage, runner)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_worker.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomserver/segmentation/runner.py src/stomserver/segmentation/worker.py tests/test_worker.py
git commit -m "feat: add segmentation runner interface, FakeRunner, and worker core

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: `GET /studies/{id}/masks` and `/masks/labels`

Serve the mask `.nii.gz` and the labels JSON; return `409` until the job is `done`.

**Files:**
- Modify: `src/stomserver/api/routes_studies.py`
- Test: `tests/test_api_masks.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api_masks.py`:

```python
from stomserver.segmentation.runner import FakeRunner
from stomserver.segmentation.worker import _run_segmentation


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _upload(client, token, nifti_bytes):
    return client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)},
                       headers=_auth(token)).json()["study_id"]


def test_masks_409_before_done(client, account_token, nifti_bytes):
    _, token = account_token
    study_id = _upload(client, token, nifti_bytes)
    client.post(f"/studies/{study_id}/segment", headers=_auth(token))
    r = client.get(f"/studies/{study_id}/masks", headers=_auth(token))
    assert r.status_code == 409


def test_masks_served_after_done(client, db_factory, storage, account_token, nifti_bytes):
    _, token = account_token
    study_id = _upload(client, token, nifti_bytes)
    job_id = client.post(f"/studies/{study_id}/segment", headers=_auth(token)).json()["job_id"]

    # Run the worker synchronously to completion.
    _run_segmentation(job_id, db_factory, storage, FakeRunner())

    r = client.get(f"/studies/{study_id}/masks", headers=_auth(token))
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/gzip"
    assert len(r.content) > 0

    r2 = client.get(f"/studies/{study_id}/masks/labels", headers=_auth(token))
    assert r2.status_code == 200
    assert "mandibular-canal" in r2.text
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_api_masks.py -v`
Expected: FAIL — 404 (mask routes not implemented)

- [ ] **Step 3: Implement**

Append to `src/stomserver/api/routes_studies.py` (add `Response` to the FastAPI import line and `StorageKeyError` import at top):

At the top, extend imports:

```python
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from ..storage.base import Storage, StorageKeyError
```

Append these endpoints:

```python
def _latest_done_job(db: Session, study: Study) -> Job | None:
    return (
        db.query(Job)
        .filter_by(study_id=study.id, status="done")
        .order_by(Job.id.desc())
        .first()
    )


@router.get("/studies/{study_id}/masks")
def get_mask(
    study_id: int,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> Response:
    study = db.query(Study).filter_by(id=study_id, account_id=account.id).first()
    if study is None:
        raise HTTPException(status_code=404, detail="study not found")
    job = _latest_done_job(db, study)
    if job is None or not job.mask_storage_key:
        raise HTTPException(status_code=409, detail="mask not ready")
    try:
        data = storage.get(job.mask_storage_key)
    except StorageKeyError:
        raise HTTPException(status_code=404, detail="mask file missing")
    return Response(content=data, media_type="application/gzip")


@router.get("/studies/{study_id}/masks/labels")
def get_mask_labels(
    study_id: int,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
) -> Response:
    study = db.query(Study).filter_by(id=study_id, account_id=account.id).first()
    if study is None:
        raise HTTPException(status_code=404, detail="study not found")
    job = _latest_done_job(db, study)
    if job is None:
        raise HTTPException(status_code=409, detail="mask not ready")
    labels_key = job.mask_storage_key.rsplit("/", 1)[0] + "/mask_labels.json"
    try:
        data = storage.get(labels_key)
    except StorageKeyError:
        raise HTTPException(status_code=404, detail="labels file missing")
    return Response(content=data, media_type="application/json")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_api_masks.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomserver/api/routes_studies.py tests/test_api_masks.py
git commit -m "feat: add mask and labels download endpoints

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: End-to-end integration test

One test exercising the whole pipeline through the API with the worker run synchronously via the `FakeQueue` sync handler.

**Files:**
- Test: `tests/test_integration_e2e.py`

- [ ] **Step 1: Write the test**

Create `tests/test_integration_e2e.py`:

```python
import io

from stomcore.mask_io import load_mask_nifti
from stomcore.nifti_io import load_volume_nifti
from stomserver.segmentation.runner import FakeRunner
from stomserver.segmentation.worker import _run_segmentation


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_full_pipeline(client, db_factory, storage, account_token, nifti_bytes, tmp_path):
    _, token = account_token

    # Wire the queue to run the worker synchronously on enqueue.
    client.app.state.queue.set_sync_handler(
        lambda job_id: _run_segmentation(job_id, db_factory, storage, FakeRunner())
    )

    study_id = client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)},
                           headers=_auth(token)).json()["study_id"]
    job = client.post(f"/studies/{study_id}/segment", headers=_auth(token)).json()
    assert client.get(f"/jobs/{job['job_id']}", headers=_auth(token)).json()["status"] == "done"

    mask_resp = client.get(f"/studies/{study_id}/masks", headers=_auth(token))
    labels_resp = client.get(f"/studies/{study_id}/masks/labels", headers=_auth(token))
    assert mask_resp.status_code == 200 and labels_resp.status_code == 200

    # Persist returned files and load them back; geometry must match the uploaded volume.
    (tmp_path / "m.nii.gz").write_bytes(mask_resp.content)
    (tmp_path / "m.json").write_bytes(labels_resp.content)
    mask = load_mask_nifti(tmp_path / "m.nii.gz", tmp_path / "m.json")

    (tmp_path / "v.nii.gz").write_bytes(nifti_bytes)
    volume = load_volume_nifti(tmp_path / "v.nii.gz")
    assert mask.is_compatible_with(volume)
    assert mask.shape == volume.shape
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_integration_e2e.py -v`
Expected: `1 passed`

> If `client.app` is not available in your FastAPI/Starlette version, use the `queue` fixture directly: add `queue` to the test signature and call `queue.set_sync_handler(...)` (the `client` fixture builds the app with that same `queue` instance).

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_e2e.py
git commit -m "test: add end-to-end pipeline integration test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Admin script, weights downloader, NOTICE

`create_account.py` issues a token (prints once); `download_weights.py` fetches the DentalSegmentator weights; `NOTICE` records attribution.

**Files:**
- Create: `scripts/create_account.py`, `scripts/download_weights.py`, `NOTICE`
- Test: `tests/test_create_account_script.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_create_account_script.py`:

```python
from scripts.create_account import create_account
from stomserver.auth import hash_token
from stomserver.db.models import Account, ApiToken
from stomserver.db.session import create_all, make_engine, make_session_factory


def test_create_account_issues_token():
    engine = make_engine("sqlite://")
    create_all(engine)
    factory = make_session_factory(engine)

    token = create_account(factory, "Clinic X")
    assert isinstance(token, str) and len(token) > 20

    db = factory()
    acct = db.query(Account).filter_by(name="Clinic X").one()
    row = db.query(ApiToken).filter_by(token_hash=hash_token(token)).one()
    assert row.account_id == acct.id
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_create_account_script.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts'` or `scripts.create_account`

- [ ] **Step 3: Implement**

Create `scripts/__init__.py` (empty):

```python
```

Create `scripts/create_account.py`:

```python
"""Admin CLI: create an Account and issue an API token (printed once)."""

from __future__ import annotations

import argparse
import secrets

from stomserver.auth import hash_token
from stomserver.config import load_config
from stomserver.db.models import Account, ApiToken
from stomserver.db.session import create_all, make_engine, make_session_factory


def create_account(session_factory, name: str) -> str:
    """Create an account + token; return the RAW token (store it, it's shown once)."""
    db = session_factory()
    try:
        account = Account(name=name)
        db.add(account)
        db.flush()
        token = secrets.token_urlsafe(32)
        db.add(ApiToken(token_hash=hash_token(token), account_id=account.id))
        db.commit()
        return token
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create an account and issue an API token.")
    parser.add_argument("name", help="account / clinic name")
    args = parser.parse_args(argv)

    cfg = load_config()
    engine = make_engine(cfg.db_url)
    create_all(engine)
    factory = make_session_factory(engine)
    token = create_account(factory, args.name)
    print(f"account '{args.name}' created. API token (store it now, shown once):")
    print(token)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

Create `scripts/download_weights.py`:

```python
"""Download DentalSegmentator nnU-Net weights from Zenodo into MODEL_DIR.

Weights are NOT committed to git. See NOTICE for attribution (CC-BY 4.0).
Record: https://zenodo.org/records/10829675
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

from stomserver.config import load_config

WEIGHTS_URL = (
    "https://zenodo.org/records/10829675/files/"
    "Dataset112_DentalSegmentator_v100.zip?download=1"
)
WEIGHTS_ZIP = "Dataset112_DentalSegmentator_v100.zip"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download DentalSegmentator weights.")
    parser.add_argument("--model-dir", default=load_config().model_dir,
                        help="target directory for weights")
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    zip_path = model_dir / WEIGHTS_ZIP

    print(f"downloading weights to {zip_path} ...")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, zip_path)
    except Exception as exc:  # noqa: BLE001
        print(f"error: download failed: {exc}", file=sys.stderr)
        return 1

    print("extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(model_dir)
    print(f"done. weights in {model_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

Create `NOTICE`:

```
This product uses the DentalSegmentator model and the nnU-Net framework.

DentalSegmentator (pretrained nnU-Net v2 weights, CC-BY 4.0):
  G. Dot et al., "DentalSegmentator: robust deep learning-based CBCT image
  segmentation." Weights: https://zenodo.org/records/10829675

nnU-Net:
  Isensee, F., et al. "nnU-Net: a self-configuring method for deep
  learning-based biomedical image segmentation." Nature Methods (2021).

Used under the Creative Commons Attribution 4.0 International license.
```

Add to `.gitignore` (append): `models/` and `*.zip`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_create_account_script.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/ NOTICE .gitignore tests/test_create_account_script.py
git commit -m "feat: add create_account admin CLI, weights downloader, NOTICE

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Real `DentalSegmentatorRunner` + slow smoke test

Implement the real nnU-Net v2 inference inside `DentalSegmentatorRunner.predict` and add a `@pytest.mark.slow` smoke test that is skipped when weights are not present. This is the only task that touches `nnunetv2`.

**Files:**
- Modify: `src/stomserver/segmentation/runner.py`
- Test: `tests/test_runner_real.py`

- [ ] **Step 1: Write the slow smoke test (skipped without weights)**

Create `tests/test_runner_real.py`:

```python
import os

import numpy as np
import pytest

from stomcore.geometry import Geometry
from stomcore.volume import Volume
from stomserver.segmentation.runner import DentalSegmentatorRunner

WEIGHTS_DIR = os.environ.get("STOM_MODEL_DIR", "./models")


def _weights_present() -> bool:
    from pathlib import Path
    return any(Path(WEIGHTS_DIR).glob("**/dataset.json"))


@pytest.mark.slow
@pytest.mark.skipif(not _weights_present(), reason="DentalSegmentator weights not downloaded")
def test_real_runner_predicts_matching_shape():
    geo = Geometry.identity(spacing=(0.4, 0.4, 0.4))
    vol = Volume(np.zeros((32, 32, 32), dtype=np.int16), geo)
    runner = DentalSegmentatorRunner(WEIGHTS_DIR)
    labels = runner.predict(vol)
    assert labels.shape == vol.shape
```

- [ ] **Step 2: Run to confirm it is collected and skipped**

Run: `.venv/bin/python -m pytest tests/test_runner_real.py -v`
Expected: `1 skipped` (weights not present). The default suite never runs the real model.

- [ ] **Step 3: Implement the real runner**

Replace the `DentalSegmentatorRunner` class in `src/stomserver/segmentation/runner.py` with:

```python
class DentalSegmentatorRunner:
    """Real nnU-Net v2 runner for the DentalSegmentator model.

    model_dir must be an nnU-Net results folder containing the trained model
    (a `Dataset112_*` folder with `dataset.json` and `plans.json`). Inference
    runs on CPU when no GPU is available.
    """

    def __init__(self, model_dir: str) -> None:
        self._model_dir = model_dir

    def predict(self, volume: Volume) -> np.ndarray:
        import tempfile
        from pathlib import Path

        import SimpleITK as sitk
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        from stomcore.sitk_interop import sitk_from_volume

        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            # nnU-Net expects <case>_<channel:04d>.nii.gz
            sitk.WriteImage(sitk_from_volume(volume), str(in_dir / "case_0000.nii.gz"),
                            useCompression=True)

            predictor = nnUNetPredictor(
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                allow_tqdm=False,
            )
            predictor.initialize_from_trained_model_folder(
                self._model_dir,
                use_folds=("all",),
                checkpoint_name="checkpoint_final.pth",
            )
            predictor.predict_from_files(
                str(in_dir), str(out_dir),
                save_probabilities=False, overwrite=True,
            )
            result = sitk.ReadImage(str(out_dir / "case.nii.gz"))
            return sitk.GetArrayFromImage(result).astype(np.uint16)
```

> NOTE: the exact `use_folds`/`checkpoint_name`/model-folder layout depends on how `download_weights.py` extracts the Zenodo archive. After downloading, inspect the extracted folder: the directory passed to `initialize_from_trained_model_folder` must be the one containing `dataset.json`, `plans.json`, and `fold_*/` (or `fold_all/`). Adjust `use_folds` and `checkpoint_name` to match what is actually present, and verify the label ids in `dataset.json` match `DENTALSEGMENTATOR_LABELS` (update `labels.py` if they differ). This verification requires the downloaded weights and is expected to be done when running the slow test for real.

- [ ] **Step 4: Confirm the default suite still passes and the slow test still skips**

Run: `.venv/bin/python -m pytest -q`
Expected: all non-slow tests pass; `test_runner_real` shows as skipped.

- [ ] **Step 5: Commit**

```bash
git add src/stomserver/segmentation/runner.py tests/test_runner_real.py
git commit -m "feat: implement real DentalSegmentatorRunner with slow smoke test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Full-suite green + README

Confirm the entire suite is green and document how to run the server, worker, and admin script.

**Files:**
- Create: `src/stomserver/README.md`
- Test: full suite

- [ ] **Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (Plan 1 + Plan 2), with `test_runner_real` skipped. Report the exact counts.

- [ ] **Step 2: Write `src/stomserver/README.md`**

Create `src/stomserver/README.md`:

```markdown
# stomserver

Cloud backend for CBCT segmentation. Depends on `stomcore`.

## Run (dev)

Install: `pip install -e ".[dev,server]"` (and `".[nnunet]"` for real inference).

1. Create an account + token:
   `python scripts/create_account.py "Clinic A"`  → prints the token once.
2. Download model weights (once):
   `python scripts/download_weights.py`  → into `STOM_MODEL_DIR` (default `./models`).
3. Start Redis (native): `redis-server` (default `redis://localhost:6379/0`).
4. Start the API: `uvicorn "stomserver.api.app:create_app" --factory --reload`.
5. Start a worker: `rq worker segmentation` (with `STOM_*` env vars set).

## Config (env vars)

- `STOM_DB_URL` (default `sqlite:///stom.db`)
- `STOM_STORAGE_DIR` (default `./storage`)
- `STOM_REDIS_URL` (default `redis://localhost:6379/0`)
- `STOM_MODEL_DIR` (default `./models`)
- `STOM_MAX_UPLOAD_BYTES` (default 500 MB)

## API

- `POST /studies` (multipart `.nii.gz`) → `{study_id, shape, spacing}`
- `POST /studies/{id}/segment` → `{job_id, status}`
- `GET /jobs/{id}` → `{job_id, status, error?}`
- `GET /studies/{id}/masks` → `mask.nii.gz`
- `GET /studies/{id}/masks/labels` → `mask_labels.json`
- `GET /healthz`

All except `/healthz` require `Authorization: Bearer <token>`.
```

- [ ] **Step 3: Commit**

```bash
git add src/stomserver/README.md
git commit -m "docs: add stomserver README and confirm full suite green

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- §3 components: API factory (T7), worker (T10/T14), queue (T6), DB (T4), Storage (T3), config (T2). ✓
- §4 data model + storage keys + geometry invariant: models (T4), LocalFileStorage keys (T3/T8/T10), `is_compatible_with` check in worker (T10). ✓
- §5 API + auth (Bearer/account isolation): auth (T5/T7), endpoints (T8/T9/T11), 401/404/400/409/503 (T8/T9/T11). ✓
- §6 worker + `stomcore.mask_io` + runner/FakeRunner + weights script + NOTICE: T1, T10, T13, T14. ✓
- §7 error handling: global handler (T7), per-endpoint statuses (T8/T9/T11), worker failed-not-crash (T10). ✓
- §8 testing (unit/API/worker/integration/slow): T1,T3,T4,T5,T6,T8,T9,T10,T11,T12,T13,T14. ✓
- Token issuance (`create_account.py`, shared `hash_token`): T5 (hash) + T13 (script). ✓

**Placeholder scan:** No TBD/TODO. Two tasks (T7 stub routers, T10 `NotImplementedError` runner) are explicitly deferred-and-filled in later named tasks (T8–11, T14), with the interim state runnable and tested — not placeholders, but staged construction.

**Type consistency:** `Config` fields (T2) used in T7/T10/T13/T14. `Storage.put/get/exists/delete` + `StorageKeyError` (T3) used in T8/T10/T11. Models `Account/ApiToken/Study/Job` field names (T4) used consistently in T7–T13. `hash_token` (T5) used in T7/T13 and conftest. `enqueue_segmentation` (T6) used in T9/conftest. `SegmentationRunner.predict`/`FakeRunner` (T10) used in T11/T12/T14. `_run_segmentation(job_id, session_factory, storage, runner)` signature identical across T10/T11/T12. `DENTALSEGMENTATOR_LABELS` (T5) used in T10. `save_mask_nifti/load_mask_nifti` (T1) used in T10/T11/T12.
