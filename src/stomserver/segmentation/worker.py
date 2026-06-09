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
