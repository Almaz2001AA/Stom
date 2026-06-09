"""Study routes: upload (this task), masks (Task 10)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..auth import get_current_account
from ..db.models import Account, Job, Study
from ..storage.base import Storage
from .deps import get_db, get_queue, get_storage
from .schemas import JobStatus, StudyCreated

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

    shape = [int(s) for s in volume.shape]
    # NIfTI stores spacing as float32; round to undo float32 round-trip noise.
    spacing = [round(float(s), 6) for s in volume.geometry.spacing]

    study = Study(
        account_id=account.id,
        original_filename=file.filename or "upload.nii.gz",
        storage_key="",  # set after we know the id
        shape=json.dumps(shape),
        spacing=json.dumps(spacing),
    )
    db.add(study)
    db.flush()  # assigns study.id

    key = _study_key(account.id, study.id, "volume.nii.gz")
    study.storage_key = key
    storage.put(key, raw)
    db.commit()

    return StudyCreated(
        study_id=study.id,
        shape=shape,
        spacing=spacing,
    )


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
