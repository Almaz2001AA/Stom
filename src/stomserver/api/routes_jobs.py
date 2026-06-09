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
