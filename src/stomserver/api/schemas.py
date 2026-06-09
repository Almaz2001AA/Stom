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
