"""Job queue interface."""

from __future__ import annotations

from typing import Protocol


class JobQueue(Protocol):
    def enqueue_segmentation(self, job_id: int) -> None: ...
