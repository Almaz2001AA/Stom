"""RQ-backed job queue."""

from __future__ import annotations

import rq

_WORKER_FUNC = "stomserver.segmentation.worker.run_segmentation"


class RqJobQueue:
    def __init__(self, redis_conn, queue_name: str = "segmentation") -> None:
        self._queue = rq.Queue(queue_name, connection=redis_conn)

    def enqueue_segmentation(self, job_id: int) -> None:
        self._queue.enqueue(_WORKER_FUNC, job_id)
