"""RQ-backed job queue."""

from __future__ import annotations

import rq

_WORKER_FUNC = "stomserver.segmentation.worker.run_segmentation"
_FAILURE_FUNC = "stomserver.segmentation.worker.handle_job_failure"


class RqJobQueue:
    def __init__(self, redis_conn, queue_name: str = "segmentation") -> None:
        self._queue = rq.Queue(queue_name, connection=redis_conn)

    def enqueue_segmentation(self, job_id: int) -> None:
        # on_failure marks the DB job failed if the work-horse dies mid-run.
        self._queue.enqueue(_WORKER_FUNC, job_id,
                            on_failure=rq.Callback(_FAILURE_FUNC))
