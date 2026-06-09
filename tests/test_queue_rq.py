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
