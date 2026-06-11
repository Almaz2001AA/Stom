import numpy as np

from stomcore.geometry import Geometry
from stomcore.mask import LabelInfo, SegmentationMask
from stomcore.volume import Volume
from stomclient import slice_renderer as sr
from stomclient.app_controller import AppController, State
from stomclient.cloud_client import JobStatus, StudyInfo
from stomclient.serialization import mask_to_bytes


def _volume(geo=None):
    geo = geo or Geometry.identity((0.3, 0.3, 0.3))
    return Volume(np.zeros((4, 5, 6), dtype=np.int16), geo)


def _mask_bytes(geo):
    labels = np.zeros((4, 5, 6), dtype=np.uint16)
    labels[0, 0, 0] = 1
    mask = SegmentationMask(labels, geo, {1: LabelInfo(1, "t", (255, 0, 0), True)})
    return mask_to_bytes(mask)


class FakeCloud:
    def __init__(self, status_sequence, mask_bytes=(b"", b"")):
        self._statuses = list(status_sequence)
        self._mask_bytes = mask_bytes
        self.uploaded = False

    def upload_study(self, nifti_bytes, filename):
        self.uploaded = True
        return StudyInfo(study_id=1, shape=[4, 5, 6], spacing=[0.3, 0.3, 0.3])

    def start_segmentation(self, study_id):
        return JobStatus(job_id=9, status="queued")

    def poll_status(self, job_id):
        return self._statuses.pop(0)

    def download_mask(self, study_id):
        return self._mask_bytes


def test_load_volume_centers_index_and_sets_state():
    c = AppController(FakeCloud([]))
    c.load_volume(_volume())
    assert c.state == State.LOADED
    assert c.plane == sr.AXIAL
    assert c.index == sr.slice_count(c.volume, sr.AXIAL) // 2


def test_submit_transitions_to_segmenting():
    cloud = FakeCloud([])
    c = AppController(cloud)
    c.load_volume(_volume())
    c.submit()
    assert cloud.uploaded is True
    assert c.state == State.SEGMENTING
    assert c.study_id == 1
    assert c.job_id == 9


def test_poll_running_then_done_loads_mask():
    geo = Geometry.identity((0.3, 0.3, 0.3))
    cloud = FakeCloud(
        [JobStatus(9, "running"), JobStatus(9, "done")],
        mask_bytes=_mask_bytes(geo),
    )
    c = AppController(cloud)
    c.load_volume(_volume(geo))
    c.submit()
    assert c.poll() is False           # still running
    assert c.state == State.SEGMENTING
    assert c.poll() is True            # done
    assert c.state == State.MASK_READY
    assert c.mask is not None


def test_poll_failed_sets_failed_state():
    cloud = FakeCloud([JobStatus(9, "failed", error="OOM")])
    c = AppController(cloud)
    c.load_volume(_volume())
    c.submit()
    assert c.poll() is True
    assert c.state == State.FAILED
    assert "OOM" in c.error


def test_poll_done_with_incompatible_mask_is_rejected():
    vol_geo = Geometry.identity((0.3, 0.3, 0.3))
    drift_geo = Geometry.identity((1.0, 1.0, 1.0))  # different spacing
    cloud = FakeCloud([JobStatus(9, "done")], mask_bytes=_mask_bytes(drift_geo))
    c = AppController(cloud)
    c.load_volume(_volume(vol_geo))
    c.submit()
    assert c.poll() is True
    assert c.state == State.FAILED
    assert "geometry" in c.error.lower()


import pytest


def test_submit_failure_sets_failed_and_reraises():
    class BoomCloud:
        def upload_study(self, nifti_bytes, filename):
            from stomclient.cloud_client import CloudError
            raise CloudError("network down")

    c = AppController(BoomCloud())
    c.load_volume(_volume())
    with pytest.raises(Exception):
        c.submit()
    assert c.state == State.FAILED
    assert "network down" in c.error
    # machine is retryable: guard allows submit() again from FAILED
    with pytest.raises(Exception):
        c.submit()


def test_poll_without_inflight_job_returns_true():
    c = AppController(FakeCloud([]))
    c.load_volume(_volume())          # state LOADED, no job
    assert c.poll() is True           # nothing to poll
    assert c.state == State.LOADED    # unchanged


class FakeEngine:
    """Local engine stand-in: returns a prepared mask or raises."""

    def __init__(self, mask=None, boom=None):
        self._mask = mask
        self._boom = boom
        self.called = False

    def segment(self, volume):
        self.called = True
        if self._boom is not None:
            raise self._boom
        return self._mask


def _mask(geo, shape=(4, 5, 6)):
    labels = np.zeros(shape, dtype=np.uint16)
    labels[0, 0, 0] = 1
    return SegmentationMask(labels, geo, {1: LabelInfo(1, "t", (255, 0, 0), True)})


def test_local_mode_submit_produces_mask_without_cloud():
    geo = Geometry.identity((0.3, 0.3, 0.3))
    cloud = FakeCloud([])
    engine = FakeEngine(mask=_mask(geo))
    c = AppController(cloud, engine=engine)
    c.load_volume(_volume(geo))
    c.set_local_mode(True)
    c.submit()
    assert engine.called is True
    assert cloud.uploaded is False          # nothing uploaded in local mode
    assert c.state == State.MASK_READY
    assert c.mask is not None
    assert c.poll() is True                  # already terminal


def test_local_mode_failure_sets_failed_and_reraises():
    engine = FakeEngine(boom=RuntimeError("out of memory"))
    c = AppController(FakeCloud([]), engine=engine)
    c.load_volume(_volume())
    c.set_local_mode(True)
    with pytest.raises(RuntimeError):
        c.submit()
    assert c.state == State.FAILED
    assert "out of memory" in c.error


def test_local_mode_incompatible_mask_rejected():
    vol_geo = Geometry.identity((0.3, 0.3, 0.3))
    drift_geo = Geometry.identity((1.0, 1.0, 1.0))
    engine = FakeEngine(mask=_mask(drift_geo))
    c = AppController(FakeCloud([]), engine=engine)
    c.load_volume(_volume(vol_geo))
    c.set_local_mode(True)
    c.submit()
    assert c.state == State.FAILED
    assert "geometry" in c.error.lower()


def test_set_local_mode_without_engine_raises():
    c = AppController(FakeCloud([]))          # no engine wired
    assert c.local_available is False
    with pytest.raises(RuntimeError):
        c.set_local_mode(True)


def test_set_engine_enables_local_and_notifies():
    geo = Geometry.identity((0.3, 0.3, 0.3))
    fired = []
    c = AppController(FakeCloud([]), on_change=lambda: fired.append(1))
    assert c.local_available is False
    c.set_engine(FakeEngine(mask=_mask(geo)))
    assert c.local_available is True
    assert fired                              # change notification fired
    c.set_local_mode(True)                    # no longer raises


def test_set_label_visible_toggles():
    geo = Geometry.identity((0.3, 0.3, 0.3))
    c = AppController(FakeCloud([]))
    c.load_volume(_volume(geo))
    labels = np.zeros((4, 5, 6), dtype=np.uint16)
    c.mask = SegmentationMask(labels, geo, {1: LabelInfo(1, "t", (255, 0, 0), True)})
    c.set_label_visible(1, False)
    assert c.mask.label_map[1].visible is False


def test_add_and_clear_measurements():
    c = AppController(FakeCloud([]))
    c.load_volume(_volume())
    c.add_measurement((0.0, 0.0), (0.0, 10.0))
    assert len(c.measurements) == 1
    c.clear_measurements()
    assert len(c.measurements) == 0
