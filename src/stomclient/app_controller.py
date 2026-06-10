"""Qt-agnostic session state machine driving cloud + rendering state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from enum import Enum

from stomcore.mask import SegmentationMask
from stomcore.volume import Volume

from . import slice_renderer as sr
from .measurement import LinearMeasurement, MeasurementSet
from .serialization import mask_from_bytes, volume_to_nifti_bytes


class State(str, Enum):
    EMPTY = "empty"
    LOADED = "loaded"
    UPLOADING = "uploading"
    SEGMENTING = "segmenting"
    MASK_READY = "mask_ready"
    FAILED = "failed"


class AppController:
    def __init__(self, cloud_client, on_change: Callable[[], None] = lambda: None) -> None:
        self._cloud = cloud_client
        self._on_change = on_change
        self.state = State.EMPTY
        self.volume: Volume | None = None
        self.mask = None
        self.plane = sr.AXIAL
        self.index = 0
        self.window_center = 0.0
        self.window_width = 1.0
        self.measurements = MeasurementSet()
        self.study_id: int | None = None
        self.job_id: int | None = None
        self.error: str | None = None

    def _changed(self) -> None:
        self._on_change()

    def load_volume(self, volume: Volume) -> None:
        self.volume = volume
        self.mask = None
        self.plane = sr.AXIAL
        self.index = sr.slice_count(volume, sr.AXIAL) // 2
        self.window_center, self.window_width = sr.default_window_level(volume)
        self.measurements = MeasurementSet()
        self.study_id = self.job_id = None
        self.error = None
        self.state = State.LOADED
        self._changed()

    def set_plane(self, plane: str) -> None:
        self.plane = plane
        count = sr.slice_count(self.volume, plane)
        self.index = min(self.index, count - 1)
        self._changed()

    def set_index(self, index: int) -> None:
        count = sr.slice_count(self.volume, self.plane)
        self.index = max(0, min(index, count - 1))
        self._changed()

    def set_window_level(self, center: float, width: float) -> None:
        self.window_center = center
        self.window_width = max(width, 1.0)
        self._changed()

    def set_label_visible(self, label_id: int, visible: bool) -> None:
        if self.mask is None:
            return
        info = self.mask.label_map.get(label_id)
        if info is None:
            return
        new_map = dict(self.mask.label_map)
        new_map[label_id] = replace(info, visible=visible)
        self.mask = SegmentationMask(self.mask.labels, self.mask.geometry, new_map)
        self._changed()

    def add_measurement(self, p0: tuple[float, float], p1: tuple[float, float]) -> None:
        if self.volume is None:
            return
        self.measurements.add(
            LinearMeasurement(p0, p1, self.plane, self.volume.geometry)
        )
        self._changed()

    def clear_measurements(self) -> None:
        self.measurements.clear()
        self._changed()

    def submit(self) -> None:
        if self.state not in (State.LOADED, State.FAILED, State.MASK_READY):
            raise RuntimeError(f"cannot submit from state {self.state}")
        self.error = None
        self.state = State.UPLOADING
        self._changed()
        try:
            nifti = volume_to_nifti_bytes(self.volume)
            info = self._cloud.upload_study(nifti, "study.nii.gz")
            self.study_id = info.study_id
            job = self._cloud.start_segmentation(info.study_id)
            self.job_id = job.job_id
        except Exception as exc:  # noqa: BLE001 - any failure must leave a retryable FAILED state
            self.state = State.FAILED
            self.error = str(exc)
            self._changed()
            raise
        self.state = State.SEGMENTING
        self._changed()

    def poll(self) -> bool:
        """Poll once. Returns True when the job reached a terminal state."""
        if self.state is not State.SEGMENTING:
            return True
        job = self._cloud.poll_status(self.job_id)
        if job.status == "failed":
            self.error = job.error or "segmentation failed"
            self.state = State.FAILED
            self._changed()
            return True
        if job.status == "done":
            mask_bytes, labels_bytes = self._cloud.download_mask(self.study_id)
            mask = mask_from_bytes(mask_bytes, labels_bytes)
            if not mask.is_compatible_with(self.volume):
                self.error = "returned mask geometry does not match volume"
                self.state = State.FAILED
                self._changed()
                return True
            self.mask = mask
            self.state = State.MASK_READY
            self._changed()
            return True
        return False
