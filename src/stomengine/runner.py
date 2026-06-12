"""Segmentation runner: interface, deterministic fake, and real nnU-Net runner."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np

from stomcore.geometry import Geometry
from stomcore.volume import Volume

_TRUTHY_VALUES = {"1", "true", "yes", "on"}

# A progress callback receives (steps_done, steps_total) for the sliding-window
# tile loop — the long part of CPU inference — so the UI can show a percentage.
ProgressCb = Callable[[int, int], None]


def tta_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether test-time augmentation (mirroring) is on for inference.

    TTA evaluates every mirror of each tile, roughly octupling CPU inference
    time for a small accuracy gain. It is **off by default** so local CPU
    segmentation is fast (≈1.5-2 min instead of ~10); set ``STOM_ENABLE_TTA``
    to a truthy value (``1``/``true``/``yes``/``on``) to restore full-accuracy
    mirroring.
    """
    env = os.environ if env is None else env
    return env.get("STOM_ENABLE_TTA", "").strip().lower() in _TRUTHY_VALUES


def num_threads(env: Mapping[str, str] | None = None) -> int:
    """CPU threads for inference: ``STOM_NUM_THREADS`` if set, else all cores.

    Forces full CPU utilisation for the conv-heavy sliding-window inference; the
    frozen engine can otherwise default to too few threads. A positive
    ``STOM_NUM_THREADS`` overrides (e.g. to leave cores free for the desktop).
    """
    env = os.environ if env is None else env
    raw = env.get("STOM_NUM_THREADS", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return os.cpu_count() or 1

# DentalSegmentator (Dataset112) foreground intensity stats, from the weights'
# dataset_fingerprint.json. The model's CTNormalization assumes inputs live in
# this CT/HU-like domain; CBCT with a different intensity calibration must be
# mapped onto it or the model returns an empty (all-background) mask.
MODEL_FG_MEAN = 1178.26
MODEL_FG_STD = 611.71
_AIR_THRESHOLD = -1000.0


def harmonize_to_model_domain(
    voxels: np.ndarray, air_threshold: float = _AIR_THRESHOLD
) -> np.ndarray:
    """Z-score-map foreground intensities onto the model's training domain.

    Foreground (voxels above ``air_threshold``) is recentred to
    ``MODEL_FG_MEAN``/``MODEL_FG_STD`` so scanners with non-HU calibration look
    like the training data. Degenerate volumes (no foreground or ~zero variance)
    are returned unchanged.
    """
    fg = voxels > air_threshold
    if int(fg.sum()) < 1000:
        return voxels
    mu = float(voxels[fg].mean())
    sd = float(voxels[fg].std())
    if sd < 1.0:
        return voxels
    out = (voxels.astype(np.float32) - mu) / sd * MODEL_FG_STD + MODEL_FG_MEAN
    return np.clip(out, -1024.0, 4000.0).astype(np.int16)


class SegmentationRunner(Protocol):
    def predict(
        self, volume: Volume, *, progress: ProgressCb | None = None
    ) -> tuple[np.ndarray, Geometry]:
        """Return ``(labels, geometry)`` for the prediction.

        ``labels`` is a [z, y, x] label volume; ``geometry`` is the spatial
        geometry of that prediction. The caller verifies it is compatible with
        the input volume, so a runner that silently alters shape/spacing/origin
        is detected rather than trusted. ``progress``, if given, is called with
        ``(steps_done, steps_total)`` as inference advances.
        """
        ...


class FakeRunner:
    """Deterministic stand-in: labels a few fixed voxels. No model needed."""

    def predict(
        self, volume: Volume, *, progress: ProgressCb | None = None
    ) -> tuple[np.ndarray, Geometry]:
        if progress is not None:
            progress(1, 1)  # exercise the progress wiring end to end
        labels = np.zeros(volume.shape, dtype=np.uint16)
        flat = labels.reshape(-1)
        for i in range(min(5, flat.size)):
            flat[i] = i + 1
        return labels, volume.geometry


def _tile_progress(progress: ProgressCb | None):
    """Route nnU-Net's sliding-window tile loop to ``progress``.

    nnU-Net renders tile progress with a ``tqdm`` bar (``tqdm(total=n_tiles)``
    then ``pbar.update()`` per tile). We temporarily swap the ``tqdm`` symbol in
    its inference module for a tiny counter that forwards ``(done, total)`` to
    ``progress`` — no fork, no version-specific subclassing. A no-op when
    ``progress`` is None.
    """
    if progress is None:
        return contextlib.nullcontext()

    class _Bar:
        def __init__(self, iterable=None, *, total=None, **_kwargs):
            self._iterable = iterable
            if total is None and iterable is not None:
                try:
                    total = len(iterable)
                except TypeError:
                    total = None
            self.total = total
            self.n = 0
            self._report()

        def _report(self):
            if self.total:
                with contextlib.suppress(Exception):
                    progress(min(self.n, self.total), self.total)

        def update(self, n=1):
            self.n += n
            self._report()

        def __iter__(self):
            for obj in self._iterable:
                yield obj
                self.update(1)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def set_postfix(self, *_a, **_k):
            pass

        def set_description(self, *_a, **_k):
            pass

        def refresh(self):
            pass

        def close(self):
            pass

    @contextlib.contextmanager
    def _ctx():
        from nnunetv2.inference import predict_from_raw_data as _pr

        original = _pr.tqdm
        _pr.tqdm = _Bar
        try:
            yield
        finally:
            _pr.tqdm = original

    return _ctx()


class DentalSegmentatorRunner:
    """Real nnU-Net v2 runner for the DentalSegmentator model.

    model_dir must be an nnU-Net results folder containing the trained model
    (a `Dataset112_*` folder with `dataset.json` and `plans.json`). Inference
    runs on CPU when no GPU is available.
    """

    def __init__(self, model_dir: str, *, use_tta: bool | None = None) -> None:
        self._model_dir = model_dir
        # TTA mirroring ~8x's CPU inference; off by default (resolve from
        # STOM_ENABLE_TTA) unless an explicit choice is passed.
        self._use_tta = tta_enabled() if use_tta is None else use_tta

    def predict(
        self, volume: Volume, *, progress: ProgressCb | None = None
    ) -> tuple[np.ndarray, Geometry]:
        import tempfile
        from pathlib import Path

        import SimpleITK as sitk
        import torch
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        from stomcore.sitk_interop import geometry_from_sitk, sitk_from_volume

        # Use all CPU cores for the conv-heavy sliding-window inference; the
        # frozen engine can otherwise under-utilise the CPU.
        torch.set_num_threads(num_threads())

        # Map this scanner's intensities onto the model's training domain so
        # non-HU CBCT does not segment to an empty mask.
        harmonized = Volume(harmonize_to_model_domain(volume.voxels), volume.geometry)

        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "case_0000.nii.gz"
            out_trunc = Path(tmp) / "case"
            sitk.WriteImage(sitk_from_volume(harmonized), str(in_path),
                            useCompression=True)

            predictor = nnUNetPredictor(
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                use_mirroring=self._use_tta,
                allow_tqdm=False,
            )
            predictor.initialize_from_trained_model_folder(
                self._model_dir,
                use_folds=(0,),  # DentalSegmentator v1.0.0 ships a single fold_0
                checkpoint_name="checkpoint_final.pth",
            )
            # Single-array, fully in-process inference. predict_from_files spawns
            # multiprocessing.Pool workers; in the frozen Windows engine each
            # worker re-execs the exe, which caused a freeze_support fork-bomb and
            # — when launched from the windowed GUI — a console-control crash
            # (exit 0xC000013A). predict_single_npy_array spawns no child process.
            data, props = SimpleITKIO().read_images([str(in_path)])
            with _tile_progress(progress):
                predictor.predict_single_npy_array(
                    data, props, output_file_truncated=str(out_trunc)
                )
            result = sitk.ReadImage(str(out_trunc) + ".nii.gz")
            labels = sitk.GetArrayFromImage(result).astype(np.uint16)
            return labels, geometry_from_sitk(result)
