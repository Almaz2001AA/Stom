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


# Default smallest connected component (per label) kept by the denoise pass.
# CBCT speckle is typically a handful of voxels (<2 mm³); real structures — even
# a thin slice of the mandibular canal — form large connected runs, so 8 mm³
# clears the noise with a wide safety margin. Tunable via STOM_MIN_COMPONENT_MM3.
DEFAULT_MIN_COMPONENT_MM3 = 8.0


def postprocess_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether to denoise the raw prediction. On unless ``STOM_DISABLE_POSTPROCESS``."""
    env = os.environ if env is None else env
    return env.get("STOM_DISABLE_POSTPROCESS", "").strip().lower() not in _TRUTHY_VALUES


def min_component_mm3(env: Mapping[str, str] | None = None) -> float:
    """Denoise volume threshold in mm³ (``STOM_MIN_COMPONENT_MM3`` or the default)."""
    env = os.environ if env is None else env
    raw = env.get("STOM_MIN_COMPONENT_MM3", "").strip()
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_MIN_COMPONENT_MM3
    return value if value >= 0 else DEFAULT_MIN_COMPONENT_MM3


def denoise_labels(
    labels: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    min_mm3: float = DEFAULT_MIN_COMPONENT_MM3,
) -> np.ndarray:
    """Drop small disconnected islands from a label volume — the main noise source.

    The raw nnU-Net argmax scatters tiny false-positive blobs through the volume.
    For each label we find 3-D connected components (26-neighbour) and zero out
    any whose physical volume is below ``min_mm3``; real anatomy (teeth, jaws,
    canal) forms components far larger than the threshold, so it is untouched.
    ``min_mm3 <= 0`` disables the pass. Pure post-processing — no model needed.
    """
    if min_mm3 <= 0:
        return labels
    from scipy import ndimage  # engine dependency (pulled by nnU-Net/skimage)

    voxel_mm3 = abs(float(spacing[0]) * float(spacing[1]) * float(spacing[2]))
    if voxel_mm3 <= 0:
        return labels
    min_voxels = max(1, int(round(min_mm3 / voxel_mm3)))

    cleaned = labels.copy()
    structure = ndimage.generate_binary_structure(3, 3)  # full 26-connectivity
    for value in np.unique(labels):
        if value == 0:
            continue
        components, n = ndimage.label(labels == value, structure=structure)
        if n == 0:
            continue
        counts = np.bincount(components.reshape(-1))
        small = np.nonzero(counts[1:] < min_voxels)[0] + 1
        if small.size:
            cleaned[np.isin(components, small)] = 0
    return cleaned

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
            geometry = geometry_from_sitk(result)
            # Denoise: drop the small false-positive islands nnU-Net scatters
            # through the raw argmax (the dominant source of visible noise).
            if postprocess_enabled():
                labels = denoise_labels(labels, geometry.spacing,
                                        min_mm3=min_component_mm3())
            return labels, geometry


def clamp_air_padding(
    voxels: np.ndarray, air_value: float = _AIR_THRESHOLD
) -> np.ndarray:
    """Clamp out-of-FOV padding to a typical air value for per-image ZScore.

    ToothFairy2 normalises each image by its own mean/std (ZScoreNormalization),
    so extreme scanner padding (e.g. ``-2048``) would skew those statistics.
    Clamping everything below ``air_value`` up to it keeps the foreground stats
    representative. Returns the array unchanged if nothing is below the floor.
    """
    if not np.any(voxels < air_value):
        return voxels
    out = voxels.copy()
    out[out < air_value] = air_value
    return out


class ToothFairy2Runner:
    """nnU-Net v2 runner for the 49-class ToothFairy2 per-tooth (FDI) model.

    Differs from :class:`DentalSegmentatorRunner` in three ways: it uses
    per-image ZScore normalisation (so the raw scan is fed as-is — only the
    out-of-FOV air padding is clamped — instead of being harmonised to a fixed
    HU domain), it loads ``checkpoint_best.pth``, and it exports through the
    memory-frugal path in :mod:`stomengine.lowmem` because the 49-class logits
    buffer is far too large (~13 GB) for nnU-Net's default float32 export on the
    6-8 GB PCs we target. Set ``STOM_DISABLE_LOWMEM=1`` to force the in-RAM path.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        use_tta: bool | None = None,
        low_memory: bool | None = None,
    ) -> None:
        self._model_dir = model_dir
        self._use_tta = tta_enabled() if use_tta is None else use_tta
        self._low_memory = low_memory

    def _build_predictor(self, low_memory: bool):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        kwargs = dict(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            use_mirroring=self._use_tta,
            allow_tqdm=False,
        )
        if low_memory:
            from .lowmem import MemmapLogitsPredictor

            class _LowMemPredictor(MemmapLogitsPredictor, nnUNetPredictor):
                pass

            return _LowMemPredictor(**kwargs)
        return nnUNetPredictor(**kwargs)

    def predict(
        self, volume: Volume, *, progress: ProgressCb | None = None
    ) -> tuple[np.ndarray, Geometry]:
        import tempfile
        from pathlib import Path

        import SimpleITK as sitk
        import torch
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
        from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy

        from stomcore.sitk_interop import sitk_from_volume

        from .lowmem import logits_to_labels_lowmem, low_memory_enabled

        torch.set_num_threads(num_threads())
        low_memory = low_memory_enabled() if self._low_memory is None else self._low_memory

        # Per-image ZScore: feed the raw scan, only clamping out-of-FOV padding.
        clamped = Volume(clamp_air_padding(volume.voxels), volume.geometry)

        with tempfile.TemporaryDirectory() as tmp:
            in_path = Path(tmp) / "case_0000.mha"
            sitk.WriteImage(sitk_from_volume(clamped), str(in_path),
                            useCompression=True)

            predictor = self._build_predictor(low_memory)
            if low_memory:
                predictor.logits_memmap_dir = tmp
            predictor.initialize_from_trained_model_folder(
                self._model_dir,
                use_folds=(0,),  # pretrained ToothFairy2 ships a single fold_0
                checkpoint_name="checkpoint_best.pth",
            )

            # Replicate predict_single_npy_array up to the logits, then take the
            # memory-frugal label export instead of nnU-Net's float32 one.
            data, props = SimpleITKIO().read_images([str(in_path)])
            ppa = PreprocessAdapterFromNpy(
                [data], [None], [props], [None],
                predictor.plans_manager, predictor.dataset_json,
                predictor.configuration_manager,
                num_threads_in_multithreaded=1, verbose=False,
            )
            dct = next(ppa)
            with _tile_progress(progress):
                logits = predictor.predict_logits_from_preprocessed_data(dct["data"])
            labels = logits_to_labels_lowmem(predictor, logits, dct["data_properties"])
            del logits  # release the (possibly memmap-backed) buffer before cleanup

        labels = labels.astype(np.uint16)
        geometry = volume.geometry
        if postprocess_enabled():
            labels = denoise_labels(labels, geometry.spacing,
                                    min_mm3=min_component_mm3())
        return labels, geometry
