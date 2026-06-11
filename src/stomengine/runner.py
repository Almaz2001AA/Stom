"""Segmentation runner: interface, deterministic fake, and real nnU-Net runner."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Protocol

import numpy as np

from stomcore.geometry import Geometry
from stomcore.volume import Volume

_TTA_DISABLE_VALUES = {"1", "true", "yes", "on"}


def tta_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether test-time augmentation (mirroring) is on for inference.

    TTA evaluates every mirror of each tile, roughly octupling CPU inference
    time for a small accuracy gain. Set ``STOM_DISABLE_TTA`` to a truthy value
    (``1``/``true``/``yes``/``on``) to turn it off for ~8x faster local
    segmentation (≈1.5-2 min instead of ~10). Default: enabled.
    """
    env = os.environ if env is None else env
    return env.get("STOM_DISABLE_TTA", "").strip().lower() not in _TTA_DISABLE_VALUES

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
    def predict(self, volume: Volume) -> tuple[np.ndarray, Geometry]:
        """Return ``(labels, geometry)`` for the prediction.

        ``labels`` is a [z, y, x] label volume; ``geometry`` is the spatial
        geometry of that prediction. The caller verifies it is compatible with
        the input volume, so a runner that silently alters shape/spacing/origin
        is detected rather than trusted.
        """
        ...


class FakeRunner:
    """Deterministic stand-in: labels a few fixed voxels. No model needed."""

    def predict(self, volume: Volume) -> tuple[np.ndarray, Geometry]:
        labels = np.zeros(volume.shape, dtype=np.uint16)
        flat = labels.reshape(-1)
        for i in range(min(5, flat.size)):
            flat[i] = i + 1
        return labels, volume.geometry


class DentalSegmentatorRunner:
    """Real nnU-Net v2 runner for the DentalSegmentator model.

    model_dir must be an nnU-Net results folder containing the trained model
    (a `Dataset112_*` folder with `dataset.json` and `plans.json`). Inference
    runs on CPU when no GPU is available.
    """

    def __init__(self, model_dir: str, *, use_tta: bool | None = None) -> None:
        self._model_dir = model_dir
        # TTA mirroring ~8x's CPU inference; resolve from STOM_DISABLE_TTA unless
        # an explicit choice is passed.
        self._use_tta = tta_enabled() if use_tta is None else use_tta

    def predict(self, volume: Volume) -> tuple[np.ndarray, Geometry]:
        import tempfile
        from pathlib import Path

        import SimpleITK as sitk
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        from stomcore.sitk_interop import geometry_from_sitk, sitk_from_volume

        # Map this scanner's intensities onto the model's training domain so
        # non-HU CBCT does not segment to an empty mask.
        harmonized = Volume(harmonize_to_model_domain(volume.voxels), volume.geometry)

        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            # nnU-Net expects <case>_<channel:04d>.nii.gz
            sitk.WriteImage(sitk_from_volume(harmonized), str(in_dir / "case_0000.nii.gz"),
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
            predictor.predict_from_files(
                str(in_dir), str(out_dir),
                save_probabilities=False, overwrite=True,
            )
            result = sitk.ReadImage(str(out_dir / "case.nii.gz"))
            labels = sitk.GetArrayFromImage(result).astype(np.uint16)
            return labels, geometry_from_sitk(result)
