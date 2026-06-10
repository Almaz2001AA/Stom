"""Segmentation runner: interface, deterministic fake, and real nnU-Net runner."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from stomcore.geometry import Geometry
from stomcore.volume import Volume


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

    def __init__(self, model_dir: str) -> None:
        self._model_dir = model_dir

    def predict(self, volume: Volume) -> tuple[np.ndarray, Geometry]:
        import tempfile
        from pathlib import Path

        import SimpleITK as sitk
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        from stomcore.sitk_interop import geometry_from_sitk, sitk_from_volume

        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            # nnU-Net expects <case>_<channel:04d>.nii.gz
            sitk.WriteImage(sitk_from_volume(volume), str(in_dir / "case_0000.nii.gz"),
                            useCompression=True)

            predictor = nnUNetPredictor(
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
