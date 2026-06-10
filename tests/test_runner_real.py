import os

import numpy as np
import pytest

from stomcore.geometry import Geometry
from stomcore.volume import Volume
from stomserver.segmentation.runner import DentalSegmentatorRunner

WEIGHTS_DIR = os.environ.get("STOM_MODEL_DIR", "./models")


def _model_dir() -> str | None:
    """Locate the nnU-Net trained-model folder (the one holding dataset.json).

    Accepts either the folder itself (STOM_MODEL_DIR) or any parent of it
    (e.g. the default ./models that the weights were extracted into).
    """
    from pathlib import Path
    hits = sorted(Path(WEIGHTS_DIR).glob("**/dataset.json"))
    return str(hits[0].parent) if hits else None


@pytest.mark.slow
@pytest.mark.skipif(_model_dir() is None, reason="DentalSegmentator weights not downloaded")
def test_real_runner_predicts_matching_shape():
    geo = Geometry.identity(spacing=(0.4, 0.4, 0.4))
    vol = Volume(np.zeros((32, 32, 32), dtype=np.int16), geo)
    runner = DentalSegmentatorRunner(_model_dir())
    labels, geometry = runner.predict(vol)
    assert labels.shape == vol.shape
    assert geometry.is_compatible(geo)
