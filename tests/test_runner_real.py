import os

import numpy as np
import pytest

from stomcore.geometry import Geometry
from stomcore.volume import Volume
from stomserver.segmentation.runner import DentalSegmentatorRunner

WEIGHTS_DIR = os.environ.get("STOM_MODEL_DIR", "./models")


def _weights_present() -> bool:
    from pathlib import Path
    return any(Path(WEIGHTS_DIR).glob("**/dataset.json"))


@pytest.mark.slow
@pytest.mark.skipif(not _weights_present(), reason="DentalSegmentator weights not downloaded")
def test_real_runner_predicts_matching_shape():
    geo = Geometry.identity(spacing=(0.4, 0.4, 0.4))
    vol = Volume(np.zeros((32, 32, 32), dtype=np.int16), geo)
    runner = DentalSegmentatorRunner(WEIGHTS_DIR)
    labels = runner.predict(vol)
    assert labels.shape == vol.shape
