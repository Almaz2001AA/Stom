import numpy as np
import pytest

from stomcore.geometry import Geometry
from stomcore.volume import Volume
from stomengine import DENTALSEGMENTATOR_LABELS, FakeRunner, InProcessEngine


def _volume(geo=None):
    geo = geo or Geometry.identity((0.3, 0.3, 0.3))
    return Volume(np.zeros((4, 5, 6), dtype=np.int16), geo)


def test_inprocess_engine_builds_mask_from_runner():
    engine = InProcessEngine(FakeRunner())
    vol = _volume()
    mask = engine.segment(vol)
    assert mask.is_compatible_with(vol)
    assert mask.label_map == DENTALSEGMENTATOR_LABELS


def test_inprocess_engine_rejects_incompatible_runner_output():
    class DriftRunner:
        def predict(self, volume):
            # Different spacing than the input -> incompatible mask.
            geo = Geometry.identity((1.0, 1.0, 1.0))
            return np.zeros(volume.shape, dtype=np.uint16), geo

    engine = InProcessEngine(DriftRunner())
    with pytest.raises(ValueError, match="match input volume"):
        engine.segment(_volume())
