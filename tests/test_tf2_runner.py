"""Lightweight (no-model) tests for ToothFairy2Runner helpers."""

import numpy as np

from stomengine.runner import ToothFairy2Runner, clamp_air_padding


def test_clamp_air_padding_raises_extreme_padding():
    vox = np.array([[-2048, -1000, 0, 500]], dtype=np.int16)
    out = clamp_air_padding(vox)
    assert out.min() == -1000
    assert out[0, 0] == -1000  # extreme padding lifted to the air floor
    assert out[0, 3] == 500    # foreground untouched
    # original is not mutated
    assert vox[0, 0] == -2048


def test_clamp_air_padding_noop_when_nothing_below_floor():
    vox = np.array([[-1000, 0, 1000]], dtype=np.int16)
    out = clamp_air_padding(vox)
    assert out is vox  # returns the same array, no copy


def test_runner_stores_config_without_loading_model():
    # Construction must not touch the (possibly absent) weights.
    runner = ToothFairy2Runner("/no/such/model", use_tta=True, low_memory=False)
    assert runner._model_dir == "/no/such/model"
    assert runner._use_tta is True
    assert runner._low_memory is False
