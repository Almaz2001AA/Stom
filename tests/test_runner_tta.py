"""TTA (test-time mirroring) + CPU-thread speed controls.

TTA roughly octuples CPU inference time for a small accuracy gain, so it is OFF
by default — a ~10 min local segmentation drops to ~1.5-2 min. Full-accuracy
mirroring is opt-in via STOM_ENABLE_TTA. These tests cover the env parsing and
that the runner records the resolved choice without needing torch/nnU-Net.
"""

import os

from stomengine.runner import DentalSegmentatorRunner, num_threads, tta_enabled


def test_tta_disabled_by_default():
    assert tta_enabled({}) is False


def test_enable_tta_truthy_values():
    for val in ("1", "true", "True", "YES", "on", " on "):
        assert tta_enabled({"STOM_ENABLE_TTA": val}) is True, val


def test_enable_tta_falsy_values_keep_tta_off():
    for val in ("", "0", "false", "no", "off"):
        assert tta_enabled({"STOM_ENABLE_TTA": val}) is False, val


def test_runner_defaults_use_tta_from_env(monkeypatch):
    monkeypatch.delenv("STOM_ENABLE_TTA", raising=False)
    assert DentalSegmentatorRunner("/nonexistent")._use_tta is False

    monkeypatch.setenv("STOM_ENABLE_TTA", "1")
    assert DentalSegmentatorRunner("/nonexistent")._use_tta is True


def test_runner_explicit_use_tta_overrides_env(monkeypatch):
    monkeypatch.delenv("STOM_ENABLE_TTA", raising=False)
    assert DentalSegmentatorRunner("/nonexistent", use_tta=True)._use_tta is True
    monkeypatch.setenv("STOM_ENABLE_TTA", "1")
    assert DentalSegmentatorRunner("/nonexistent", use_tta=False)._use_tta is False


def test_num_threads_defaults_to_all_cores():
    assert num_threads({}) == (os.cpu_count() or 1)


def test_num_threads_env_override():
    assert num_threads({"STOM_NUM_THREADS": "4"}) == 4


def test_num_threads_ignores_invalid_override():
    for val in ("", "0", "-2", "abc"):
        assert num_threads({"STOM_NUM_THREADS": val}) == (os.cpu_count() or 1), val
