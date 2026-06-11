"""Phase C: TTA (test-time mirroring) speed toggle via STOM_DISABLE_TTA.

TTA roughly octuples CPU inference time for a small accuracy gain. Disabling it
brings a ~10 min local segmentation down to ~1.5-2 min. These tests cover the
env parsing and that the runner records the resolved choice without needing
torch/nnU-Net installed.
"""

from stomengine.runner import DentalSegmentatorRunner, tta_enabled


def test_tta_enabled_by_default():
    assert tta_enabled({}) is True


def test_disable_tta_truthy_values():
    for val in ("1", "true", "True", "YES", "on", " on "):
        assert tta_enabled({"STOM_DISABLE_TTA": val}) is False, val


def test_disable_tta_falsy_values_keep_tta_on():
    for val in ("", "0", "false", "no", "off"):
        assert tta_enabled({"STOM_DISABLE_TTA": val}) is True, val


def test_runner_defaults_use_tta_from_env(monkeypatch):
    monkeypatch.delenv("STOM_DISABLE_TTA", raising=False)
    assert DentalSegmentatorRunner("/nonexistent")._use_tta is True

    monkeypatch.setenv("STOM_DISABLE_TTA", "1")
    assert DentalSegmentatorRunner("/nonexistent")._use_tta is False


def test_runner_explicit_use_tta_overrides_env(monkeypatch):
    monkeypatch.setenv("STOM_DISABLE_TTA", "1")
    assert DentalSegmentatorRunner("/nonexistent", use_tta=True)._use_tta is True
    monkeypatch.delenv("STOM_DISABLE_TTA", raising=False)
    assert DentalSegmentatorRunner("/nonexistent", use_tta=False)._use_tta is False
