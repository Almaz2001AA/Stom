import numpy as np

from stomserver.segmentation.runner import (
    MODEL_FG_MEAN,
    MODEL_FG_STD,
    harmonize_to_model_domain,
)


def test_harmonize_maps_foreground_to_model_stats():
    """A compressed/shifted CBCT foreground is re-centered to the model's domain."""
    rng = np.random.default_rng(0)
    vox = np.full((24, 24, 24), -2048, dtype=np.int16)  # air/padding
    vox[6:18, 6:18, 6:18] = rng.normal(-500, 100, (12, 12, 12)).astype(np.int16)

    out = harmonize_to_model_domain(vox)
    fg = out > -1000

    assert out.shape == vox.shape
    assert out.dtype == np.int16
    assert abs(float(out[fg].mean()) - MODEL_FG_MEAN) < 60
    assert abs(float(out[fg].std()) - MODEL_FG_STD) < 100


def test_harmonize_passthrough_on_constant_volume():
    """A degenerate (zero-variance) volume is returned unchanged, not divided by 0."""
    vox = np.zeros((8, 8, 8), dtype=np.int16)
    out = harmonize_to_model_domain(vox)
    assert np.array_equal(out, vox)


def test_harmonize_passthrough_when_no_foreground():
    """An all-air volume (nothing above the air threshold) is left as-is."""
    vox = np.full((8, 8, 8), -3000, dtype=np.int16)
    out = harmonize_to_model_domain(vox)
    assert np.array_equal(out, vox)
