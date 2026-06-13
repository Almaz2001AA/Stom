import numpy as np

from stomengine.runner import (
    DEFAULT_MIN_COMPONENT_MM3,
    denoise_labels,
    min_component_mm3,
    postprocess_enabled,
)


def test_removes_small_islands_keeps_large_structure():
    labels = np.zeros((20, 20, 20), dtype=np.uint16)
    labels[4:14, 4:14, 4:14] = 3        # a solid 10^3 = 1000-voxel "tooth"
    labels[0, 0, 0] = 3                  # a 1-voxel speckle
    labels[19, 19, 19] = 3              # another speckle
    out = denoise_labels(labels, (1.0, 1.0, 1.0), min_mm3=8.0)
    assert (out == 3).sum() == 1000     # block kept, both speckles gone
    assert out[0, 0, 0] == 0 and out[19, 19, 19] == 0


def test_denoise_is_per_label():
    labels = np.zeros((25, 25, 25), dtype=np.uint16)
    labels[2:12, 2:12, 2:12] = 2        # big mandible blob (1000 voxels)
    labels[22, 22, 22] = 5              # isolated 1-voxel canal speckle -> noise
    labels[16:19, 16:19, 16:19] = 5     # 27-voxel canal piece, far away -> kept
    out = denoise_labels(labels, (1.0, 1.0, 1.0), min_mm3=8.0)
    assert (out == 2).sum() == 1000
    assert out[22, 22, 22] == 0         # 1-voxel speckle removed
    assert (out == 5).sum() == 27       # 27-voxel piece survives (27 mm³ > 8)


def test_threshold_scales_with_spacing():
    # The same 27-voxel blob is noise at fine spacing but real at coarse spacing.
    labels = np.zeros((30, 30, 30), dtype=np.uint16)
    labels[10:13, 10:13, 10:13] = 4     # 27 voxels
    fine = denoise_labels(labels, (0.3, 0.3, 0.3), min_mm3=8.0)
    assert (fine == 4).sum() == 0       # 27 * 0.027 = 0.73 mm³ < 8 -> cleared
    coarse = denoise_labels(labels, (1.0, 1.0, 1.0), min_mm3=8.0)
    assert (coarse == 4).sum() == 27    # 27 * 1 = 27 mm³ > 8 -> kept


def test_zero_threshold_is_passthrough():
    labels = np.zeros((10, 10, 10), dtype=np.uint16)
    labels[0, 0, 0] = 3
    out = denoise_labels(labels, (1.0, 1.0, 1.0), min_mm3=0.0)
    assert np.array_equal(out, labels)  # disabled -> nothing removed


def test_background_untouched_and_dtype_preserved():
    labels = np.zeros((8, 8, 8), dtype=np.uint16)
    labels[2:6, 2:6, 2:6] = 1
    out = denoise_labels(labels, (1.0, 1.0, 1.0))
    assert out.dtype == labels.dtype
    assert (out == 0).sum() == labels.size - 64


def test_postprocess_enabled_default_and_toggle():
    assert postprocess_enabled({}) is True
    assert postprocess_enabled({"STOM_DISABLE_POSTPROCESS": "1"}) is False
    assert postprocess_enabled({"STOM_DISABLE_POSTPROCESS": "no"}) is True


def test_min_component_mm3_reads_env():
    assert min_component_mm3({}) == DEFAULT_MIN_COMPONENT_MM3
    assert min_component_mm3({"STOM_MIN_COMPONENT_MM3": "20"}) == 20.0
    assert min_component_mm3({"STOM_MIN_COMPONENT_MM3": "junk"}) == DEFAULT_MIN_COMPONENT_MM3
