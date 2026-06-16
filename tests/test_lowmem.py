"""Tests for the memory-frugal nnU-Net export path (stomengine.lowmem)."""

import numpy as np

from stomengine.lowmem import (
    DEFAULT_ARGMAX_SLAB_Z,
    _nearest_resample_labels,
    chunked_argmax_z,
    logits_to_labels_lowmem,
    low_memory_enabled,
)


class _FakePlansManager:
    def __init__(self, transpose_backward):
        self.transpose_backward = transpose_backward


class _FakePredictor:
    """Just enough surface for logits_to_labels_lowmem."""

    def __init__(self, transpose_backward):
        self.plans_manager = _FakePlansManager(transpose_backward)


def test_low_memory_enabled_default_and_toggle():
    assert low_memory_enabled({}) is True
    assert low_memory_enabled({"STOM_DISABLE_LOWMEM": "1"}) is False
    assert low_memory_enabled({"STOM_DISABLE_LOWMEM": "yes"}) is False
    assert low_memory_enabled({"STOM_DISABLE_LOWMEM": "0"}) is True


def test_chunked_argmax_matches_full_argmax_across_slab_sizes():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((49, 37, 11, 9)).astype(np.float16)
    expected = logits.argmax(0).astype(np.uint8)
    for slab in (1, 5, DEFAULT_ARGMAX_SLAB_Z, 100):
        out = chunked_argmax_z(logits, slab_z=slab)
        assert out.dtype == np.uint8
        assert out.shape == logits.shape[1:]
        assert np.array_equal(out, expected)


def test_chunked_argmax_reads_memmap_in_slabs(tmp_path):
    # A memmap-backed buffer must argmax identically to an in-RAM one.
    shape = (6, 20, 8, 8)
    path = tmp_path / "logits.dat"
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=shape)
    rng = np.random.default_rng(2)
    mm[:] = rng.standard_normal(shape).astype(np.float16)
    mm.flush()
    out = chunked_argmax_z(mm, slab_z=4)
    assert np.array_equal(out, np.asarray(mm).argmax(0).astype(np.uint8))


def test_nearest_resample_exact_target_shape_and_noop():
    labels = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    up = _nearest_resample_labels(labels, (4, 4, 4))
    assert up.shape == (4, 4, 4)
    assert up.dtype == np.uint8
    # nearest upsample replicates voxels, introduces no new label values
    assert set(np.unique(up)).issubset(set(np.unique(labels)))
    # identical target shape returns the same array (no copy)
    assert _nearest_resample_labels(labels, (2, 2, 2)) is labels


def test_nearest_resample_downsample_picks_existing_labels():
    labels = np.zeros((4, 4, 4), dtype=np.uint8)
    labels[2:, 2:, 2:] = 7
    down = _nearest_resample_labels(labels, (2, 2, 2))
    assert down.shape == (2, 2, 2)
    assert set(np.unique(down)).issubset({0, 7})


def test_logits_to_labels_reverts_crop_and_transpose():
    # 3-class logits on a 2x2x2 net grid; identity resample; crop in z.
    logits = np.zeros((3, 2, 2, 2), dtype=np.float16)
    logits[2] = 5.0  # everything argmaxes to label 2
    props = {
        "shape_after_cropping_and_before_resampling": (2, 2, 2),
        "shape_before_cropping": (4, 2, 2),
        "bbox_used_for_cropping": [[1, 3], [0, 2], [0, 2]],
    }
    seg = logits_to_labels_lowmem(_FakePredictor([0, 1, 2]), logits, props)
    assert seg.shape == (4, 2, 2)
    assert seg.dtype == np.uint8
    # label 2 only where the crop was inserted (z in [1,3)), background elsewhere
    assert np.all(seg[1:3] == 2)
    assert np.all(seg[0] == 0) and np.all(seg[3] == 0)


def test_logits_to_labels_applies_transpose_backward():
    # Non-identity transpose_backward must be honoured (swap z<->x).
    logits = np.zeros((2, 1, 2, 3), dtype=np.float16)
    logits[1, 0, 0, 0] = 9.0  # one voxel argmaxes to label 1
    props = {
        "shape_after_cropping_and_before_resampling": (1, 2, 3),
        "shape_before_cropping": (1, 2, 3),
        "bbox_used_for_cropping": [[0, 1], [0, 2], [0, 3]],
    }
    seg = logits_to_labels_lowmem(_FakePredictor([2, 1, 0]), logits, props)
    # (1,2,3) transposed by [2,1,0] -> (3,2,1)
    assert seg.shape == (3, 2, 1)
    assert seg[0, 0, 0] == 1
    assert seg.sum() == 1
