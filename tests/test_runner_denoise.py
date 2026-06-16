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


# --- keep-largest-component -------------------------------------------------

def test_keep_largest_removes_smaller_fragments_of_a_label():
    from stomengine.runner import keep_largest_component

    labels = np.zeros((20, 20, 20), dtype=np.uint16)
    labels[2:12, 2:12, 2:12] = 1        # big "skull" block (1000 voxels)
    labels[18, 18, 18] = 1              # floating fragment of the same label
    out = keep_largest_component(labels, {1})
    assert (out == 1).sum() == 1000     # only the largest piece survives
    assert out[18, 18, 18] == 0


def test_keep_largest_leaves_other_labels_and_two_canals_intact():
    from stomengine.runner import keep_largest_component

    labels = np.zeros((20, 20, 20), dtype=np.uint16)
    # label 5 = "both canals" (two separate pieces) — NOT in the keep set.
    labels[2:6, 2:6, 2:6] = 5
    labels[2:6, 14:18, 14:18] = 5
    # label 1 has a stray fragment and is in the keep set.
    labels[8:12, 8:12, 8:12] = 1
    labels[0, 0, 0] = 1
    out = keep_largest_component(labels, {1})
    assert (out == 5).sum() == 2 * 64   # both canal pieces untouched
    assert out[0, 0, 0] == 0            # label-1 fragment removed
    assert (out == 1).sum() == 64


# --- fill-holes -------------------------------------------------------------

def test_fill_label_holes_closes_enclosed_void_only_for_label():
    from stomengine.runner import fill_label_holes

    labels = np.zeros((9, 9, 9), dtype=np.uint16)
    labels[2:7, 2:7, 2:7] = 2          # solid block of label 2
    labels[4, 4, 4] = 0                # carve an enclosed void
    out = fill_label_holes(labels, {2})
    assert out[4, 4, 4] == 2           # void filled with the label


def test_fill_holes_never_overwrites_a_neighbouring_label():
    from stomengine.runner import fill_label_holes

    labels = np.zeros((9, 9, 9), dtype=np.uint16)
    labels[2:7, 2:7, 2:7] = 2
    labels[4, 4, 4] = 3                # a different label sits inside
    out = fill_label_holes(labels, {2})
    assert out[4, 4, 4] == 3           # neighbour preserved, not clobbered


# --- integrated postprocess + toggles ---------------------------------------

def test_postprocess_default_keeps_substantial_fragments(monkeypatch):
    # By default keep-largest is OFF (it deletes real fragmented anatomy), so a
    # large disconnected piece of a single-structure label must SURVIVE — only
    # sub-threshold speckle is denoised away.
    from stomengine.runner import postprocess_labels

    monkeypatch.delenv("STOM_ENABLE_KEEP_LARGEST", raising=False)
    labels = np.zeros((20, 20, 20), dtype=np.uint16)
    labels[2:12, 2:12, 2:12] = 1       # main block (1000 voxels)
    labels[14:18, 14:18, 14:18] = 1    # a big disconnected piece (64 voxels) — real
    labels[0, 0, 0] = 1                 # 1-voxel speckle — noise
    out = postprocess_labels(labels, (1.0, 1.0, 1.0), single_component_labels={1})
    assert (out == 1).sum() == 1000 + 64   # big piece kept, speckle gone
    assert out[0, 0, 0] == 0


def test_postprocess_keep_largest_opt_in(monkeypatch):
    from stomengine.runner import postprocess_labels

    monkeypatch.setenv("STOM_ENABLE_KEEP_LARGEST", "1")
    labels = np.zeros((20, 20, 20), dtype=np.uint16)
    labels[2:12, 2:12, 2:12] = 1
    labels[14:18, 14:18, 14:18] = 1    # opt-in: this gets dropped
    out = postprocess_labels(labels, (1.0, 1.0, 1.0), single_component_labels={1})
    assert (out == 1).sum() == 1000


def test_postprocess_labels_respects_disable_env(monkeypatch):
    from stomengine.runner import postprocess_labels

    monkeypatch.setenv("STOM_DISABLE_POSTPROCESS", "1")
    labels = np.zeros((6, 6, 6), dtype=np.uint16)
    labels[0, 0, 0] = 1                 # would normally be cleaned
    labels[2:5, 2:5, 2:5] = 1
    out = postprocess_labels(labels, (1.0, 1.0, 1.0), single_component_labels={1})
    assert np.array_equal(out, labels)  # untouched when disabled


def test_keep_largest_and_fill_holes_env_toggles():
    from stomengine.runner import fill_holes_enabled, keep_largest_enabled

    assert keep_largest_enabled({}) is False  # OFF by default — unsafe on real data
    assert keep_largest_enabled({"STOM_ENABLE_KEEP_LARGEST": "1"}) is True
    assert fill_holes_enabled({}) is True
    assert fill_holes_enabled({"STOM_DISABLE_FILL_HOLES": "yes"}) is False
