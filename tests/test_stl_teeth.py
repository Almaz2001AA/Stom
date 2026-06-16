"""Tests for per-tooth (FDI) STL export."""

import struct

import numpy as np

from stomcore.geometry import Geometry
from stomcore.mask import LabelInfo, SegmentationMask
from stomcore.stl_export import (
    FDI_TOOTH_IDS,
    export_teeth_stl,
    is_tooth_label,
)


def _block(labels, value, z, y, x):
    labels[z:z + 3, y:y + 3, x:x + 3] = value


def _tf2_like_mask():
    """A tiny mask with two anatomy labels and three FDI teeth."""
    labels = np.zeros((12, 12, 12), dtype=np.uint16)
    _block(labels, 1, 0, 0, 0)    # Lower Jawbone (anatomy)
    _block(labels, 5, 0, 0, 4)    # Left Maxillary Sinus (anatomy)
    _block(labels, 11, 4, 4, 0)   # Upper Right Central Incisor (tooth)
    _block(labels, 16, 4, 4, 4)   # Upper Right First Molar (tooth)
    _block(labels, 38, 8, 8, 8)   # Lower Left Third Molar (tooth)
    label_map = {
        1: LabelInfo(1, "Lower Jawbone", (200, 170, 130)),
        5: LabelInfo(5, "Left Maxillary Sinus", (120, 180, 230)),
        11: LabelInfo(11, "Upper Right Central Incisor", (255, 245, 200)),
        16: LabelInfo(16, "Upper Right First Molar", (251, 230, 225)),
        38: LabelInfo(38, "Lower Left Third Molar", (227, 225, 235)),
    }
    return SegmentationMask(labels, Geometry.identity((0.3, 0.3, 0.3)), label_map)


def test_fdi_set_has_32_permanent_teeth():
    assert len(FDI_TOOTH_IDS) == 32
    assert FDI_TOOTH_IDS == {q * 10 + p for q in (1, 2, 3, 4) for p in range(1, 9)}


def test_is_tooth_label_matches_fdi_only():
    assert is_tooth_label(11) and is_tooth_label(48)
    # anatomy ids and FDI gaps are not teeth
    for non_tooth in (0, 1, 5, 9, 10, 19, 20, 29, 49):
        assert not is_tooth_label(non_tooth)


def test_export_teeth_only_writes_one_file_per_tooth(tmp_path):
    mask = _tf2_like_mask()
    files = export_teeth_stl(mask, tmp_path)
    names = sorted(f.name for f in files)
    assert names == [
        "11_Upper_Right_Central_Incisor.stl",
        "16_Upper_Right_First_Molar.stl",
        "38_Lower_Left_Third_Molar.stl",
    ]
    # anatomy labels (1, 5) must not be exported
    assert not any(f.name.startswith(("01_", "05_")) for f in files)


def test_export_teeth_each_file_is_watertight(tmp_path):
    mask = _tf2_like_mask()
    files = export_teeth_stl(mask, tmp_path)
    assert files
    for path in files:
        data = path.read_bytes()
        (count,) = struct.unpack("<I", data[80:84])
        assert count > 0
        assert len(data) == 84 + count * 50  # well-formed binary STL


def test_export_teeth_empty_when_no_per_tooth_labels(tmp_path):
    # A coarse mask (DentalSegmentator-style) with teeth as one block label 3.
    labels = np.zeros((8, 8, 8), dtype=np.uint16)
    labels[2:5, 2:5, 2:5] = 3
    mask = SegmentationMask(
        labels, Geometry.identity((0.3, 0.3, 0.3)),
        {3: LabelInfo(3, "Upper Teeth", (255, 255, 240))},
    )
    assert export_teeth_stl(mask, tmp_path) == []
