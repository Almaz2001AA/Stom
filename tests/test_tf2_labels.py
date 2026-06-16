"""Tests for the ToothFairy2 49-class FDI label map."""

from stomengine.tf2_labels import TOOTHFAIRY2_LABELS


def test_counts_anatomy_and_teeth():
    teeth = [k for k in TOOTHFAIRY2_LABELS if k >= 11]
    anatomy = [k for k in TOOTHFAIRY2_LABELS if k < 11]
    assert len(teeth) == 32  # full adult dentition
    assert len(anatomy) == 10  # jaws, canals, sinuses, pharynx, bridge, crown, implant


def test_na_placeholder_ids_are_omitted():
    for gap in (19, 20, 29, 30, 39, 40):
        assert gap not in TOOTHFAIRY2_LABELS


def test_fdi_numbering_and_names():
    assert TOOTHFAIRY2_LABELS[11].name == "Upper Right Central Incisor"
    assert TOOTHFAIRY2_LABELS[18].name == "Upper Right Third Molar (Wisdom Tooth)"
    assert TOOTHFAIRY2_LABELS[21].name == "Upper Left Central Incisor"
    assert TOOTHFAIRY2_LABELS[31].name == "Lower Left Central Incisor"
    assert TOOTHFAIRY2_LABELS[48].name == "Lower Right Third Molar (Wisdom Tooth)"


def test_anatomy_names():
    assert TOOTHFAIRY2_LABELS[1].name == "Lower Jawbone"
    assert TOOTHFAIRY2_LABELS[2].name == "Upper Jawbone"
    assert TOOTHFAIRY2_LABELS[3].name == "Left Inferior Alveolar Canal"
    assert TOOTHFAIRY2_LABELS[10].name == "Implant"


def test_every_label_id_matches_its_info_and_colors_valid():
    for label_id, info in TOOTHFAIRY2_LABELS.items():
        assert info.label_id == label_id
        assert len(info.color) == 3
        assert all(0 <= c <= 255 for c in info.color)
    # background is never a foreground label
    assert 0 not in TOOTHFAIRY2_LABELS
