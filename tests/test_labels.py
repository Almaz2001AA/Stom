from stomcore.mask import LabelInfo
from stomserver.segmentation.labels import DENTALSEGMENTATOR_LABELS


def test_label_map_has_five_structures():
    assert set(DENTALSEGMENTATOR_LABELS.keys()) == {1, 2, 3, 4, 5}


def test_labels_are_labelinfo_with_matching_ids():
    for label_id, info in DENTALSEGMENTATOR_LABELS.items():
        assert isinstance(info, LabelInfo)
        assert info.label_id == label_id
        assert len(info.color) == 3
