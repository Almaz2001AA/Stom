import numpy as np

from stomcore.geometry import Geometry
from stomcore.mask import LabelInfo, SegmentationMask
from stomcore.volume import Volume
from stomclient.serialization import mask_from_bytes, volume_to_nifti_bytes


def test_volume_to_nifti_bytes_is_gzip_magic():
    vol = Volume(np.zeros((3, 4, 5), dtype=np.int16), Geometry.identity((0.3, 0.3, 0.3)))
    data = volume_to_nifti_bytes(vol)
    assert data[:2] == b"\x1f\x8b"  # gzip magic


def test_mask_from_bytes_roundtrip():
    geo = Geometry.identity((0.3, 0.3, 0.3))
    labels = np.zeros((3, 4, 5), dtype=np.uint16)
    labels[0, 0, 0] = 1
    label_map = {1: LabelInfo(1, "tooth", (255, 0, 0), True)}
    mask = SegmentationMask(labels, geo, label_map)

    from stomclient.serialization import mask_to_bytes

    mask_bytes, labels_bytes = mask_to_bytes(mask)
    restored = mask_from_bytes(mask_bytes, labels_bytes)

    assert restored.shape == (3, 4, 5)
    assert restored.label_map[1].name == "tooth"
    assert restored.is_compatible_with(Volume(labels, geo))
