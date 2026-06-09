import os

import numpy as np
import pytest
import SimpleITK as sitk


def _write_dicom_series(
    directory,
    n_slices,
    rows=16,
    cols=16,
    spacing=(0.3, 0.3, 0.3),
    series_uid: str = "1.2.826.0.1.3680043.2.1125.1.1234567890",
    name_prefix: str = "slice",
):
    """Write a minimal valid CT DICOM series into `directory`. Returns the dir path."""
    arr = (np.arange(n_slices * rows * cols).reshape(n_slices, rows, cols) % 1000).astype(np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    tags = {
        "0008|0060": "CT",                 # Modality
        "0020|000e": series_uid,           # Series Instance UID
        "0008|0016": "1.2.840.10008.5.1.4.1.1.2",  # SOP Class UID (CT Image Storage)
        "0028|0100": "16",                 # Bits Allocated
        "0028|0101": "16",                 # Bits Stored
        "0028|0102": "15",                 # High Bit
        "0028|0103": "1",                  # Pixel Representation (signed)
    }
    for i in range(img.GetDepth()):
        slice_i = img[:, :, i]
        for tag, value in tags.items():
            slice_i.SetMetaData(tag, value)
        position = img.TransformIndexToPhysicalPoint((0, 0, i))
        slice_i.SetMetaData("0020|0032", "\\".join(f"{c:.4f}" for c in position))  # Image Position
        slice_i.SetMetaData("0020|0013", str(i))  # Instance Number
        slice_i.SetMetaData("0008|0018", f"{series_uid}.{i}")  # SOP Instance UID
        writer.SetFileName(os.path.join(directory, f"{name_prefix}_{i:03d}.dcm"))
        writer.Execute(slice_i)
    return directory


@pytest.fixture
def dicom_series(tmp_path):
    """A valid 8-slice synthetic CT series directory."""
    d = tmp_path / "series"
    d.mkdir()
    return _write_dicom_series(str(d), n_slices=8)


@pytest.fixture
def single_slice_series(tmp_path):
    """A degenerate 1-slice series directory."""
    d = tmp_path / "single"
    d.mkdir()
    return _write_dicom_series(str(d), n_slices=1)


@pytest.fixture
def multi_series(tmp_path):
    """A directory containing two distinct DICOM series."""
    d = tmp_path / "multi"
    d.mkdir()
    _write_dicom_series(
        str(d), n_slices=4,
        series_uid="1.2.826.0.1.3680043.2.1125.1.1111111111", name_prefix="a",
    )
    _write_dicom_series(
        str(d), n_slices=4,
        series_uid="1.2.826.0.1.3680043.2.1125.1.2222222222", name_prefix="b",
    )
    return str(d)
