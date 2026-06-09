import pytest

from stomcore.dicom_loader import DicomError, DicomLoader
from stomcore.volume import Volume


def test_loads_series_into_volume(dicom_series):
    vol = DicomLoader.load(dicom_series)
    assert isinstance(vol, Volume)
    assert vol.shape == (8, 16, 16)  # [z, y, x]
    assert vol.geometry.spacing[0] == pytest.approx(0.3)


def test_raises_when_no_series_found(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DicomError, match="no DICOM series"):
        DicomLoader.load(str(empty))


def test_raises_on_single_slice_volume(single_slice_series):
    with pytest.raises(DicomError, match="too few slices"):
        DicomLoader.load(single_slice_series)


def test_raises_when_directory_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(DicomError, match="not a directory"):
        DicomLoader.load(str(missing))


def test_raises_on_multiple_series(multi_series):
    with pytest.raises(DicomError, match="multiple DICOM series"):
        DicomLoader.load(multi_series)


def test_wraps_sitk_runtime_error_as_dicom_error(tmp_path, monkeypatch):
    import SimpleITK as sitk

    # A real directory so the "not a directory" guard passes.
    d = tmp_path / "broken"
    d.mkdir()

    def boom(self, directory):
        raise RuntimeError("GDCM exploded")

    monkeypatch.setattr(sitk.ImageSeriesReader, "GetGDCMSeriesIDs", boom)
    with pytest.raises(DicomError, match="failed to scan"):
        DicomLoader.load(str(d))
