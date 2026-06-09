"""Load a single DICOM CBCT series into a Volume."""

from __future__ import annotations

import os

import SimpleITK as sitk

from .sitk_interop import volume_from_sitk
from .volume import Volume

MIN_SLICES = 2


class DicomError(Exception):
    """Raised when a DICOM directory cannot be loaded as a single CBCT series."""


class DicomLoader:
    @staticmethod
    def load(directory: str | os.PathLike) -> Volume:
        directory = str(directory)
        if not os.path.isdir(directory):
            raise DicomError(f"not a directory: {directory}")

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(directory)
        if not series_ids:
            raise DicomError(f"no DICOM series found in {directory}")
        if len(series_ids) > 1:
            raise DicomError(
                f"multiple DICOM series found ({len(series_ids)}); expected exactly one"
            )

        file_names = reader.GetGDCMSeriesFileNames(directory, series_ids[0])
        if len(file_names) < MIN_SLICES:
            raise DicomError(
                f"too few slices ({len(file_names)}); expected a 3D volume"
            )

        reader.SetFileNames(file_names)
        image = reader.Execute()
        return volume_from_sitk(image)
