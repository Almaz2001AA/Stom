"""Core data layer for CBCT segmentation."""

from .dicom_loader import DicomError, DicomLoader
from .geometry import Geometry
from .mask import LabelInfo, SegmentationMask
from .nifti_io import load_volume_nifti, save_volume_nifti
from .volume import Volume

__version__ = "0.1.0"

__all__ = [
    "DicomError",
    "DicomLoader",
    "Geometry",
    "LabelInfo",
    "SegmentationMask",
    "Volume",
    "load_volume_nifti",
    "save_volume_nifti",
    "__version__",
]
