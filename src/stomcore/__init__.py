"""Core data layer for CBCT segmentation."""

from .dicom_loader import DicomError, DicomLoader
from .geometry import Geometry
from .mask import LabelInfo, SegmentationMask
from .mask_io import load_mask_nifti, save_mask_nifti
from .nifti_io import load_volume_nifti, save_volume_nifti
from .volume import Volume

# Keep in sync with [project].version in pyproject.toml at each release bump.
# Used as the fallback version source when packaged metadata is unavailable
# (see stomclient.updates.current_version).
__version__ = "0.4.2"

__all__ = [
    "DicomError",
    "DicomLoader",
    "Geometry",
    "LabelInfo",
    "SegmentationMask",
    "Volume",
    "load_mask_nifti",
    "load_volume_nifti",
    "save_mask_nifti",
    "save_volume_nifti",
    "__version__",
]
