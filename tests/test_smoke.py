from importlib.metadata import version

import stomcore


def test_version_is_exposed():
    # The baked-in fallback must match the installed package metadata so the
    # client never misreports its version (and nags about phantom updates).
    assert stomcore.__version__ == version("stomcore")


def test_public_types_are_reexported():
    from stomcore import (
        DicomError,
        DicomLoader,
        Geometry,
        LabelInfo,
        SegmentationMask,
        Volume,
        load_volume_nifti,
        save_volume_nifti,
    )

    assert all(
        obj is not None
        for obj in (
            DicomError,
            DicomLoader,
            Geometry,
            LabelInfo,
            SegmentationMask,
            Volume,
            load_volume_nifti,
            save_volume_nifti,
        )
    )
