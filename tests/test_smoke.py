import stomcore


def test_version_is_exposed():
    assert stomcore.__version__ == "0.1.0"


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
