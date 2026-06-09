"""`stom-dicom2nifti` — convert a DICOM CBCT series to a NIfTI file."""

from __future__ import annotations

import argparse
import sys

from .dicom_loader import DicomError, DicomLoader
from .nifti_io import save_volume_nifti


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stom-dicom2nifti",
        description="Convert a DICOM CBCT series directory to a NIfTI (.nii.gz) file.",
    )
    parser.add_argument("dicom_dir", help="directory containing one DICOM series")
    parser.add_argument("output", help="output path, e.g. study.nii.gz")
    args = parser.parse_args(argv)

    try:
        volume = DicomLoader.load(args.dicom_dir)
    except DicomError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    save_volume_nifti(volume, args.output)
    print(f"saved volume {volume.shape} -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
