"""Download DentalSegmentator nnU-Net weights from Zenodo into MODEL_DIR.

Weights are NOT committed to git. See NOTICE for attribution (CC-BY 4.0).
Record: https://zenodo.org/records/10829675
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

from stomserver.config import load_config

WEIGHTS_URL = (
    "https://zenodo.org/records/10829675/files/"
    "Dataset112_DentalSegmentator_v100.zip?download=1"
)
WEIGHTS_ZIP = "Dataset112_DentalSegmentator_v100.zip"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download DentalSegmentator weights.")
    parser.add_argument("--model-dir", default=load_config().model_dir,
                        help="target directory for weights")
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    zip_path = model_dir / WEIGHTS_ZIP

    print(f"downloading weights to {zip_path} ...")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, zip_path)
    except Exception as exc:  # noqa: BLE001
        print(f"error: download failed: {exc}", file=sys.stderr)
        return 1

    print("extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(model_dir)
    print(f"done. weights in {model_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
