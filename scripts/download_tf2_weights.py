"""Download the ToothFairy2 (per-tooth, 49-class) nnU-Net weights into models/.

The weights are NOT committed to git (~781 MB) and are gated behind the
ToothFairy2 challenge sign-up, so they must be re-hosted at a direct-download
URL (GitHub release asset, Zenodo, etc.) and pointed to via ``--url`` or the
``TF2_WEIGHTS_URL`` env var — the same pattern the CI build uses.

The archive must contain a top-level ``Dataset112_ToothFairy2/`` folder laid out
as a flat nnU-Net trained-model directory::

    Dataset112_ToothFairy2/
      dataset.json
      plans.json
      fold_0/checkpoint_best.pth

Build such an archive from the validated local model with, e.g.::

    cp -r tf2_data/tf2_model /tmp/Dataset112_ToothFairy2
    (cd /tmp && zip -r tf2_weights.zip Dataset112_ToothFairy2)

License: CC BY-SA 4.0 — see the repository NOTICE for required attribution.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

_EXPECTED = Path("Dataset112_ToothFairy2") / "fold_0" / "checkpoint_best.pth"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download ToothFairy2 weights.")
    parser.add_argument(
        "--url", default=os.environ.get("TF2_WEIGHTS_URL"),
        help="direct-download URL for the TF2 weights archive "
             "(default: $TF2_WEIGHTS_URL)",
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="target directory to extract into (default: ./models)",
    )
    args = parser.parse_args(argv)

    if not args.url:
        print("error: pass --url or set TF2_WEIGHTS_URL", file=sys.stderr)
        return 2

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    zip_path = model_dir / "tf2_weights.zip"

    print(f"downloading ToothFairy2 weights to {zip_path} ...")
    try:
        urllib.request.urlretrieve(args.url, zip_path)
    except Exception as exc:  # noqa: BLE001
        print(f"error: download failed: {exc}", file=sys.stderr)
        return 1

    print("extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(model_dir)

    if not (model_dir / _EXPECTED).is_file():
        print(
            f"error: archive did not contain {model_dir / _EXPECTED} — the zip "
            "must hold a top-level Dataset112_ToothFairy2/ flat nnU-Net model",
            file=sys.stderr,
        )
        return 1
    print(f"done. ToothFairy2 model in {model_dir / 'Dataset112_ToothFairy2'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
