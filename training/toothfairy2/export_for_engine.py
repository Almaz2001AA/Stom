"""Assemble a trained ToothFairy2 model into the layout the Stom engine expects.

After nnU-Net training (see README.md), the result lives under nnUNet_results as
``<trainer>__<plans>__<config>/fold_<X>/checkpoint_*.pth``. The Stom engine
(:class:`stomengine.runner.ToothFairy2Runner`) loads a *flat* model folder:

    Dataset112_ToothFairy2/
      dataset.json
      plans.json
      fold_0/checkpoint_best.pth   # use_folds=(0,), checkpoint_name="checkpoint_best.pth"

This script copies the trained ``dataset.json`` + ``plans.json`` and the chosen
checkpoint into that flat layout (renaming the fold to ``fold_0`` and the
checkpoint to ``checkpoint_best.pth``), then optionally verifies it loads with
the same nnU-Net version the engine ships. The engine reads the configuration
name straight from the checkpoint's ``init_args``, so no folder-name encoding is
needed.

Usage::

    python export_for_engine.py \
        --results-dir nnUNet_results/Dataset112_ToothFairy2/nnUNetTrainer_onlyMirror01_1500ep__nnUNetResEncUNetLPlans_torchres__3d_fullres \
        --fold all \
        --checkpoint checkpoint_final.pth \
        --out ./engine_model

Then zip ``<out>/Dataset112_ToothFairy2`` (top-level folder = Dataset112_ToothFairy2),
upload it, set the repo variable TF2_WEIGHTS_URL to its URL, and dispatch the
build-engine-pack workflow (version input e.g. v0.5.0) to ship it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MODEL_NAME = "Dataset112_ToothFairy2"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export a trained TF2 model for the Stom engine.")
    p.add_argument("--results-dir", required=True,
                   help="nnU-Net results trainer dir (…/<trainer>__<plans>__<config>)")
    p.add_argument("--fold", default="all",
                   help="trained fold to export (default: all)")
    p.add_argument("--checkpoint", default="checkpoint_final.pth",
                   help="checkpoint file to ship (default: checkpoint_final.pth; "
                        "the 'all' fold has no validation, so 'final' is the model)")
    p.add_argument("--out", default="./engine_model",
                   help="output dir; the flat model lands in <out>/Dataset112_ToothFairy2")
    p.add_argument("--verify", action="store_true",
                   help="load the assembled model with nnU-Net to confirm it is valid")
    args = p.parse_args(argv)

    src = Path(args.results_dir)
    fold_dir = src / f"fold_{args.fold}"
    ckpt = fold_dir / args.checkpoint
    for required in (src / "dataset.json", src / "plans.json", ckpt):
        if not required.is_file():
            print(f"error: missing {required}", file=sys.stderr)
            return 1

    dst = Path(args.out) / MODEL_NAME
    (dst / "fold_0").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "dataset.json", dst / "dataset.json")
    shutil.copy2(src / "plans.json", dst / "plans.json")
    shutil.copy2(ckpt, dst / "fold_0" / "checkpoint_best.pth")
    print(f"assembled engine model at {dst}")
    print("  dataset.json, plans.json, fold_0/checkpoint_best.pth")

    if args.verify:
        try:
            import torch
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        except Exception as exc:  # noqa: BLE001
            print(f"verify skipped (nnunetv2/torch not importable here): {exc}")
            return 0
        pred = nnUNetPredictor(device=torch.device("cpu"), allow_tqdm=False)
        pred.initialize_from_trained_model_folder(
            str(dst), use_folds=(0,), checkpoint_name="checkpoint_best.pth"
        )
        n = pred.label_manager.num_segmentation_heads
        print(f"verify OK: model loads, {n} segmentation heads")

    print()
    print("next steps:")
    print(f"  1. zip the model:  (cd {Path(args.out)} && zip -r {MODEL_NAME}.zip {MODEL_NAME})")
    print("     -> the zip MUST contain a top-level Dataset112_ToothFairy2/ folder")
    print("  2. upload the zip to a public URL (GitHub release asset / Zenodo)")
    print("  3. set the repo variable TF2_WEIGHTS_URL to that URL")
    print("  4. Actions -> build-engine-pack -> Run workflow (version e.g. v0.5.0)")
    print("     -> rebuilds the engine-pack with the new model, updates the `engine` release")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
