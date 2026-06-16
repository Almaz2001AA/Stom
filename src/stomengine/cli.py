"""CLI for the engine-pack: run DentalSegmentator on a NIfTI volume.

Usage::

    stom-engine predict <input.nii.gz> <output_dir>

Writes ``<output_dir>/mask.nii.gz`` and ``<output_dir>/mask_labels.json``.
The slim desktop client shells out to this entry point (see
:class:`stomengine.engine.SubprocessEngine`) so it need not bundle torch/nnU-Net
itself. The engine-pack bundles the weights; the model directory is resolved
from ``STOM_MODEL_DIR`` or a ``models/...`` path next to the executable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from stomcore.mask_io import save_mask_nifti
from stomcore.nifti_io import load_volume_nifti

_MODEL_SUBPATH = Path("models") / "Dataset112_DentalSegmentator_v100" / (
    "nnUNetTrainer__nnUNetPlans__3d_fullres"
)
# Bundled location of the optional 49-class ToothFairy2 model (per-tooth FDI).
# Laid out flat (dataset.json + plans.json + fold_0/ directly inside), which is a
# valid nnU-Net trained-model folder — matches how the weights archive is packed.
_TF2_MODEL_SUBPATH = Path("models") / "Dataset112_ToothFairy2"


def selected_model(env=None) -> str:
    """Which model to run: ``toothfairy2`` (49-class FDI) or the default."""
    env = os.environ if env is None else env
    value = env.get("STOM_MODEL", "").strip().lower()
    return "toothfairy2" if value in {"toothfairy2", "tf2"} else "dentalsegmentator"


def resolve_model_dir(model: str | None = None) -> str:
    """Locate the model weights: env override, else bundled next to the exe."""
    env = os.environ.get("STOM_MODEL_DIR")
    if env:
        return env
    model = selected_model() if model is None else model
    subpath = _TF2_MODEL_SUBPATH if model == "toothfairy2" else _MODEL_SUBPATH
    # PyInstaller unpacks bundled data under sys._MEIPASS; in source layout fall
    # back to the repo root (two levels up from this file: src/stomengine/..).
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return str(base / subpath)


def _build_engine():
    from .engine import InProcessEngine

    if os.environ.get("STOM_ENGINE_FAKE") == "1":
        # Test/smoke seam: exercise the subprocess wiring without the real model.
        from .runner import FakeRunner

        return InProcessEngine(FakeRunner())

    model = selected_model()
    if model == "toothfairy2":
        from .runner import ToothFairy2Runner
        from .tf2_labels import TOOTHFAIRY2_LABELS

        return InProcessEngine(
            ToothFairy2Runner(resolve_model_dir(model)), labels=TOOTHFAIRY2_LABELS
        )

    from .runner import DentalSegmentatorRunner

    return InProcessEngine(DentalSegmentatorRunner(resolve_model_dir(model)))


def _emit_progress(done: int, total: int) -> None:
    """Print one machine-readable progress line for the parent to parse.

    ``SubprocessEngine`` streams our stdout and turns ``PROGRESS <done> <total>``
    into a live percentage in the GUI. Flush so it arrives tile-by-tile, not in
    one buffered dump at exit.
    """
    print(f"PROGRESS {done} {total}", flush=True)


def run_predict(
    input_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    engine,
    *,
    progress=None,
) -> None:
    """Testable core: load volume, segment, write mask + labels to output_dir."""
    volume = load_volume_nifti(input_path)
    mask = engine.segment(volume) if progress is None else engine.segment(
        volume, progress=progress
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_mask_nifti(mask, out / "mask.nii.gz", out / "mask_labels.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stom-engine")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="segment a NIfTI volume")
    p.add_argument("input", help="input volume (.nii.gz)")
    p.add_argument("output_dir", help="directory for mask.nii.gz + mask_labels.json")
    args = parser.parse_args(argv)

    if args.command == "predict":
        try:
            run_predict(
                args.input, args.output_dir, _build_engine(), progress=_emit_progress
            )
        except Exception as exc:  # noqa: BLE001 - report to caller via stderr/exit code
            print(f"stom-engine: {exc}", file=sys.stderr)
            return 1
        return 0
    return 2  # pragma: no cover - argparse enforces a valid command


if __name__ == "__main__":
    raise SystemExit(main())
