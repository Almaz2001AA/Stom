# PyInstaller spec for the engine-pack: a self-contained `stom-engine` CLI that
# bundles torch (CPU), nnU-Net, and the DentalSegmentator weights. Built on a
# Windows runner in CI and zipped into the engine-pack the slim client downloads
# on first local segmentation. One-folder build (COLLECT) for reliability.
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# Bundle the heavy native packages in full so the frozen CLI finds DLLs/data.
for _pkg in ("SimpleITK", "torch", "nnunetv2", "acvl_utils",
             "dynamic_network_architectures", "batchgenerators", "scipy", "skimage"):
    try:
        _d, _b, _h = collect_all(_pkg)
        datas += _d
        binaries += _b
        hiddenimports += _h
    except Exception:
        pass

# nnU-Net resolves trainers/architectures by dynamic import; pull the trees in.
for _pkg in ("nnunetv2", "dynamic_network_architectures"):
    try:
        hiddenimports += collect_submodules(_pkg)
    except Exception:
        pass

hiddenimports += ["stomcore", "stomengine"]

# Bundle the model weights under models/... so resolve_model_dir() finds them at
# sys._MEIPASS/models/Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres
_MODEL_ROOT = (Path(SPECPATH).parent / "models" / "Dataset112_DentalSegmentator_v100").resolve()
for _f in _MODEL_ROOT.rglob("*"):
    if _f.is_file() and "__MACOSX" not in str(_f) and not _f.name.startswith("._"):
        _rel = _f.relative_to(_MODEL_ROOT).parent
        _dest = Path("models") / "Dataset112_DentalSegmentator_v100" / _rel
        datas.append((str(_f), str(_dest)))

# Ship the attribution NOTICE alongside the weights (required by the model
# licenses: CC BY 4.0 for DentalSegmentator, CC BY-SA 4.0 for ToothFairy2).
_NOTICE = (Path(SPECPATH).parent / "NOTICE").resolve()
if _NOTICE.is_file():
    datas.append((str(_NOTICE), "."))

a = Analysis(
    ["engine_launch.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["PySide6", "stomserver", "fastapi", "uvicorn", "rq", "redis", "sqlalchemy"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="stom-engine",
    console=True,           # CLI: keep the console
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="stom-engine",     # -> dist/stom-engine/
)
