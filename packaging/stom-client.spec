# PyInstaller spec for the Stom desktop client (built on Windows in CI).
# One-folder build (COLLECT) — more reliable than one-file for PySide6 + SimpleITK.
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
# Bundle the heavy native packages in full so the frozen app finds their data/DLLs.
for _pkg in ("SimpleITK", "PySide6"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

hiddenimports += ["stomcore", "stomclient", "stomengine"]

# Keep the GUI installer slim: the on-device inference stack (torch + nnU-Net and
# its transitive deps) is NOT bundled here. stomengine imports them lazily inside
# predict(); the slim client either runs cloud mode or, in local mode, shells out
# to the separately-downloaded engine-pack (Phase B). Excluding them drops ~1.5 GB.
_ENGINE_DEPS = [
    "torch", "torchvision", "nnunetv2", "acvl_utils",
    "dynamic_network_architectures", "batchgenerators", "batchgeneratorsv2",
    "sklearn", "scipy", "pandas", "matplotlib",
]

a = Analysis(
    ["launch.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["stomserver", "fastapi", "uvicorn", "rq", "redis", "sqlalchemy"]
    + _ENGINE_DEPS,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="stom-client",
    console=False,          # GUI app: no console window
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="stom-client",     # -> dist/stom-client/
)
