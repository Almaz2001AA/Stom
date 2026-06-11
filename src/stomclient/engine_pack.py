"""Provision the on-device engine-pack: download + verify + extract on first use.

The slim client ships without torch/nnU-Net. On first local segmentation it
fetches the engine-pack — a self-contained ``stom-engine`` build that bundles the
weights — verifies its SHA-256, and unpacks it under the user's local app data.
Later runs reuse the unpacked copy.

Functions take an injectable ``opener`` so the download path is unit-testable
without network access.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen

EXE_NAME = "stom-engine.exe" if os.name == "nt" else "stom-engine"

# Stable URL: GitHub serves the latest release's asset by name. CI publishes
# both the manifest and the versioned engine-pack zip it points at.
MANIFEST_URL = (
    "https://github.com/Almaz2001AA/Stom/releases/latest/download/"
    "engine-pack-manifest.json"
)

ProgressCb = Callable[[int, int], None]  # (bytes_done, bytes_total)


def fetch_manifest(url: str = MANIFEST_URL, *, opener=urlopen) -> dict:
    """Fetch the engine-pack manifest ``{"version", "url", "sha256"}``."""
    with opener(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def engine_dir() -> Path:
    """Per-user install location for the unpacked engine-pack."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "Stom" / "engine"


def find_engine_exe(root: Path | None = None) -> Path | None:
    """Return the installed ``stom-engine`` executable, or None if absent."""
    root = root or engine_dir()
    if not root.exists():
        return None
    for p in root.rglob(EXE_NAME):
        if p.is_file():
            return p
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, *, progress: ProgressCb | None = None, opener=urlopen) -> None:
    """Stream ``url`` to ``dest`` in chunks, reporting progress."""
    with opener(url) as resp:
        try:
            total = int(resp.headers.get("Content-Length", 0))
        except (AttributeError, TypeError, ValueError):
            total = 0
        done = 0
        with open(dest, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)


def install_pack(zip_path: Path, *, sha256: str | None = None, root: Path | None = None) -> Path:
    """Verify and extract an engine-pack zip; return the engine executable path."""
    root = root or engine_dir()
    if sha256 and _sha256(zip_path).lower() != sha256.lower():
        raise ValueError("engine-pack checksum mismatch — download corrupted")
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root)
    exe = find_engine_exe(root)
    if exe is None:
        raise ValueError("engine-pack did not contain a stom-engine executable")
    if os.name != "nt":
        exe.chmod(0o755)
    return exe


def provision(manifest: dict, *, progress: ProgressCb | None = None, opener=urlopen,
              root: Path | None = None) -> Path:
    """Download + verify + extract the engine-pack described by ``manifest``.

    ``manifest`` = ``{"url": ..., "sha256": ..., "version": ...}``.
    Returns the path to the ready-to-run engine executable.
    """
    root = root or engine_dir()
    with tempfile.TemporaryDirectory() as tmp:
        zp = Path(tmp) / "engine-pack.zip"
        download(manifest["url"], zp, progress=progress, opener=opener)
        return install_pack(zp, sha256=manifest.get("sha256"), root=root)
