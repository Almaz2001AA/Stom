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
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen

EXE_NAME = "stom-engine.exe" if os.name == "nt" else "stom-engine"

# Records which engine-pack version is unpacked under engine_dir(), so the client
# can tell an out-of-date (or pre-versioning, marker-less) install from a current
# one and offer an update. Written by provision() after a successful extract.
MARKER_NAME = "installed.json"

# Stable URL: GitHub serves the latest release's asset by name. CI publishes
# both the manifest and the versioned engine-pack zip it points at.
MANIFEST_URL = (
    "https://github.com/Almaz2001AA/Stom/releases/latest/download/"
    "engine-pack-manifest.json"
)

ProgressCb = Callable[[int, int], None]  # (bytes_done, bytes_total)


def fetch_manifest(url: str = MANIFEST_URL, *, opener=urlopen) -> dict:
    """Fetch the engine-pack manifest ``{"version", "url", "sha256"}``.

    May also carry an optional ``"models"`` list naming the segmentation models
    bundled in the pack (e.g. ``["dentalsegmentator", "toothfairy2"]``); older
    manifests omit it. Unknown keys are ignored, so this stays backward/forward
    compatible.
    """
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


def read_marker(root: Path | None = None) -> dict:
    """The unpacked engine-pack's install record ``{"version", "sha256"}``.

    Returns ``{}`` when no marker exists (nothing installed, or a pre-versioning
    install) or it is unreadable.
    """
    marker = (root or engine_dir()) / MARKER_NAME
    if not marker.exists():
        return {}
    try:
        data = json.loads(marker.read_text())
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}


def installed_version(root: Path | None = None) -> str | None:
    """Version of the unpacked engine-pack, or None if unknown/not installed.

    Returns None when no marker exists — either nothing is installed, or it is a
    pre-versioning install (e.g. the broken v0.1.2/v0.1.3 pack). Callers pair this
    with :func:`find_engine_exe` to tell "nothing installed" from "legacy install".
    """
    return read_marker(root).get("version")


def _write_marker(version: str | None, sha256: str | None, root: Path) -> None:
    if not version:
        return
    (root / MARKER_NAME).write_text(
        json.dumps({"version": version, "sha256": sha256})
    )


def engine_update_available(manifest: dict, root: Path | None = None) -> bool:
    """Whether the installed engine-pack is stale relative to ``manifest``.

    Freshness is judged by the pack's *content*, not just its version string:
    when both the install record and the manifest carry a SHA-256 we compare
    those, so a republished pack with the same version still registers as an
    update (and an identical pack never falsely does). We fall back to the
    version string only when a checksum isn't available on both sides.

    True when a pack is installed but its checksum/version differs from the
    manifest, or when an engine exe is present without a marker (legacy/broken
    pack — e.g. the v0.1.3 install that hangs on freeze_support). False when
    nothing is installed at all (that is the *install* flow, not an update).
    """
    root = root or engine_dir()
    if find_engine_exe(root) is None:
        return False  # nothing installed -> install flow, not an update
    marker = read_marker(root)
    if not marker:
        return True  # exe present but no marker -> legacy/broken pack, update it

    man_sha = (manifest.get("sha256") or "").lower()
    cur_sha = (marker.get("sha256") or "").lower()
    if man_sha and cur_sha:
        return man_sha != cur_sha
    return marker.get("version") != manifest.get("version")


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


def install_pack(zip_path: Path, *, sha256: str | None = None, root: Path | None = None,
                  clean: bool = False) -> Path:
    """Verify and extract an engine-pack zip; return the engine executable path.

    ``clean`` wipes ``root`` before extracting so an update does not leave stale
    files from the previous pack behind.
    """
    root = root or engine_dir()
    if sha256 and _sha256(zip_path).lower() != sha256.lower():
        raise ValueError("engine-pack checksum mismatch — download corrupted")
    if clean and root.exists():
        shutil.rmtree(root)
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
              root: Path | None = None, clean: bool = False) -> Path:
    """Download + verify + extract the engine-pack described by ``manifest``.

    ``manifest`` = ``{"url": ..., "sha256": ..., "version": ...}``.
    Records the installed version so later runs can detect an out-of-date pack.
    Pass ``clean=True`` for an update so stale files from the old pack are removed.
    Returns the path to the ready-to-run engine executable.
    """
    root = root or engine_dir()
    with tempfile.TemporaryDirectory() as tmp:
        zp = Path(tmp) / "engine-pack.zip"
        download(manifest["url"], zp, progress=progress, opener=opener)
        exe = install_pack(zp, sha256=manifest.get("sha256"), root=root, clean=clean)
    _write_marker(manifest.get("version"), manifest.get("sha256"), root)
    return exe
