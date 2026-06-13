"""Check whether a newer client (StomClientSetup.exe) has been released.

The slim client knows its own version (the packaged ``stomcore`` version) and asks
the GitHub releases API for the latest published tag. If the release is newer it
offers to download + launch the installer. All network access goes through an
injectable ``opener`` so the logic is unit-testable without hitting the network,
and every call is written to fail soft: a startup update check must never raise.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.request import urlopen

ProgressCb = Callable[[int, int], None]  # (bytes_done, bytes_total)

REPO = "Almaz2001AA/Stom"
LATEST_RELEASE_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
INSTALLER_NAME = "StomClientSetup.exe"


def current_version() -> str:
    """This client's version.

    Primary source is the installed ``stomcore`` package metadata. In a frozen
    PyInstaller build that metadata is bundled via ``copy_metadata`` (see
    ``packaging/stom-client.spec``); should it ever be missing we fall back to the
    baked-in ``stomcore.__version__`` rather than ``"0"`` — a ``"0"`` here makes
    every release look newer than us and nags the user to update on every launch.
    """
    try:
        return version("stomcore")
    except PackageNotFoundError:  # pragma: no cover - metadata bundled in practice
        try:
            from stomcore import __version__

            return __version__
        except Exception:  # noqa: BLE001 - last-ditch; never break startup
            return "0"


def parse_version(tag: str) -> tuple[int, ...]:
    """Parse ``"v0.1.4"`` / ``"0.1.4"`` into ``(0, 1, 4)`` for comparison.

    Non-numeric or pre-release suffixes are ignored component-wise; anything
    unparseable yields ``(0,)`` so it never sorts above a real version.
    """
    tag = (tag or "").strip().lstrip("vV")
    parts: list[int] = []
    for chunk in tag.split("."):
        num = ""
        for ch in chunk:
            if ch.isdigit():
                num += ch
            else:
                break
        if num == "":
            break
        parts.append(int(num))
    return tuple(parts) or (0,)


def client_update_available(current: str, latest: str) -> bool:
    """Whether ``latest`` is a strictly newer version than ``current``."""
    return parse_version(latest) > parse_version(current)


def fetch_latest_release(url: str = LATEST_RELEASE_URL, *, opener=urlopen) -> dict:
    """Return ``{"version": tag, "url": <installer asset URL or None>}``.

    Reads the GitHub releases API for the latest release and locates the
    ``StomClientSetup.exe`` asset's download URL.
    """
    with opener(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    asset_url = None
    for asset in data.get("assets", []):
        if asset.get("name") == INSTALLER_NAME:
            asset_url = asset.get("browser_download_url")
            break
    return {"version": data.get("tag_name", ""), "url": asset_url}


def check_for_client_update(*, opener=urlopen) -> dict | None:
    """Latest release info if it is newer than this client, else None.

    Network-tolerant: any failure returns None so startup never blocks or errors.
    """
    try:
        current = current_version()
        # If we cannot determine our own version, stay quiet rather than nag: a
        # bogus low version would make every release look like an update forever.
        if parse_version(current) <= (0,):
            return None
        latest = fetch_latest_release(opener=opener)
        if latest.get("url") and client_update_available(current, latest["version"]):
            return latest
    except Exception:  # noqa: BLE001 - update check must never break startup
        return None
    return None


def download_installer(url: str, dest: Path, *, progress: ProgressCb | None = None,
                       opener=urlopen) -> Path:
    """Stream the installer to ``dest`` (chunked, with progress); return ``dest``."""
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
    return dest


def launch_installer(path: Path) -> None:
    """Start the downloaded installer so it can replace the running app.

    Windows-only in practice; ``os.startfile`` returns immediately, after which
    the caller should quit so the installer (CloseApplications=yes) can update.
    """
    os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606 - Windows installer launch
