import io
import json

from stomclient import updates
from stomclient.updates import (
    check_for_client_update,
    client_update_available,
    download_installer,
    fetch_latest_release,
    parse_version,
)


class _FakeResp:
    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data))}

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _opener_for(data: bytes):
    def _open(url):
        return _FakeResp(data)
    return _open


def _release_json(tag: str, with_asset: bool = True) -> bytes:
    assets = []
    if with_asset:
        assets = [{"name": "StomClientSetup.exe",
                   "browser_download_url": f"http://x/{tag}/StomClientSetup.exe"}]
    return json.dumps({"tag_name": tag, "assets": assets}).encode()


def test_parse_version_strips_v_prefix():
    assert parse_version("v0.1.4") == (0, 1, 4)
    assert parse_version("0.1.4") == (0, 1, 4)


def test_parse_version_handles_garbage():
    assert parse_version("") == (0,)
    assert parse_version("nonsense") == (0,)


def test_client_update_available_compares_semver():
    assert client_update_available("0.1.4", "v0.1.5") is True
    assert client_update_available("0.1.5", "v0.1.5") is False
    assert client_update_available("0.1.5", "v0.1.4") is False
    assert client_update_available("0.1.9", "v0.2.0") is True


def test_fetch_latest_release_extracts_tag_and_asset():
    rel = fetch_latest_release(opener=_opener_for(_release_json("v0.1.5")))
    assert rel["version"] == "v0.1.5"
    assert rel["url"].endswith("StomClientSetup.exe")


def test_fetch_latest_release_no_asset_returns_none_url():
    rel = fetch_latest_release(opener=_opener_for(_release_json("v0.1.5", with_asset=False)))
    assert rel["url"] is None


def test_check_for_update_returns_release_when_newer(monkeypatch):
    monkeypatch.setattr(updates, "current_version", lambda: "0.1.4")
    rel = check_for_client_update(opener=_opener_for(_release_json("v0.1.5")))
    assert rel is not None and rel["version"] == "v0.1.5"


def test_check_for_update_none_when_current(monkeypatch):
    monkeypatch.setattr(updates, "current_version", lambda: "0.1.5")
    assert check_for_client_update(opener=_opener_for(_release_json("v0.1.5"))) is None


def test_check_for_update_quiet_when_own_version_unknown(monkeypatch):
    # A frozen build missing its metadata reads as "0"; it must NOT nag forever.
    monkeypatch.setattr(updates, "current_version", lambda: "0")
    assert check_for_client_update(opener=_opener_for(_release_json("v0.2.0"))) is None


def test_check_for_update_swallows_network_errors():
    def _boom(url):
        raise OSError("offline")
    assert check_for_client_update(opener=_boom) is None


def test_download_installer_writes_file_and_progress(tmp_path):
    data = b"MZ" + b"\x00" * (2 * (1 << 20))
    seen = []
    dest = download_installer("http://x/s.exe", tmp_path / "s.exe",
                              progress=lambda d, t: seen.append((d, t)),
                              opener=_opener_for(data))
    assert dest.read_bytes() == data
    assert seen[-1] == (len(data), len(data))
