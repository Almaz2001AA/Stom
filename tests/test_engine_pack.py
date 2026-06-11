import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from stomclient import engine_pack
from stomclient.engine_pack import (
    EXE_NAME,
    download,
    find_engine_exe,
    install_pack,
    provision,
)


def _make_pack(extra_name: str = EXE_NAME, nested: bool = False) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        arc = f"stom-engine/{extra_name}" if nested else extra_name
        z.writestr(arc, b"#!/bin/sh\necho engine\n")
        z.writestr("README.txt", b"engine-pack")
    return buf.getvalue()


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


def test_find_returns_none_when_absent(tmp_path):
    assert find_engine_exe(tmp_path / "nope") is None


def test_install_pack_extracts_and_locates_exe(tmp_path):
    zp = tmp_path / "p.zip"
    zp.write_bytes(_make_pack(nested=True))
    exe = install_pack(zp, root=tmp_path / "root")
    assert exe.name == EXE_NAME
    assert exe.is_file()
    assert find_engine_exe(tmp_path / "root") == exe


def test_install_pack_checksum_mismatch_raises(tmp_path):
    zp = tmp_path / "p.zip"
    zp.write_bytes(_make_pack())
    with pytest.raises(ValueError, match="checksum mismatch"):
        install_pack(zp, sha256="deadbeef", root=tmp_path / "root")


def test_install_pack_checksum_match_ok(tmp_path):
    data = _make_pack()
    zp = tmp_path / "p.zip"
    zp.write_bytes(data)
    good = hashlib.sha256(data).hexdigest()
    exe = install_pack(zp, sha256=good, root=tmp_path / "root")
    assert exe.is_file()


def test_install_pack_without_exe_raises(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("README.txt", b"no engine here")
    zp = tmp_path / "p.zip"
    zp.write_bytes(buf.getvalue())
    with pytest.raises(ValueError, match="did not contain"):
        install_pack(zp, root=tmp_path / "root")


def test_download_writes_and_reports_progress(tmp_path):
    data = b"x" * (3 * (1 << 20) + 17)  # >3 chunks
    seen = []
    dest = tmp_path / "out.bin"
    download("http://x/pack", dest, progress=lambda d, t: seen.append((d, t)), opener=_opener_for(data))
    assert dest.read_bytes() == data
    assert seen[-1] == (len(data), len(data))   # final progress is complete


def test_provision_end_to_end(tmp_path):
    data = _make_pack(nested=True)
    manifest = {"url": "http://x/engine.zip", "sha256": hashlib.sha256(data).hexdigest()}
    exe = provision(manifest, opener=_opener_for(data), root=tmp_path / "root")
    assert exe.is_file()
    assert exe.name == EXE_NAME


def test_fetch_manifest_parses_json():
    payload = b'{"version": "0.1.0", "url": "http://x/e.zip", "sha256": "abc"}'
    manifest = engine_pack.fetch_manifest("http://x/m.json", opener=_opener_for(payload))
    assert manifest["url"] == "http://x/e.zip"
    assert manifest["sha256"] == "abc"


def test_engine_dir_is_under_user_home(monkeypatch):
    # Sanity: resolves to a Stom/engine path without raising.
    d = engine_pack.engine_dir()
    assert d.name == "engine" and d.parent.name == "Stom"
