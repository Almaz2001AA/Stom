import pytest

from stomserver.storage.base import StorageKeyError
from stomserver.storage.local import LocalFileStorage


def test_put_get_roundtrip(tmp_path):
    s = LocalFileStorage(str(tmp_path))
    s.put("acct/1/volume.nii.gz", b"hello")
    assert s.get("acct/1/volume.nii.gz") == b"hello"


def test_exists(tmp_path):
    s = LocalFileStorage(str(tmp_path))
    assert s.exists("missing") is False
    s.put("k", b"x")
    assert s.exists("k") is True


def test_get_missing_raises(tmp_path):
    s = LocalFileStorage(str(tmp_path))
    with pytest.raises(StorageKeyError):
        s.get("nope")


def test_delete(tmp_path):
    s = LocalFileStorage(str(tmp_path))
    s.put("k", b"x")
    s.delete("k")
    assert s.exists("k") is False
    s.delete("k")  # idempotent, no error


def test_rejects_path_traversal(tmp_path):
    s = LocalFileStorage(str(tmp_path))
    with pytest.raises(ValueError):
        s.put("../escape", b"x")
