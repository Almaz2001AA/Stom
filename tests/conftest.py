import os

import numpy as np
import pytest
import SimpleITK as sitk


def _write_dicom_series(
    directory,
    n_slices,
    rows=16,
    cols=16,
    spacing=(0.3, 0.3, 0.3),
    series_uid: str = "1.2.826.0.1.3680043.2.1125.1.1234567890",
    name_prefix: str = "slice",
):
    """Write a minimal valid CT DICOM series into `directory`. Returns the dir path."""
    arr = (np.arange(n_slices * rows * cols).reshape(n_slices, rows, cols) % 1000).astype(np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    tags = {
        "0008|0060": "CT",                 # Modality
        "0020|000e": series_uid,           # Series Instance UID
        "0008|0016": "1.2.840.10008.5.1.4.1.1.2",  # SOP Class UID (CT Image Storage)
        "0028|0100": "16",                 # Bits Allocated
        "0028|0101": "16",                 # Bits Stored
        "0028|0102": "15",                 # High Bit
        "0028|0103": "1",                  # Pixel Representation (signed)
    }
    for i in range(img.GetDepth()):
        slice_i = img[:, :, i]
        for tag, value in tags.items():
            slice_i.SetMetaData(tag, value)
        position = img.TransformIndexToPhysicalPoint((0, 0, i))
        slice_i.SetMetaData("0020|0032", "\\".join(f"{c:.4f}" for c in position))  # Image Position
        slice_i.SetMetaData("0020|0013", str(i))  # Instance Number
        slice_i.SetMetaData("0008|0018", f"{series_uid}.{i}")  # SOP Instance UID
        writer.SetFileName(os.path.join(directory, f"{name_prefix}_{i:03d}.dcm"))
        writer.Execute(slice_i)
    return directory


@pytest.fixture
def dicom_series(tmp_path):
    """A valid 8-slice synthetic CT series directory."""
    d = tmp_path / "series"
    d.mkdir()
    return _write_dicom_series(str(d), n_slices=8)


@pytest.fixture
def single_slice_series(tmp_path):
    """A degenerate 1-slice series directory."""
    d = tmp_path / "single"
    d.mkdir()
    return _write_dicom_series(str(d), n_slices=1)


@pytest.fixture
def multi_series(tmp_path):
    """A directory containing two distinct DICOM series."""
    d = tmp_path / "multi"
    d.mkdir()
    _write_dicom_series(
        str(d), n_slices=4,
        series_uid="1.2.826.0.1.3680043.2.1125.1.1111111111", name_prefix="a",
    )
    _write_dicom_series(
        str(d), n_slices=4,
        series_uid="1.2.826.0.1.3680043.2.1125.1.2222222222", name_prefix="b",
    )
    return str(d)


import numpy as np
import pytest
from fastapi.testclient import TestClient

from stomcore.geometry import Geometry
from stomcore.nifti_io import save_volume_nifti
from stomcore.volume import Volume
from stomserver.api.app import create_app
from stomserver.auth import hash_token
from stomserver.db.models import Account, ApiToken
from stomserver.db.session import create_all, make_engine, make_session_factory
from stomserver.storage.local import LocalFileStorage


class FakeQueue:
    """Records enqueued job ids; optionally runs them synchronously."""

    def __init__(self):
        self.enqueued: list[int] = []
        self._sync_handler = None

    def set_sync_handler(self, handler):
        self._sync_handler = handler

    def enqueue_segmentation(self, job_id: int) -> None:
        self.enqueued.append(job_id)
        if self._sync_handler is not None:
            self._sync_handler(job_id)


@pytest.fixture
def db_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    create_all(engine)
    return make_session_factory(engine)


@pytest.fixture
def storage(tmp_path):
    return LocalFileStorage(str(tmp_path / "storage"))


@pytest.fixture
def queue():
    return FakeQueue()


@pytest.fixture
def account_token(db_factory):
    db = db_factory()
    acct = Account(name="Clinic A")
    db.add(acct)
    db.flush()
    raw_token = "test-token-A"
    db.add(ApiToken(token_hash=hash_token(raw_token), account_id=acct.id))
    db.commit()
    return acct.id, raw_token


@pytest.fixture
def client(db_factory, storage, queue):
    app = create_app(session_factory=db_factory, storage=storage, queue=queue)
    return TestClient(app)


def _nifti_bytes(tmp_path, shape=(8, 16, 16), name="vol.nii.gz"):
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    vol = Volume(np.arange(int(np.prod(shape)), dtype=np.int16).reshape(shape), geo)
    path = tmp_path / name
    save_volume_nifti(vol, path)
    return path.read_bytes()


@pytest.fixture
def nifti_bytes(tmp_path):
    return _nifti_bytes(tmp_path)
