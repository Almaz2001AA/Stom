import numpy as np

from stomcore.mask_io import load_mask_nifti
from stomserver.db.models import Account, Job, Study
from stomserver.segmentation.runner import FakeRunner
from stomserver.segmentation.worker import _run_segmentation


def _seed_study_job(db_factory, storage, tmp_path):
    from stomcore.geometry import Geometry
    from stomcore.nifti_io import save_volume_nifti
    from stomcore.volume import Volume

    db = db_factory()
    acct = Account(name="A")
    db.add(acct)
    db.flush()
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    vol = Volume(np.zeros((6, 5, 4), dtype=np.int16), geo)
    vpath = tmp_path / "v.nii.gz"
    save_volume_nifti(vol, vpath)
    key = f"{acct.id}/studies/1/volume.nii.gz"
    storage.put(key, vpath.read_bytes())
    study = Study(account_id=acct.id, original_filename="v.nii.gz", storage_key=key,
                  shape="[6, 5, 4]", spacing="[0.3, 0.3, 0.3]")
    db.add(study)
    db.flush()
    job = Job(study_id=study.id, account_id=acct.id, model_name="dentalsegmentator")
    db.add(job)
    db.commit()
    return job.id, acct.id


def test_worker_success(db_factory, storage, tmp_path):
    job_id, acct_id = _seed_study_job(db_factory, storage, tmp_path)
    _run_segmentation(job_id, db_factory, storage, FakeRunner())

    db = db_factory()
    job = db.get(Job, job_id)
    assert job.status == "done"
    assert job.error is None
    assert job.mask_storage_key == f"{acct_id}/studies/1/mask.nii.gz"
    assert storage.exists(job.mask_storage_key)
    assert storage.exists(f"{acct_id}/studies/1/mask_labels.json")


def test_worker_marks_failed_on_runner_error(db_factory, storage, tmp_path):
    job_id, _ = _seed_study_job(db_factory, storage, tmp_path)

    class BoomRunner:
        def predict(self, volume):
            raise RuntimeError("inference exploded")

    _run_segmentation(job_id, db_factory, storage, BoomRunner())

    db = db_factory()
    job = db.get(Job, job_id)
    assert job.status == "failed"
    assert "inference exploded" in job.error


def test_fake_runner_returns_matching_shape():
    from stomcore.geometry import Geometry
    from stomcore.volume import Volume

    vol = Volume(np.zeros((6, 5, 4), dtype=np.int16), Geometry.identity(spacing=(1, 1, 1)))
    labels = FakeRunner().predict(vol)
    assert labels.shape == (6, 5, 4)
    assert set(np.unique(labels)).issubset({0, 1, 2, 3, 4, 5})
