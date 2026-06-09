from stomserver.segmentation.runner import FakeRunner
from stomserver.segmentation.worker import _run_segmentation


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _upload(client, token, nifti_bytes):
    return client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)},
                       headers=_auth(token)).json()["study_id"]


def test_masks_409_before_done(client, account_token, nifti_bytes):
    _, token = account_token
    study_id = _upload(client, token, nifti_bytes)
    client.post(f"/studies/{study_id}/segment", headers=_auth(token))
    r = client.get(f"/studies/{study_id}/masks", headers=_auth(token))
    assert r.status_code == 409


def test_masks_served_after_done(client, db_factory, storage, account_token, nifti_bytes):
    _, token = account_token
    study_id = _upload(client, token, nifti_bytes)
    job_id = client.post(f"/studies/{study_id}/segment", headers=_auth(token)).json()["job_id"]

    # Run the worker synchronously to completion.
    _run_segmentation(job_id, db_factory, storage, FakeRunner())

    r = client.get(f"/studies/{study_id}/masks", headers=_auth(token))
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/gzip"
    assert len(r.content) > 0

    r2 = client.get(f"/studies/{study_id}/masks/labels", headers=_auth(token))
    assert r2.status_code == 200
    assert "mandibular-canal" in r2.text
