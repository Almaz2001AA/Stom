def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _upload(client, token, nifti_bytes):
    r = client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)},
                    headers=_auth(token))
    return r.json()["study_id"]


def test_segment_enqueues_job(client, account_token, queue, nifti_bytes):
    _, token = account_token
    study_id = _upload(client, token, nifti_bytes)
    r = client.post(f"/studies/{study_id}/segment", headers=_auth(token))
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "queued"
    assert queue.enqueued == [job_id]


def test_segment_enqueue_failure_leaves_no_orphan_job(
    db_factory, storage, account_token, nifti_bytes
):
    """If enqueue fails (Redis down), the request 503s and no queued Job lingers."""
    from fastapi.testclient import TestClient

    from stomserver.api.app import create_app
    from stomserver.db.models import Job

    class BoomQueue:
        def enqueue_segmentation(self, job_id):
            raise RuntimeError("redis down")

    app = create_app(session_factory=db_factory, storage=storage, queue=BoomQueue())
    boom_client = TestClient(app, raise_server_exceptions=False)
    _, token = account_token
    study_id = _upload(boom_client, token, nifti_bytes)

    r = boom_client.post(f"/studies/{study_id}/segment", headers=_auth(token))
    assert r.status_code == 503

    db = db_factory()
    assert db.query(Job).count() == 0


def test_segment_unknown_study_404(client, account_token):
    _, token = account_token
    r = client.post("/studies/999/segment", headers=_auth(token))
    assert r.status_code == 404


def test_job_status(client, account_token, nifti_bytes):
    _, token = account_token
    study_id = _upload(client, token, nifti_bytes)
    job_id = client.post(f"/studies/{study_id}/segment", headers=_auth(token)).json()["job_id"]
    r = client.get(f"/jobs/{job_id}", headers=_auth(token))
    assert r.status_code == 200
    assert r.json() == {"job_id": job_id, "status": "queued", "error": None}


def test_job_isolation_across_accounts(client, db_factory, account_token, nifti_bytes):
    from stomserver.auth import hash_token
    from stomserver.db.models import Account, ApiToken

    _, token_a = account_token
    study_id = _upload(client, token_a, nifti_bytes)
    job_id = client.post(f"/studies/{study_id}/segment", headers=_auth(token_a)).json()["job_id"]

    # Second account
    db = db_factory()
    acct_b = Account(name="Clinic B")
    db.add(acct_b)
    db.flush()
    db.add(ApiToken(token_hash=hash_token("token-B"), account_id=acct_b.id))
    db.commit()

    r = client.get(f"/jobs/{job_id}", headers=_auth("token-B"))
    assert r.status_code == 404
