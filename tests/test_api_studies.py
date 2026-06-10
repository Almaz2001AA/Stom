def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_upload_requires_auth(client, nifti_bytes):
    r = client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)})
    assert r.status_code == 401
    assert r.json()["code"] == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_invalid_token_advertises_bearer(client, nifti_bytes):
    r = client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)},
                    headers=_auth("not-a-real-token"))
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"] == "Bearer"


def test_upload_success(client, account_token, nifti_bytes):
    _, token = account_token
    r = client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)},
                    headers=_auth(token))
    assert r.status_code == 201
    body = r.json()
    assert body["shape"] == [8, 16, 16]
    assert body["spacing"] == [0.3, 0.3, 0.3]
    assert isinstance(body["study_id"], int)


def test_missing_file_returns_validation_summary(client, account_token):
    _, token = account_token
    r = client.post("/studies", headers=_auth(token))
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == 422
    assert body["detail"].startswith("validation error")
    assert "file" in body["detail"]


def test_upload_rejects_bad_nifti(client, account_token):
    _, token = account_token
    r = client.post("/studies", files={"file": ("bad.nii.gz", b"not a nifti")},
                    headers=_auth(token))
    assert r.status_code == 400
    assert r.json()["code"] == 400


from stomserver.config import Config


def test_upload_rejects_too_large(client, account_token, nifti_bytes):
    _, token = account_token
    # Shrink the limit on the running app to below the payload size.
    base = client.app.state.config
    client.app.state.config = Config(
        db_url=base.db_url, storage_dir=base.storage_dir, redis_url=base.redis_url,
        model_dir=base.model_dir, max_upload_bytes=10,
        job_timeout_seconds=base.job_timeout_seconds,
    )
    try:
        r = client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)},
                        headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 413
        assert r.json()["code"] == 413
    finally:
        client.app.state.config = base
