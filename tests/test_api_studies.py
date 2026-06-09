def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_upload_requires_auth(client, nifti_bytes):
    r = client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)})
    assert r.status_code == 401
    assert r.json()["code"] == 401


def test_upload_success(client, account_token, nifti_bytes):
    _, token = account_token
    r = client.post("/studies", files={"file": ("v.nii.gz", nifti_bytes)},
                    headers=_auth(token))
    assert r.status_code == 201
    body = r.json()
    assert body["shape"] == [8, 16, 16]
    assert body["spacing"] == [0.3, 0.3, 0.3]
    assert isinstance(body["study_id"], int)


def test_upload_rejects_bad_nifti(client, account_token):
    _, token = account_token
    r = client.post("/studies", files={"file": ("bad.nii.gz", b"not a nifti")},
                    headers=_auth(token))
    assert r.status_code == 400
    assert r.json()["code"] == 400
