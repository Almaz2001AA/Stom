import httpx
import pytest
import respx

from stomclient.cloud_client import AuthError, CloudClient, CloudError, JobStatus, NotReady, StudyInfo

BASE = "https://api.test"


def _client():
    return CloudClient(BASE, token="tok", retries=0, sleep=lambda *_: None)


@respx.mock
def test_check_connection_true():
    respx.get(f"{BASE}/healthz").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    assert _client().check_connection() is True


@respx.mock
def test_upload_study_parses_response():
    route = respx.post(f"{BASE}/studies").mock(
        return_value=httpx.Response(
            201, json={"study_id": 7, "shape": [8, 16, 16], "spacing": [0.3, 0.3, 0.3]}
        )
    )
    info = _client().upload_study(b"nifti-bytes", "study.nii.gz")
    assert info == StudyInfo(study_id=7, shape=[8, 16, 16], spacing=[0.3, 0.3, 0.3])
    assert route.calls.last.request.headers["authorization"] == "Bearer tok"


@respx.mock
def test_start_segmentation_parses_job():
    respx.post(f"{BASE}/studies/7/segment").mock(
        return_value=httpx.Response(202, json={"job_id": 3, "status": "queued", "error": None})
    )
    js = _client().start_segmentation(7)
    assert js == JobStatus(job_id=3, status="queued", error=None)


@respx.mock
def test_poll_status_parses_job():
    respx.get(f"{BASE}/jobs/3").mock(
        return_value=httpx.Response(200, json={"job_id": 3, "status": "done", "error": None})
    )
    assert _client().poll_status(3).status == "done"


@respx.mock
def test_download_mask_returns_both_blobs():
    respx.get(f"{BASE}/studies/7/masks").mock(
        return_value=httpx.Response(200, content=b"MASK")
    )
    respx.get(f"{BASE}/studies/7/masks/labels").mock(
        return_value=httpx.Response(200, content=b"{}")
    )
    mask, labels = _client().download_mask(7)
    assert mask == b"MASK"
    assert labels == b"{}"


@respx.mock
def test_401_raises_auth_error():
    respx.get(f"{BASE}/jobs/1").mock(return_value=httpx.Response(401, json={"detail": "x"}))
    with pytest.raises(AuthError):
        _client().poll_status(1)


@respx.mock
def test_409_raises_not_ready():
    respx.get(f"{BASE}/studies/1/masks").mock(return_value=httpx.Response(409))
    with pytest.raises(NotReady):
        _client().download_mask(1)


@respx.mock
def test_500_retries_then_succeeds():
    route = respx.get(f"{BASE}/jobs/2")
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(200, json={"job_id": 2, "status": "done", "error": None}),
    ]
    client = CloudClient(BASE, token="t", retries=1, sleep=lambda *_: None)
    assert client.poll_status(2).status == "done"
    assert route.call_count == 2


@respx.mock
def test_network_error_after_retries_raises_cloud_error():
    respx.get(f"{BASE}/jobs/9").mock(side_effect=httpx.ConnectError("boom"))
    client = CloudClient(BASE, token="t", retries=1, sleep=lambda *_: None)
    with pytest.raises(CloudError):
        client.poll_status(9)
