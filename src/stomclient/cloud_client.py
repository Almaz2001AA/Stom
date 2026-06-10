"""HTTP client for the stomserver API. Hides httpx behind typed methods."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx


class CloudError(Exception):
    """Any non-recoverable cloud failure."""


class AuthError(CloudError):
    """401 — missing or invalid token."""


class NotReady(CloudError):
    """409 — resource exists but is not ready yet."""


@dataclass
class StudyInfo:
    study_id: int
    shape: list[int]
    spacing: list[float]


@dataclass
class JobStatus:
    job_id: int
    status: str
    error: str | None = None


class CloudClient:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        *,
        timeout: float = 30.0,
        retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        client: httpx.Client | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._timeout = timeout
        self._retries = retries
        self._sleep = sleep
        self._client = client or httpx.Client(timeout=timeout)

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        url = f"{self._base}{path}"
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                resp = self._client.request(method, url, headers=self._headers, **kwargs)
            except httpx.HTTPError as exc:
                last = CloudError(f"network error: {exc}")
                if attempt < self._retries:
                    self._sleep(2 ** attempt)
                    continue
                raise last from exc
            if resp.status_code == 401:
                raise AuthError("invalid or missing token")
            if resp.status_code == 409:
                raise NotReady("resource not ready")
            if resp.status_code >= 500:
                last = CloudError(f"server error {resp.status_code}")
                if attempt < self._retries:
                    self._sleep(2 ** attempt)
                    continue
                raise last
            if resp.status_code >= 400:
                raise CloudError(f"request failed {resp.status_code}: {resp.text}")
            return resp
        raise last or CloudError("request failed")  # pragma: no cover

    def check_connection(self) -> bool:
        try:
            resp = self._client.get(f"{self._base}/healthz", timeout=self._timeout)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200 and resp.json().get("status") == "ok"

    def upload_study(self, nifti_bytes: bytes, filename: str) -> StudyInfo:
        resp = self._request(
            "POST", "/studies",
            files={"file": (filename, nifti_bytes, "application/gzip")},
        )
        d = resp.json()
        return StudyInfo(study_id=d["study_id"], shape=d["shape"], spacing=d["spacing"])

    def start_segmentation(self, study_id: int) -> JobStatus:
        d = self._request("POST", f"/studies/{study_id}/segment").json()
        return JobStatus(job_id=d["job_id"], status=d["status"], error=d.get("error"))

    def poll_status(self, job_id: int) -> JobStatus:
        d = self._request("GET", f"/jobs/{job_id}").json()
        return JobStatus(job_id=d["job_id"], status=d["status"], error=d.get("error"))

    def download_mask(self, study_id: int) -> tuple[bytes, bytes]:
        mask = self._request("GET", f"/studies/{study_id}/masks").content
        labels = self._request("GET", f"/studies/{study_id}/masks/labels").content
        return mask, labels
