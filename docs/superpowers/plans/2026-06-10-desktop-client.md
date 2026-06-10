# Desktop Client (`stomclient`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PySide6 desktop client that imports a DICOM CBCT study, runs it through the `stomserver` cloud segmentation pipeline, overlays the returned masks on 2D slices, and supports linear measurements in millimetres.

**Architecture:** Thin-view / testable-core. All logic (config, HTTP client, slice rendering math, measurement math, session state machine) lives in pure Python modules under `src/stomclient/` with **no Qt imports**, fully tested headless with pytest. The Qt/PySide6 widgets under `src/stomclient/ui/` are thin adapters. 2D slices are rendered `numpy → QImage` + `QPainter` (no VTK in MVP). The client reuses `stomcore` (DicomLoader, nifti_io, mask_io, Volume/Geometry/SegmentationMask) and talks to the existing `stomserver` HTTP API.

**Tech Stack:** Python 3.11+, PySide6, httpx, numpy, `stomcore`; tests with pytest, respx (mocks httpx), pytest-qt (optional, offscreen).

---

## File Structure

**Core (no Qt):**
- `src/stomclient/__init__.py` — package marker + version.
- `src/stomclient/config.py` — `ClientConfig` dataclass + `load()`/`save()` TOML at `~/.config/stom/client.toml`.
- `src/stomclient/cloud_client.py` — `CloudClient`, `StudyInfo`, `JobStatus`, exceptions `CloudError`/`AuthError`/`NotReady`.
- `src/stomclient/slice_renderer.py` — plane constants, `slice_array`, `slice_count`, `apply_window_level`, `composite_overlay`, `default_window_level`.
- `src/stomclient/measurement.py` — `plane_spacing`, `LinearMeasurement`, `MeasurementSet`.
- `src/stomclient/serialization.py` — `volume_to_nifti_bytes`, `mask_to_bytes`, `mask_from_bytes` (bridge to `stomcore` I/O via temp files).
- `src/stomclient/coords.py` — `widget_to_image` (pure widget→image pixel mapping for measurement drawing).
- `src/stomclient/app_controller.py` — `State` enum, `AppController` session state machine (incl. label-visibility + measurement interactions).

**View (Qt, thin):**
- `src/stomclient/ui/__init__.py`
- `src/stomclient/ui/qt_image.py` — `ndarray_to_qimage` helper.
- `src/stomclient/ui/slice_widget.py` — `SliceWidget` QWidget.
- `src/stomclient/ui/settings_dialog.py` — `SettingsDialog`.
- `src/stomclient/ui/main_window.py` — `MainWindow` + cloud worker thread + mask list, W/L drag, measure mode, PNG/mask export.
- `src/stomclient/__main__.py` — `main()` entrypoint.

**Tests:**
- `tests/test_client_config.py`, `tests/test_cloud_client.py`, `tests/test_slice_renderer.py`, `tests/test_measurement.py`, `tests/test_serialization.py`, `tests/test_coords.py`, `tests/test_app_controller.py`, `tests/test_client_ui_smoke.py`.

**Docs:** `src/stomclient/README.md`.

---

## Task 1: Scaffolding and dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `src/stomclient/__init__.py`
- Test: `tests/test_client_smoke.py`

- [ ] **Step 1: Add the `client` extra, dev deps, and entrypoint to `pyproject.toml`**

In `[project.optional-dependencies]`, replace the `dev` line and add a `client` extra:

```toml
dev = ["pytest>=8.0", "respx>=0.21", "pytest-qt>=4.4", "tomli-w>=1.0"]
server = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "sqlalchemy>=2.0",
    "rq>=1.16",
    "redis>=5.0",
    "fakeredis>=2.21",
    "httpx>=0.27",
    "python-multipart>=0.0.9",
]
nnunet = ["nnunetv2>=2.5"]
client = ["PySide6>=6.6", "httpx>=0.27", "tomli-w>=1.0"]
```

In `[project.scripts]`, add the client entrypoint below the existing one:

```toml
[project.scripts]
stom-dicom2nifti = "stomcore.cli:main"
stom-client = "stomclient.__main__:main"
```

- [ ] **Step 2: Create the package marker**

Create `src/stomclient/__init__.py`:

```python
"""Desktop client for Stom CBCT segmentation."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Write the failing import test**

Create `tests/test_client_smoke.py`:

```python
def test_package_imports():
    import stomclient

    assert stomclient.__version__ == "0.1.0"
```

- [ ] **Step 4: Install the new deps and run the test**

Run: `.venv/bin/pip install -e ".[dev,client]"`
Then: `.venv/bin/python -m pytest tests/test_client_smoke.py -q`
Expected: PASS (1 passed). If PySide6 download is slow, that is expected on first install.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/stomclient/__init__.py tests/test_client_smoke.py
git commit -m "feat(client): scaffold stomclient package and deps

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `config.py` — load/save client config

**Files:**
- Create: `src/stomclient/config.py`
- Test: `tests/test_client_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_client_config.py`:

```python
import stat

from stomclient.config import ClientConfig, load, save


def test_load_missing_returns_defaults(tmp_path):
    cfg = load(tmp_path / "nope.toml")
    assert cfg == ClientConfig(server_url="", token=None, save_token=True)


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "client.toml"
    save(ClientConfig(server_url="https://api.example", token="abc"), path)
    cfg = load(path)
    assert cfg.server_url == "https://api.example"
    assert cfg.token == "abc"


def test_token_not_persisted_when_save_token_false(tmp_path):
    path = tmp_path / "client.toml"
    save(ClientConfig(server_url="u", token="secret", save_token=False), path)
    assert "secret" not in path.read_text()
    assert load(path).token is None


def test_saved_file_is_owner_only(tmp_path):
    path = tmp_path / "client.toml"
    save(ClientConfig(server_url="u", token="t"), path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_client_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomclient.config'`

- [ ] **Step 3: Implement `config.py`**

Create `src/stomclient/config.py`:

```python
"""Client configuration persisted to ~/.config/stom/client.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomli_w

DEFAULT_PATH = Path.home() / ".config" / "stom" / "client.toml"


@dataclass
class ClientConfig:
    server_url: str = ""
    token: str | None = None
    save_token: bool = True


def load(path: str | os.PathLike | None = None) -> ClientConfig:
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.exists():
        return ClientConfig()
    data = tomllib.loads(p.read_text())
    return ClientConfig(
        server_url=data.get("server_url", ""),
        token=data.get("token"),
        save_token=data.get("save_token", True),
    )


def save(config: ClientConfig, path: str | os.PathLike | None = None) -> None:
    p = Path(path) if path is not None else DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "server_url": config.server_url,
        "save_token": config.save_token,
    }
    if config.save_token and config.token:
        payload["token"] = config.token
    p.write_text(tomli_w.dumps(payload))
    os.chmod(p, 0o600)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_client_config.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/stomclient/config.py tests/test_client_config.py
git commit -m "feat(client): config load/save with 0600 perms

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `cloud_client.py` — happy-path API calls

**Files:**
- Create: `src/stomclient/cloud_client.py`
- Test: `tests/test_cloud_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cloud_client.py`:

```python
import httpx
import respx

from stomclient.cloud_client import CloudClient, JobStatus, StudyInfo

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cloud_client.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomclient.cloud_client'`

- [ ] **Step 3: Implement `cloud_client.py`**

Create `src/stomclient/cloud_client.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cloud_client.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/stomclient/cloud_client.py tests/test_cloud_client.py
git commit -m "feat(client): CloudClient happy-path API calls

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `cloud_client.py` — errors and retries

**Files:**
- Modify: `tests/test_cloud_client.py`

(The implementation from Task 3 already handles these; this task locks the behaviour with tests.)

- [ ] **Step 1: Append failing tests to `tests/test_cloud_client.py`**

```python
import pytest

from stomclient.cloud_client import AuthError, CloudError, NotReady


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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cloud_client.py -q`
Expected: PASS (9 passed). These pass because Task 3's `_request` already implements 401/409/5xx/retry handling.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cloud_client.py
git commit -m "test(client): lock CloudClient error and retry behaviour

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `slice_renderer.py` — slicing, window/level, overlay

**Files:**
- Create: `src/stomclient/slice_renderer.py`
- Test: `tests/test_slice_renderer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_slice_renderer.py`:

```python
import numpy as np

from stomcore.geometry import Geometry
from stomcore.mask import LabelInfo, SegmentationMask
from stomcore.volume import Volume
from stomclient import slice_renderer as sr


def _volume():
    voxels = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)  # [z, y, x]
    return Volume(voxels, Geometry.identity(spacing=(1, 1, 1)))


def test_slice_count_per_plane():
    v = _volume()
    assert sr.slice_count(v, sr.AXIAL) == 2
    assert sr.slice_count(v, sr.CORONAL) == 3
    assert sr.slice_count(v, sr.SAGITTAL) == 4


def test_slice_array_axial_shape_is_y_by_x():
    v = _volume()
    assert sr.slice_array(v.voxels, sr.AXIAL, 0).shape == (3, 4)
    assert sr.slice_array(v.voxels, sr.CORONAL, 0).shape == (2, 4)
    assert sr.slice_array(v.voxels, sr.SAGITTAL, 0).shape == (2, 3)


def test_apply_window_level_maps_to_uint8_range():
    s = np.array([[0, 50, 100]], dtype=np.int16)
    out = sr.apply_window_level(s, center=50, width=100)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0
    assert out[0, 2] == 255
    assert 120 <= out[0, 1] <= 135  # midpoint


def test_composite_overlay_colors_visible_label_only():
    gray = np.zeros((1, 2), dtype=np.uint8)
    mask_slice = np.array([[1, 2]], dtype=np.uint16)
    label_map = {
        1: LabelInfo(1, "a", (255, 0, 0), visible=True),
        2: LabelInfo(2, "b", (0, 255, 0), visible=False),
    }
    out = sr.composite_overlay(gray, mask_slice, label_map, alpha=1.0)
    assert tuple(out[0, 0]) == (255, 0, 0)   # visible label painted
    assert tuple(out[0, 1]) == (0, 0, 0)     # hidden label untouched


def test_default_window_level_spans_data_range():
    v = _volume()  # values 0..23
    center, width = sr.default_window_level(v)
    assert width >= 1
    assert center == 11.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_slice_renderer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomclient.slice_renderer'`

- [ ] **Step 3: Implement `slice_renderer.py`**

Create `src/stomclient/slice_renderer.py`:

```python
"""Pure slice-rendering math: plane extraction, window/level, mask overlay."""

from __future__ import annotations

import numpy as np

from stomcore.mask import LabelInfo
from stomcore.volume import Volume

AXIAL = "axial"
CORONAL = "coronal"
SAGITTAL = "sagittal"
PLANES = (AXIAL, CORONAL, SAGITTAL)


def slice_count(volume: Volume, plane: str) -> int:
    z, y, x = volume.shape
    return {AXIAL: z, CORONAL: y, SAGITTAL: x}[plane]


def slice_array(array: np.ndarray, plane: str, index: int) -> np.ndarray:
    """Extract a 2D slice from a [z, y, x] array. Returns [row, col]."""
    if plane == AXIAL:
        return array[index, :, :]      # [y, x]
    if plane == CORONAL:
        return array[:, index, :]      # [z, x]
    if plane == SAGITTAL:
        return array[:, :, index]      # [z, y]
    raise ValueError(f"unknown plane: {plane}")


def apply_window_level(slice2d: np.ndarray, center: float, width: float) -> np.ndarray:
    """Map intensities to uint8 [0, 255] using window center/width."""
    width = max(float(width), 1.0)
    lo = center - width / 2.0
    clipped = np.clip(slice2d.astype(np.float64), lo, lo + width)
    scaled = (clipped - lo) / width * 255.0
    return scaled.astype(np.uint8)


def composite_overlay(
    gray_uint8: np.ndarray,
    mask_slice: np.ndarray,
    label_map: dict[int, LabelInfo],
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend visible mask labels over a grayscale slice. Returns [row, col, 3] uint8."""
    rgb = np.repeat(gray_uint8[:, :, None].astype(np.float64), 3, axis=2)
    for label_id, info in label_map.items():
        if not info.visible:
            continue
        sel = mask_slice == label_id
        if not sel.any():
            continue
        color = np.array(info.color, dtype=np.float64)
        rgb[sel] = (1.0 - alpha) * rgb[sel] + alpha * color
    return np.clip(rgb, 0, 255).astype(np.uint8)


def default_window_level(volume: Volume) -> tuple[float, float]:
    """Center/width spanning the volume's intensity range."""
    data = volume.voxels
    lo = float(data.min())
    hi = float(data.max())
    width = max(hi - lo, 1.0)
    return (lo + hi) / 2.0, width
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_slice_renderer.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/stomclient/slice_renderer.py tests/test_slice_renderer.py
git commit -m "feat(client): slice extraction, window/level, mask overlay

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `measurement.py` — linear measurement in mm

**Files:**
- Create: `src/stomclient/measurement.py`
- Test: `tests/test_measurement.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_measurement.py`:

```python
import pytest

from stomcore.geometry import Geometry
from stomclient import slice_renderer as sr
from stomclient.measurement import LinearMeasurement, MeasurementSet, plane_spacing


def test_plane_spacing_maps_axes():
    geo = Geometry.identity(spacing=(0.3, 0.4, 0.5))  # x, y, z
    assert plane_spacing(geo, sr.AXIAL) == (0.3, 0.4)     # col=x, row=y
    assert plane_spacing(geo, sr.CORONAL) == (0.3, 0.5)   # col=x, row=z
    assert plane_spacing(geo, sr.SAGITTAL) == (0.4, 0.5)  # col=y, row=z


def test_length_isotropic_vertical():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    m = LinearMeasurement(p0=(0.0, 0.0), p1=(0.0, 10.0), plane=sr.AXIAL, geometry=geo)
    assert m.length_mm == pytest.approx(3.0)


def test_length_anisotropic():
    geo = Geometry.identity(spacing=(0.5, 1.0, 2.0))
    # axial: col spacing 0.5, row spacing 1.0; 3 cols, 4 rows
    m = LinearMeasurement(p0=(0.0, 0.0), p1=(3.0, 4.0), plane=sr.AXIAL, geometry=geo)
    # dx=3*0.5=1.5, dy=4*1.0=4.0 -> hypot=4.272...
    assert m.length_mm == pytest.approx(4.27200187, rel=1e-6)


def test_measurement_set_add_and_clear():
    geo = Geometry.identity(spacing=(1, 1, 1))
    ms = MeasurementSet()
    ms.add(LinearMeasurement((0, 0), (0, 1), sr.AXIAL, geo))
    assert len(ms) == 1
    ms.clear()
    assert len(ms) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_measurement.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomclient.measurement'`

- [ ] **Step 3: Implement `measurement.py`**

Create `src/stomclient/measurement.py`:

```python
"""Linear measurements in millimetres on a displayed 2D plane."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from stomcore.geometry import Geometry

from . import slice_renderer as sr


def plane_spacing(geometry: Geometry, plane: str) -> tuple[float, float]:
    """Return (col_spacing_mm, row_spacing_mm) for the displayed plane.

    Geometry.spacing is (x, y, z). Slice rows/cols per plane:
      axial    -> col=x, row=y
      coronal  -> col=x, row=z
      sagittal -> col=y, row=z
    """
    sx, sy, sz = geometry.spacing
    if plane == sr.AXIAL:
        return sx, sy
    if plane == sr.CORONAL:
        return sx, sz
    if plane == sr.SAGITTAL:
        return sy, sz
    raise ValueError(f"unknown plane: {plane}")


@dataclass(frozen=True)
class LinearMeasurement:
    p0: tuple[float, float]   # (col, row) in pixels
    p1: tuple[float, float]
    plane: str
    geometry: Geometry

    @property
    def length_mm(self) -> float:
        col_s, row_s = plane_spacing(self.geometry, self.plane)
        dc = (self.p1[0] - self.p0[0]) * col_s
        dr = (self.p1[1] - self.p0[1]) * row_s
        return math.hypot(dc, dr)


@dataclass
class MeasurementSet:
    items: list[LinearMeasurement] = field(default_factory=list)

    def add(self, m: LinearMeasurement) -> None:
        self.items.append(m)

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_measurement.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/stomclient/measurement.py tests/test_measurement.py
git commit -m "feat(client): linear measurement in mm

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: `serialization.py` — Volume/Mask <-> bytes bridge

**Files:**
- Create: `src/stomclient/serialization.py`
- Test: `tests/test_serialization.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_serialization.py`:

```python
import numpy as np

from stomcore.geometry import Geometry
from stomcore.mask import LabelInfo, SegmentationMask
from stomcore.volume import Volume
from stomclient.serialization import mask_from_bytes, volume_to_nifti_bytes


def test_volume_to_nifti_bytes_is_gzip_magic():
    vol = Volume(np.zeros((3, 4, 5), dtype=np.int16), Geometry.identity((0.3, 0.3, 0.3)))
    data = volume_to_nifti_bytes(vol)
    assert data[:2] == b"\x1f\x8b"  # gzip magic


def test_mask_from_bytes_roundtrip():
    geo = Geometry.identity((0.3, 0.3, 0.3))
    labels = np.zeros((3, 4, 5), dtype=np.uint16)
    labels[0, 0, 0] = 1
    label_map = {1: LabelInfo(1, "tooth", (255, 0, 0), True)}
    mask = SegmentationMask(labels, geo, label_map)

    from stomclient.serialization import mask_to_bytes

    mask_bytes, labels_bytes = mask_to_bytes(mask)
    restored = mask_from_bytes(mask_bytes, labels_bytes)

    assert restored.shape == (3, 4, 5)
    assert restored.label_map[1].name == "tooth"
    assert restored.is_compatible_with(Volume(labels, geo))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_serialization.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomclient.serialization'`

- [ ] **Step 3: Implement `serialization.py`**

Create `src/stomclient/serialization.py`:

```python
"""Bridge Volume/SegmentationMask to .nii.gz bytes via stomcore I/O."""

from __future__ import annotations

import tempfile
from pathlib import Path

from stomcore.mask import SegmentationMask
from stomcore.mask_io import load_mask_nifti, save_mask_nifti
from stomcore.nifti_io import save_volume_nifti
from stomcore.volume import Volume


def volume_to_nifti_bytes(volume: Volume) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "volume.nii.gz"
        save_volume_nifti(volume, path)
        return path.read_bytes()


def mask_to_bytes(mask: SegmentationMask) -> tuple[bytes, bytes]:
    with tempfile.TemporaryDirectory() as tmp:
        nifti = Path(tmp) / "mask.nii.gz"
        labels = Path(tmp) / "labels.json"
        save_mask_nifti(mask, nifti, labels)
        return nifti.read_bytes(), labels.read_bytes()


def mask_from_bytes(mask_bytes: bytes, labels_bytes: bytes) -> SegmentationMask:
    with tempfile.TemporaryDirectory() as tmp:
        nifti = Path(tmp) / "mask.nii.gz"
        labels = Path(tmp) / "labels.json"
        nifti.write_bytes(mask_bytes)
        labels.write_bytes(labels_bytes)
        return load_mask_nifti(nifti, labels)
```

(`mask_to_bytes` is used by tests/fakes to fabricate cloud responses; `mask_from_bytes` is used by the controller; `volume_to_nifti_bytes` by upload.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_serialization.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/stomclient/serialization.py tests/test_serialization.py
git commit -m "feat(client): Volume/Mask <-> nifti bytes bridge

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: `app_controller.py` — session state machine

**Files:**
- Create: `src/stomclient/app_controller.py`
- Test: `tests/test_app_controller.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_app_controller.py`:

```python
import numpy as np

from stomcore.geometry import Geometry
from stomcore.mask import LabelInfo, SegmentationMask
from stomcore.volume import Volume
from stomclient import slice_renderer as sr
from stomclient.app_controller import AppController, State
from stomclient.cloud_client import JobStatus, StudyInfo
from stomclient.serialization import mask_to_bytes


def _volume(geo=None):
    geo = geo or Geometry.identity((0.3, 0.3, 0.3))
    return Volume(np.zeros((4, 5, 6), dtype=np.int16), geo)


def _mask_bytes(geo):
    labels = np.zeros((4, 5, 6), dtype=np.uint16)
    labels[0, 0, 0] = 1
    mask = SegmentationMask(labels, geo, {1: LabelInfo(1, "t", (255, 0, 0), True)})
    return mask_to_bytes(mask)


class FakeCloud:
    def __init__(self, status_sequence, mask_bytes=(b"", b"")):
        self._statuses = list(status_sequence)
        self._mask_bytes = mask_bytes
        self.uploaded = False

    def upload_study(self, nifti_bytes, filename):
        self.uploaded = True
        return StudyInfo(study_id=1, shape=[4, 5, 6], spacing=[0.3, 0.3, 0.3])

    def start_segmentation(self, study_id):
        return JobStatus(job_id=9, status="queued")

    def poll_status(self, job_id):
        return self._statuses.pop(0)

    def download_mask(self, study_id):
        return self._mask_bytes


def test_load_volume_centers_index_and_sets_state():
    c = AppController(FakeCloud([]))
    c.load_volume(_volume())
    assert c.state == State.LOADED
    assert c.plane == sr.AXIAL
    assert c.index == sr.slice_count(c.volume, sr.AXIAL) // 2


def test_submit_transitions_to_segmenting():
    cloud = FakeCloud([])
    c = AppController(cloud)
    c.load_volume(_volume())
    c.submit()
    assert cloud.uploaded is True
    assert c.state == State.SEGMENTING
    assert c.study_id == 1
    assert c.job_id == 9


def test_poll_running_then_done_loads_mask():
    geo = Geometry.identity((0.3, 0.3, 0.3))
    cloud = FakeCloud(
        [JobStatus(9, "running"), JobStatus(9, "done")],
        mask_bytes=_mask_bytes(geo),
    )
    c = AppController(cloud)
    c.load_volume(_volume(geo))
    c.submit()
    assert c.poll() is False           # still running
    assert c.state == State.SEGMENTING
    assert c.poll() is True            # done
    assert c.state == State.MASK_READY
    assert c.mask is not None


def test_poll_failed_sets_failed_state():
    cloud = FakeCloud([JobStatus(9, "failed", error="OOM")])
    c = AppController(cloud)
    c.load_volume(_volume())
    c.submit()
    assert c.poll() is True
    assert c.state == State.FAILED
    assert "OOM" in c.error


def test_poll_done_with_incompatible_mask_is_rejected():
    vol_geo = Geometry.identity((0.3, 0.3, 0.3))
    drift_geo = Geometry.identity((1.0, 1.0, 1.0))  # different spacing
    cloud = FakeCloud([JobStatus(9, "done")], mask_bytes=_mask_bytes(drift_geo))
    c = AppController(cloud)
    c.load_volume(_volume(vol_geo))
    c.submit()
    assert c.poll() is True
    assert c.state == State.FAILED
    assert "geometry" in c.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_app_controller.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomclient.app_controller'`

- [ ] **Step 3: Implement `app_controller.py`**

Create `src/stomclient/app_controller.py`:

```python
"""Qt-agnostic session state machine driving cloud + rendering state."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from stomcore.volume import Volume

from . import slice_renderer as sr
from .measurement import MeasurementSet
from .serialization import mask_from_bytes, volume_to_nifti_bytes


class State(str, Enum):
    EMPTY = "empty"
    LOADED = "loaded"
    UPLOADING = "uploading"
    SEGMENTING = "segmenting"
    MASK_READY = "mask_ready"
    FAILED = "failed"


class AppController:
    def __init__(self, cloud_client, on_change: Callable[[], None] = lambda: None) -> None:
        self._cloud = cloud_client
        self._on_change = on_change
        self.state = State.EMPTY
        self.volume: Volume | None = None
        self.mask = None
        self.plane = sr.AXIAL
        self.index = 0
        self.window_center = 0.0
        self.window_width = 1.0
        self.measurements = MeasurementSet()
        self.study_id: int | None = None
        self.job_id: int | None = None
        self.error: str | None = None

    def _changed(self) -> None:
        self._on_change()

    def load_volume(self, volume: Volume) -> None:
        self.volume = volume
        self.mask = None
        self.plane = sr.AXIAL
        self.index = sr.slice_count(volume, sr.AXIAL) // 2
        self.window_center, self.window_width = sr.default_window_level(volume)
        self.measurements = MeasurementSet()
        self.study_id = self.job_id = None
        self.error = None
        self.state = State.LOADED
        self._changed()

    def set_plane(self, plane: str) -> None:
        self.plane = plane
        count = sr.slice_count(self.volume, plane)
        self.index = min(self.index, count - 1)
        self._changed()

    def set_index(self, index: int) -> None:
        count = sr.slice_count(self.volume, self.plane)
        self.index = max(0, min(index, count - 1))
        self._changed()

    def set_window_level(self, center: float, width: float) -> None:
        self.window_center = center
        self.window_width = max(width, 1.0)
        self._changed()

    def submit(self) -> None:
        if self.state not in (State.LOADED, State.FAILED, State.MASK_READY):
            raise RuntimeError(f"cannot submit from state {self.state}")
        self.error = None
        self.state = State.UPLOADING
        self._changed()
        nifti = volume_to_nifti_bytes(self.volume)
        info = self._cloud.upload_study(nifti, "study.nii.gz")
        self.study_id = info.study_id
        job = self._cloud.start_segmentation(info.study_id)
        self.job_id = job.job_id
        self.state = State.SEGMENTING
        self._changed()

    def poll(self) -> bool:
        """Poll once. Returns True when the job reached a terminal state."""
        job = self._cloud.poll_status(self.job_id)
        if job.status == "failed":
            self.error = job.error or "segmentation failed"
            self.state = State.FAILED
            self._changed()
            return True
        if job.status == "done":
            mask_bytes, labels_bytes = self._cloud.download_mask(self.study_id)
            mask = mask_from_bytes(mask_bytes, labels_bytes)
            if not mask.is_compatible_with(self.volume):
                self.error = "returned mask geometry does not match volume"
                self.state = State.FAILED
                self._changed()
                return True
            self.mask = mask
            self.state = State.MASK_READY
            self._changed()
            return True
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_app_controller.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/stomclient/app_controller.py tests/test_app_controller.py
git commit -m "feat(client): session state machine with geometry-checked mask load

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: UI helper + `SliceWidget` (offscreen smoke)

**Files:**
- Create: `src/stomclient/ui/__init__.py`
- Create: `src/stomclient/ui/qt_image.py`
- Create: `src/stomclient/ui/slice_widget.py`
- Test: `tests/test_client_ui_smoke.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_client_ui_smoke.py`:

```python
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from stomcore.geometry import Geometry
from stomcore.volume import Volume
from stomclient.app_controller import AppController


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_ndarray_to_qimage_dimensions(qapp):
    from stomclient.ui.qt_image import ndarray_to_qimage

    rgb = np.zeros((5, 7, 3), dtype=np.uint8)
    img = ndarray_to_qimage(rgb)
    assert img.width() == 7
    assert img.height() == 5


def test_slice_widget_renders_loaded_volume(qapp):
    from stomclient.ui.slice_widget import SliceWidget

    controller = AppController(cloud_client=None)
    controller.load_volume(
        Volume(np.zeros((4, 5, 6), dtype=np.int16), Geometry.identity((0.3, 0.3, 0.3)))
    )
    widget = SliceWidget(controller)
    img = widget.render_image()
    assert img.width() == 6   # axial -> x columns
    assert img.height() == 5  # axial -> y rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_client_ui_smoke.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomclient.ui'`

- [ ] **Step 3: Implement the UI helper and widget**

Create `src/stomclient/ui/__init__.py`:

```python
"""Qt view layer for the Stom desktop client."""
```

Create `src/stomclient/ui/qt_image.py`:

```python
"""Convert numpy RGB arrays to QImage."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def ndarray_to_qimage(rgb: np.ndarray) -> QImage:
    """rgb: [row, col, 3] uint8, C-contiguous. Returns an owned QImage copy."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return img.copy()  # detach from the numpy buffer
```

Create `src/stomclient/ui/slice_widget.py`:

```python
"""Thin 2D slice view: builds a QImage from controller state and paints it."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget

from .. import slice_renderer as sr
from ..app_controller import AppController
from .qt_image import ndarray_to_qimage


class SliceWidget(QWidget):
    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller
        self.setMinimumSize(256, 256)

    def render_image(self) -> QImage:
        c = self._c
        if c.volume is None:
            return QImage(1, 1, QImage.Format.Format_RGB888)
        gray = sr.apply_window_level(
            sr.slice_array(c.volume.voxels, c.plane, c.index),
            c.window_center, c.window_width,
        )
        if c.mask is not None:
            mask_slice = sr.slice_array(c.mask.labels, c.plane, c.index)
            rgb = sr.composite_overlay(gray, mask_slice, c.mask.label_map, alpha=0.5)
        else:
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
        return ndarray_to_qimage(rgb)

    def wheelEvent(self, event) -> None:  # scroll changes slice index
        step = 1 if event.angleDelta().y() > 0 else -1
        self._c.set_index(self._c.index + step)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        img = self.render_image()
        target = img.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawImage(0, 0, target)
        painter.end()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_client_ui_smoke.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/stomclient/ui/__init__.py src/stomclient/ui/qt_image.py src/stomclient/ui/slice_widget.py tests/test_client_ui_smoke.py
git commit -m "feat(client): SliceWidget renders slices with overlay

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: `SettingsDialog`

**Files:**
- Create: `src/stomclient/ui/settings_dialog.py`
- Modify: `tests/test_client_ui_smoke.py`

- [ ] **Step 1: Append the failing test**

Add to `tests/test_client_ui_smoke.py`:

```python
def test_settings_dialog_values_roundtrip(qapp):
    from stomclient.config import ClientConfig
    from stomclient.ui.settings_dialog import SettingsDialog

    dialog = SettingsDialog(ClientConfig(server_url="https://api", token="tok"))
    cfg = dialog.values()
    assert cfg.server_url == "https://api"
    assert cfg.token == "tok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_client_ui_smoke.py::test_settings_dialog_values_roundtrip -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomclient.ui.settings_dialog'`

- [ ] **Step 3: Implement `settings_dialog.py`**

Create `src/stomclient/ui/settings_dialog.py`:

```python
"""URL + token settings dialog, persisted via config.py."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
)

from ..config import ClientConfig


class SettingsDialog(QDialog):
    def __init__(self, config: ClientConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self._url = QLineEdit(config.server_url)
        self._token = QLineEdit(config.token or "")
        self._token.setEchoMode(QLineEdit.EchoMode.Password)
        self._save_token = QCheckBox("Save token to disk")
        self._save_token.setChecked(config.save_token)

        form = QFormLayout(self)
        form.addRow("Server URL", self._url)
        form.addRow("API token", self._token)
        form.addRow(self._save_token)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> ClientConfig:
        token = self._token.text().strip() or None
        return ClientConfig(
            server_url=self._url.text().strip(),
            token=token,
            save_token=self._save_token.isChecked(),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_client_ui_smoke.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/stomclient/ui/settings_dialog.py tests/test_client_ui_smoke.py
git commit -m "feat(client): settings dialog for server URL + token

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: `MainWindow` + cloud worker thread

**Files:**
- Create: `src/stomclient/ui/main_window.py`
- Modify: `tests/test_client_ui_smoke.py`

- [ ] **Step 1: Append the failing test**

Add to `tests/test_client_ui_smoke.py`:

```python
def test_main_window_builds_with_controller(qapp):
    from stomclient.app_controller import AppController
    from stomclient.ui.main_window import MainWindow

    window = MainWindow(AppController(cloud_client=None))
    assert "Stom" in window.windowTitle()
    assert window.slice_widget is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_client_ui_smoke.py::test_main_window_builds_with_controller -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomclient.ui.main_window'`

- [ ] **Step 3: Implement `main_window.py`**

Create `src/stomclient/ui/main_window.py`:

```python
"""Main window: left panel + slice view + toolbar wiring. Thin over AppController."""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from stomcore.dicom_loader import DicomError, DicomLoader

from .. import slice_renderer as sr
from ..app_controller import AppController, State
from ..cloud_client import CloudError
from .slice_widget import SliceWidget


class _SubmitWorker(QObject):
    """Runs the blocking submit() off the UI thread."""

    done = Signal()
    failed = Signal(str)

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller

    def run(self) -> None:
        try:
            self._c.submit()
            self.done.emit()
        except (CloudError, RuntimeError) as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller
        self.setWindowTitle("Stom — CBCT Viewer")

        self.slice_widget = SliceWidget(controller)
        self._status = QLabel("No study")
        self._plane = QComboBox()
        self._plane.addItems(list(sr.PLANES))
        self._plane.currentTextChanged.connect(self._on_plane)

        open_btn = QPushButton("Open DICOM…")
        open_btn.clicked.connect(self._on_open)
        segment_btn = QPushButton("Upload & Segment")
        segment_btn.clicked.connect(self._on_segment)

        left = QVBoxLayout()
        left.addWidget(open_btn)
        left.addWidget(segment_btn)
        left.addWidget(self._plane)
        left.addWidget(self._status)
        left.addStretch(1)
        left_panel = QWidget()
        left_panel.setLayout(left)

        root = QHBoxLayout()
        root.addWidget(left_panel, 0)
        root.addWidget(self.slice_widget, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._on_poll_tick)
        self._thread: QThread | None = None

    def _refresh(self) -> None:
        self._status.setText(self._c.state.value)
        self.slice_widget.update()

    def _on_plane(self, plane: str) -> None:
        if self._c.volume is not None:
            self._c.set_plane(plane)
            self._refresh()

    def _on_open(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open DICOM series")
        if not directory:
            return
        try:
            volume = DicomLoader.load(directory)
        except DicomError as exc:
            QMessageBox.critical(self, "DICOM error", str(exc))
            return
        self._c.load_volume(volume)
        self._refresh()

    def _on_segment(self) -> None:
        if self._c.volume is None:
            QMessageBox.information(self, "No study", "Open a DICOM series first.")
            return
        self._thread = QThread(self)
        worker = _SubmitWorker(self._c)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.done.connect(self._on_submitted)
        worker.failed.connect(self._on_submit_failed)
        worker.done.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
        self._worker = worker  # keep ref
        self._thread.start()
        self._refresh()

    def _on_submitted(self) -> None:
        self._refresh()
        self._poll_timer.start()

    def _on_submit_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Cloud error", message)
        self._refresh()

    def _on_poll_tick(self) -> None:
        try:
            terminal = self._c.poll()
        except CloudError as exc:
            self._poll_timer.stop()
            QMessageBox.critical(self, "Cloud error", str(exc))
            return
        if terminal:
            self._poll_timer.stop()
            if self._c.state == State.FAILED:
                QMessageBox.warning(self, "Segmentation failed", self._c.error or "")
        self._refresh()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_client_ui_smoke.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/stomclient/ui/main_window.py tests/test_client_ui_smoke.py
git commit -m "feat(client): main window wiring open/segment/poll

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Controller interactions + coordinate mapping

**Files:**
- Create: `src/stomclient/coords.py`
- Modify: `src/stomclient/app_controller.py`
- Test: `tests/test_coords.py`, `tests/test_app_controller.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_coords.py`:

```python
from stomclient.coords import widget_to_image


def test_top_left_anchored_unscaled():
    # widget 200x100, image 100x100 -> scale=min(2,1)=1, drawn top-left 100x100
    assert widget_to_image((50, 50), (200, 100), (100, 100)) == (50.0, 50.0)


def test_scaled_down():
    # widget 200x200, image 100x100 -> scale=2 -> click (100,100) maps to (50,50)
    assert widget_to_image((100, 100), (200, 200), (100, 100)) == (50.0, 50.0)


def test_outside_image_returns_none():
    assert widget_to_image((150, 50), (200, 100), (100, 100)) is None
```

Append to `tests/test_app_controller.py`:

```python
from dataclasses import replace as _replace  # noqa: F401  (ensures dataclasses import availability)


def test_set_label_visible_toggles():
    geo = Geometry.identity((0.3, 0.3, 0.3))
    c = AppController(FakeCloud([]))
    c.load_volume(_volume(geo))
    labels = np.zeros((4, 5, 6), dtype=np.uint16)
    c.mask = SegmentationMask(labels, geo, {1: LabelInfo(1, "t", (255, 0, 0), True)})
    c.set_label_visible(1, False)
    assert c.mask.label_map[1].visible is False


def test_add_and_clear_measurements():
    c = AppController(FakeCloud([]))
    c.load_volume(_volume())
    c.add_measurement((0.0, 0.0), (0.0, 10.0))
    assert len(c.measurements) == 1
    c.clear_measurements()
    assert len(c.measurements) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_coords.py tests/test_app_controller.py -q`
Expected: FAIL — `No module named 'stomclient.coords'` and `AttributeError: 'AppController' object has no attribute 'set_label_visible'`

- [ ] **Step 3: Implement `coords.py`**

Create `src/stomclient/coords.py`:

```python
"""Pure mapping from widget pixel coordinates to image (col, row) pixels.

The slice image is drawn top-left anchored with KeepAspectRatio, so the
visible image occupies [0, image_w*scale] x [0, image_h*scale] where
scale = min(widget_w/image_w, widget_h/image_h).
"""

from __future__ import annotations


def widget_to_image(
    pos: tuple[float, float],
    widget_size: tuple[float, float],
    image_size: tuple[float, float],
) -> tuple[float, float] | None:
    ww, wh = widget_size
    iw, ih = image_size
    if iw <= 0 or ih <= 0:
        return None
    scale = min(ww / iw, wh / ih)
    if scale <= 0:
        return None
    col = pos[0] / scale
    row = pos[1] / scale
    if col < 0 or row < 0 or col > iw or row > ih:
        return None
    return (col, row)
```

- [ ] **Step 4: Add interaction methods to `AppController`**

In `src/stomclient/app_controller.py`, update the imports block:

```python
from dataclasses import replace

from stomcore.mask import SegmentationMask
from stomcore.volume import Volume

from . import slice_renderer as sr
from .measurement import LinearMeasurement, MeasurementSet
from .serialization import mask_from_bytes, volume_to_nifti_bytes
```

Then add these methods to the `AppController` class (after `set_window_level`):

```python
    def set_label_visible(self, label_id: int, visible: bool) -> None:
        if self.mask is None:
            return
        info = self.mask.label_map.get(label_id)
        if info is None:
            return
        new_map = dict(self.mask.label_map)
        new_map[label_id] = replace(info, visible=visible)
        self.mask = SegmentationMask(self.mask.labels, self.mask.geometry, new_map)
        self._changed()

    def add_measurement(self, p0: tuple[float, float], p1: tuple[float, float]) -> None:
        if self.volume is None:
            return
        self.measurements.add(
            LinearMeasurement(p0, p1, self.plane, self.volume.geometry)
        )
        self._changed()

    def clear_measurements(self) -> None:
        self.measurements.clear()
        self._changed()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_coords.py tests/test_app_controller.py -q`
Expected: PASS (3 + 7 = 10 passed)

- [ ] **Step 6: Commit**

```bash
git add src/stomclient/coords.py src/stomclient/app_controller.py tests/test_coords.py tests/test_app_controller.py
git commit -m "feat(client): label-visibility + measurement interactions, coord mapping

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Wire UI — mask list, W/L drag, measure mode, PNG/mask export

**Files:**
- Modify: `src/stomclient/ui/slice_widget.py`
- Modify: `src/stomclient/ui/main_window.py`
- Modify: `tests/test_client_ui_smoke.py`

- [ ] **Step 1: Append the failing tests to `tests/test_client_ui_smoke.py`**

```python
def test_slice_widget_measure_mode_toggle(qapp):
    from stomclient.ui.slice_widget import SliceWidget

    controller = AppController(cloud_client=None)
    controller.load_volume(
        Volume(np.zeros((4, 5, 6), dtype=np.int16), Geometry.identity((0.3, 0.3, 0.3)))
    )
    widget = SliceWidget(controller)
    widget.set_measure_mode(True)
    assert widget.measure_mode is True


def test_main_window_mask_list_populates(qapp):
    from stomcore.mask import LabelInfo, SegmentationMask
    from stomclient.ui.main_window import MainWindow

    controller = AppController(cloud_client=None)
    geo = Geometry.identity((0.3, 0.3, 0.3))
    controller.load_volume(Volume(np.zeros((4, 5, 6), dtype=np.int16), geo))
    controller.mask = SegmentationMask(
        np.zeros((4, 5, 6), dtype=np.uint16), geo,
        {1: LabelInfo(1, "tooth", (255, 0, 0), True),
         2: LabelInfo(2, "canal", (0, 255, 0), True)},
    )
    window = MainWindow(controller)
    window.refresh()
    assert window.mask_list.count() == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_client_ui_smoke.py -q`
Expected: FAIL — `AttributeError: 'SliceWidget' object has no attribute 'set_measure_mode'`

- [ ] **Step 3: Replace `src/stomclient/ui/slice_widget.py` with the interactive version**

```python
"""Thin 2D slice view: window/level drag, measurement drawing, mask overlay."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .. import slice_renderer as sr
from ..app_controller import AppController
from ..coords import widget_to_image
from .qt_image import ndarray_to_qimage


class SliceWidget(QWidget):
    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller
        self.setMinimumSize(256, 256)
        self.measure_mode = False
        self._drag_start: tuple[float, float] | None = None   # measure, image coords
        self._drag_end: tuple[float, float] | None = None
        self._wl_anchor: tuple[float, float] | None = None     # window/level drag, widget px

    def set_measure_mode(self, on: bool) -> None:
        self.measure_mode = on
        self._drag_start = self._drag_end = None
        self.update()

    def _image_size(self) -> tuple[int, int]:
        c = self._c
        if c.volume is None:
            return (1, 1)
        z, y, x = c.volume.shape
        return {sr.AXIAL: (x, y), sr.CORONAL: (x, z), sr.SAGITTAL: (y, z)}[c.plane]

    def render_image(self) -> QImage:
        c = self._c
        if c.volume is None:
            return QImage(1, 1, QImage.Format.Format_RGB888)
        gray = sr.apply_window_level(
            sr.slice_array(c.volume.voxels, c.plane, c.index),
            c.window_center, c.window_width,
        )
        if c.mask is not None:
            mask_slice = sr.slice_array(c.mask.labels, c.plane, c.index)
            rgb = sr.composite_overlay(gray, mask_slice, c.mask.label_map, alpha=0.5)
        else:
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
        return ndarray_to_qimage(rgb)

    def wheelEvent(self, event) -> None:
        step = 1 if event.angleDelta().y() > 0 else -1
        self._c.set_index(self._c.index + step)
        self.update()

    def mousePressEvent(self, event) -> None:
        pos = (event.position().x(), event.position().y())
        if self.measure_mode:
            self._drag_start = widget_to_image(pos, (self.width(), self.height()), self._image_size())
            self._drag_end = self._drag_start
        else:
            self._wl_anchor = pos
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = (event.position().x(), event.position().y())
        if self.measure_mode and self._drag_start is not None:
            self._drag_end = widget_to_image(pos, (self.width(), self.height()), self._image_size())
        elif self._wl_anchor is not None:
            dx = pos[0] - self._wl_anchor[0]
            dy = pos[1] - self._wl_anchor[1]
            self._wl_anchor = pos
            self._c.set_window_level(self._c.window_center + dy, self._c.window_width + dx)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self.measure_mode and self._drag_start and self._drag_end:
            self._c.add_measurement(self._drag_start, self._drag_end)
            self._drag_start = self._drag_end = None
        self._wl_anchor = None
        self.update()

    def _scale(self) -> float:
        iw, ih = self._image_size()
        return min(self.width() / iw, self.height() / ih)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        img = self.render_image()
        scale = self._scale()
        target = img.scaled(
            int(img.width() * scale), int(img.height() * scale),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawImage(0, 0, target)

        pen = QPen(Qt.GlobalColor.yellow)
        pen.setWidth(1)
        painter.setPen(pen)
        for m in self._c.measurements:
            self._draw_line(painter, m.p0, m.p1, scale, f"{m.length_mm:.1f} mm")
        if self._drag_start and self._drag_end:
            self._draw_line(painter, self._drag_start, self._drag_end, scale, "")
        painter.end()

    def _draw_line(self, painter, p0, p1, scale, label) -> None:
        a = QPointF(p0[0] * scale, p0[1] * scale)
        b = QPointF(p1[0] * scale, p1[1] * scale)
        painter.drawLine(a, b)
        if label:
            painter.drawText(b, label)
```

- [ ] **Step 4: Replace `src/stomclient/ui/main_window.py` with the version adding mask list + export**

```python
"""Main window: panel (mask list, tools), slice view, cloud roundtrip wiring."""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from stomcore.dicom_loader import DicomError, DicomLoader
from stomcore.mask_io import save_mask_nifti

from .. import slice_renderer as sr
from ..app_controller import AppController, State
from ..cloud_client import CloudError
from .slice_widget import SliceWidget


class _SubmitWorker(QObject):
    done = Signal()
    failed = Signal(str)

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller

    def run(self) -> None:
        try:
            self._c.submit()
            self.done.emit()
        except (CloudError, RuntimeError) as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self._c = controller
        self.setWindowTitle("Stom — CBCT Viewer")

        self.slice_widget = SliceWidget(controller)
        self._status = QLabel("No study")
        self._plane = QComboBox()
        self._plane.addItems(list(sr.PLANES))
        self._plane.currentTextChanged.connect(self._on_plane)

        open_btn = QPushButton("Open DICOM…")
        open_btn.clicked.connect(self._on_open)
        segment_btn = QPushButton("Upload & Segment")
        segment_btn.clicked.connect(self._on_segment)
        self._measure_btn = QPushButton("Measure")
        self._measure_btn.setCheckable(True)
        self._measure_btn.toggled.connect(self.slice_widget.set_measure_mode)
        clear_btn = QPushButton("Clear measurements")
        clear_btn.clicked.connect(self._on_clear_measurements)
        png_btn = QPushButton("Save PNG…")
        png_btn.clicked.connect(self._on_save_png)
        mask_btn = QPushButton("Save Mask…")
        mask_btn.clicked.connect(self._on_save_mask)

        self.mask_list = QListWidget()
        self.mask_list.itemChanged.connect(self._on_mask_item_changed)

        left = QVBoxLayout()
        for w in (open_btn, segment_btn, self._plane, self._measure_btn,
                  clear_btn, png_btn, mask_btn, QLabel("Masks:"), self.mask_list,
                  self._status):
            left.addWidget(w)
        left.addStretch(1)
        left_panel = QWidget()
        left_panel.setLayout(left)

        root = QHBoxLayout()
        root.addWidget(left_panel, 0)
        root.addWidget(self.slice_widget, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(2000)
        self._poll_timer.timeout.connect(self._on_poll_tick)
        self._thread: QThread | None = None

    def refresh(self) -> None:
        self._status.setText(self._c.state.value)
        self._rebuild_mask_list()
        self.slice_widget.update()

    def _rebuild_mask_list(self) -> None:
        self.mask_list.blockSignals(True)
        self.mask_list.clear()
        if self._c.mask is not None:
            for label_id, info in sorted(self._c.mask.label_map.items()):
                item = QListWidgetItem(f"{label_id}: {info.name}")
                item.setData(Qt.ItemDataRole.UserRole, label_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    Qt.CheckState.Checked if info.visible else Qt.CheckState.Unchecked
                )
                self.mask_list.addItem(item)
        self.mask_list.blockSignals(False)

    def _on_mask_item_changed(self, item: QListWidgetItem) -> None:
        label_id = item.data(Qt.ItemDataRole.UserRole)
        self._c.set_label_visible(label_id, item.checkState() == Qt.CheckState.Checked)
        self.slice_widget.update()

    def _on_plane(self, plane: str) -> None:
        if self._c.volume is not None:
            self._c.set_plane(plane)
            self.refresh()

    def _on_open(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open DICOM series")
        if not directory:
            return
        try:
            volume = DicomLoader.load(directory)
        except DicomError as exc:
            QMessageBox.critical(self, "DICOM error", str(exc))
            return
        self._c.load_volume(volume)
        self.refresh()

    def _on_clear_measurements(self) -> None:
        self._c.clear_measurements()
        self.slice_widget.update()

    def _on_save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "slice.png", "PNG (*.png)")
        if path:
            self.slice_widget.render_image().save(path, "PNG")

    def _on_save_mask(self) -> None:
        if self._c.mask is None:
            QMessageBox.information(self, "No mask", "No mask to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save mask", "mask.nii.gz", "NIfTI (*.nii.gz)")
        if path:
            labels_path = path.replace(".nii.gz", "").rstrip(".") + "_labels.json"
            save_mask_nifti(self._c.mask, path, labels_path)

    def _on_segment(self) -> None:
        if self._c.volume is None:
            QMessageBox.information(self, "No study", "Open a DICOM series first.")
            return
        self._thread = QThread(self)
        worker = _SubmitWorker(self._c)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.done.connect(self._on_submitted)
        worker.failed.connect(self._on_submit_failed)
        worker.done.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
        self._worker = worker
        self._thread.start()
        self.refresh()

    def _on_submitted(self) -> None:
        self.refresh()
        self._poll_timer.start()

    def _on_submit_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Cloud error", message)
        self.refresh()

    def _on_poll_tick(self) -> None:
        try:
            terminal = self._c.poll()
        except CloudError as exc:
            self._poll_timer.stop()
            QMessageBox.critical(self, "Cloud error", str(exc))
            return
        if terminal:
            self._poll_timer.stop()
            if self._c.state == State.FAILED:
                QMessageBox.warning(self, "Segmentation failed", self._c.error or "")
        self.refresh()
```

Note: `MainWindow` now exposes a public `refresh()` (replacing the earlier private `_refresh`) and a `mask_list` attribute, both used by tests.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_client_ui_smoke.py -q`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add src/stomclient/ui/slice_widget.py src/stomclient/ui/main_window.py tests/test_client_ui_smoke.py
git commit -m "feat(client): mask list, W/L drag, measure drawing, PNG/mask export

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Entrypoint + README + full suite

**Files:**
- Create: `src/stomclient/__main__.py`
- Create: `src/stomclient/README.md`
- Test: full suite

- [ ] **Step 1: Implement the entrypoint**

Create `src/stomclient/__main__.py`:

```python
"""Launch the Stom desktop client."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .app_controller import AppController
    from .cloud_client import CloudClient
    from .config import load
    from .ui.main_window import MainWindow

    config = load()
    cloud = CloudClient(config.server_url, config.token) if config.server_url else None

    app = QApplication(sys.argv)
    window = MainWindow(AppController(cloud_client=cloud))
    window.resize(1100, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write the README**

Create `src/stomclient/README.md`:

```markdown
# stomclient — Stom desktop client

PySide6 desktop app: import DICOM CBCT, run cloud AI segmentation via
`stomserver`, overlay masks on 2D slices, take linear measurements in mm.

## Install

    pip install -e ".[dev,client]"

## Run

    stom-client          # or: python -m stomclient

On first run open **Settings** and enter the server URL (e.g. `http://localhost:8000`)
and your API token (issued by `scripts/create_account.py` on the server).
Config is saved to `~/.config/stom/client.toml` (mode 0600).

## Architecture

Thin-view / testable-core. Logic (config, cloud_client, slice_renderer,
measurement, app_controller) has no Qt imports and is unit-tested headless.
The `ui/` package holds the PySide6 widgets. 2D rendering is `numpy → QImage`
(VTK/3D deferred to a fast-follow). Reuses `stomcore` for DICOM/NIfTI/mask I/O.

## Tests

    python -m pytest tests/test_client_*.py tests/test_cloud_client.py \
        tests/test_slice_renderer.py tests/test_measurement.py \
        tests/test_serialization.py tests/test_coords.py \
        tests/test_app_controller.py -q

UI smoke tests run under `QT_QPA_PLATFORM=offscreen` and skip if PySide6 is absent.
```

- [ ] **Step 3: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — the Plan 2 suite (73 passed, 1 skipped) plus all new client tests, no regressions.

- [ ] **Step 4: Manual smoke (optional, needs a display + running server)**

Run: `stom-client`
Expected: window opens; Open DICOM loads a series; with a running `stomserver` + token in Settings, Upload & Segment runs the cloud roundtrip and overlays the mask.

- [ ] **Step 5: Commit**

```bash
git add src/stomclient/__main__.py src/stomclient/README.md
git commit -m "feat(client): entrypoint + README

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Deferred to fast-follow (NOT in this plan)

Tracked from the spec §2/§11 — do not implement here:
- Manual mask editing (`MaskEditor`: brush/eraser/undo) and re-upload of edits.
- 3D volume render + 2×2 multi-plane cross (introduce VTK then).
- Angular measurements.
- DICOM anonymisation before upload.
- Reconnect to an in-flight job by `job_id` after restart; local study history.
