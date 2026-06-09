# Core Data Layer (Ядро данных) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared data foundation for the CBCT segmentation product — load DICOM CBCT series, hold voxel data with spatial geometry, hold multi-label segmentation masks, convert to/from NIfTI — exposed as a library plus a `dicom→nifti` CLI.

**Architecture:** A pure-Python core (`Geometry`, `Volume`, `SegmentationMask`) with no imaging-toolkit dependency, plus a thin SimpleITK interop layer (`sitk_interop`, `nifti_io`, `dicom_loader`) that is the only code touching SimpleITK. The CLI wires `DicomLoader` → `nifti_io`. This keeps spatial logic unit-testable in isolation and confines I/O to small adapters.

**Tech Stack:** Python 3.11+, NumPy, SimpleITK (ITK Python binding), pytest. `src/` layout, packaged with setuptools via `pyproject.toml`.

---

## File Structure

```
Stom/
  pyproject.toml                         # package metadata, deps, pytest + console-script
  src/stomcore/
    __init__.py                          # version + public re-exports
    geometry.py                          # Geometry (spacing/origin/direction) — pure, no sitk
    volume.py                            # Volume (voxels + Geometry) — pure, no sitk
    mask.py                              # LabelInfo, SegmentationMask — pure, no sitk
    sitk_interop.py                      # Volume/Geometry <-> SimpleITK.Image (only sitk bridge)
    nifti_io.py                          # save/load Volume as .nii.gz
    dicom_loader.py                      # DicomLoader.load(dir) -> Volume, DicomError
    cli.py                               # `stom-dicom2nifti` entry point
  tests/
    conftest.py                          # synthetic fixtures (identity geometry, dicom series writer)
    test_geometry.py
    test_volume.py
    test_mask.py
    test_sitk_interop.py
    test_nifti_io.py
    test_dicom_loader.py
    test_cli.py
```

**Responsibilities:**
- `geometry.py` / `volume.py` / `mask.py` — pure data + invariants, hold no I/O. The single source of spatial truth.
- `sitk_interop.py` — the *only* module converting between our types and `SimpleITK.Image`. Everything else reuses it (DRY).
- `nifti_io.py` / `dicom_loader.py` — file I/O adapters built on `sitk_interop`.
- `cli.py` — argument parsing + orchestration only, no logic of its own.

---

## Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/stomcore/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `.gitignore`

- [ ] **Step 1: Initialize git repository**

The project directory is not yet a git repo. Run:

```bash
cd /opt/almaz/test/Stom
git init
git config user.name "Stom Dev"
git config user.email "dev@stom.local"
```

Expected: `Initialized empty Git repository in /opt/almaz/test/Stom/.git/`

- [ ] **Step 2: Create `.gitignore`**

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/
build/
dist/
.venv/
venv/
*.nii.gz
*.dcm
```

- [ ] **Step 3: Create `pyproject.toml`**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "stomcore"
version = "0.1.0"
description = "Core data layer for CBCT segmentation (DICOM/NIfTI, Volume, masks)"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "SimpleITK>=2.3",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
stom-dicom2nifti = "stomcore.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create package and test `__init__.py` files**

Create `src/stomcore/__init__.py`:

```python
"""Core data layer for CBCT segmentation."""

__version__ = "0.1.0"
```

Create `tests/__init__.py` (empty file):

```python
```

- [ ] **Step 5: Write a smoke test**

Create `tests/test_smoke.py`:

```python
import stomcore


def test_version_is_exposed():
    assert stomcore.__version__ == "0.1.0"
```

- [ ] **Step 6: Install the package (editable) and run the smoke test**

Run:

```bash
cd /opt/almaz/test/Stom
python -m pip install -e ".[dev]"
python -m pytest tests/test_smoke.py -v
```

Expected: `1 passed`. (If `pip install` cannot reach the network, install `numpy`, `SimpleITK`, `pytest` from your local mirror first, then re-run `pip install -e . --no-build-isolation`.)

- [ ] **Step 7: Commit**

```bash
cd /opt/almaz/test/Stom
git add pyproject.toml .gitignore src/stomcore/__init__.py tests/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold stomcore package with smoke test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Geometry

`Geometry` holds the spatial metadata that every `Volume` and `SegmentationMask` must agree on: voxel spacing (mm), world origin, and the direction cosine matrix. It is immutable and toolkit-free.

**Conventions (read before coding):**
- `spacing`, `origin` are `(x, y, z)` 3-tuples of floats — matching SimpleITK's `GetSpacing()`/`GetOrigin()`.
- `direction` is a flat 9-tuple, row-major 3×3 cosine matrix — matching SimpleITK's `GetDirection()`.

**Files:**
- Create: `src/stomcore/geometry.py`
- Test: `tests/test_geometry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_geometry.py`:

```python
import pytest

from stomcore.geometry import Geometry

IDENTITY_DIR = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def test_identity_builds_expected_geometry():
    g = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    assert g.spacing == (0.3, 0.3, 0.3)
    assert g.origin == (0.0, 0.0, 0.0)
    assert g.direction == IDENTITY_DIR


def test_geometry_is_frozen():
    g = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    with pytest.raises(Exception):
        g.spacing = (1.0, 1.0, 1.0)


def test_compatible_when_within_tolerance():
    a = Geometry(spacing=(0.30000, 0.3, 0.3), origin=(0, 0, 0), direction=IDENTITY_DIR)
    b = Geometry(spacing=(0.30001, 0.3, 0.3), origin=(0, 0, 0), direction=IDENTITY_DIR)
    assert a.is_compatible(b, tol=1e-3) is True


def test_incompatible_when_spacing_differs_beyond_tolerance():
    a = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    b = Geometry.identity(spacing=(0.4, 0.3, 0.3))
    assert a.is_compatible(b) is False


def test_incompatible_when_origin_differs():
    a = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    b = Geometry(spacing=(0.3, 0.3, 0.3), origin=(5.0, 0.0, 0.0), direction=IDENTITY_DIR)
    assert a.is_compatible(b) is False


def test_rejects_wrong_length_direction():
    with pytest.raises(ValueError):
        Geometry(spacing=(0.3, 0.3, 0.3), origin=(0, 0, 0), direction=(1.0, 0.0, 0.0))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomcore.geometry'`

- [ ] **Step 3: Write the implementation**

Create `src/stomcore/geometry.py`:

```python
"""Spatial geometry shared by volumes and masks."""

from __future__ import annotations

import math
from dataclasses import dataclass

_IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Geometry:
    """Voxel-to-world mapping: spacing (mm), origin, direction cosines.

    spacing/origin are (x, y, z); direction is a row-major flat 3x3 matrix.
    """

    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.spacing) != 3:
            raise ValueError("spacing must have 3 components")
        if len(self.origin) != 3:
            raise ValueError("origin must have 3 components")
        if len(self.direction) != 9:
            raise ValueError("direction must have 9 components (flat 3x3)")

    @classmethod
    def identity(
        cls,
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "Geometry":
        return cls(spacing=spacing, origin=origin, direction=_IDENTITY_DIRECTION)

    def is_compatible(self, other: "Geometry", tol: float = 1e-4) -> bool:
        """True if spacing, origin and direction all match within tol."""
        return (
            _all_close(self.spacing, other.spacing, tol)
            and _all_close(self.origin, other.origin, tol)
            and _all_close(self.direction, other.direction, tol)
        )


def _all_close(a: tuple[float, ...], b: tuple[float, ...], tol: float) -> bool:
    return len(a) == len(b) and all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_geometry.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomcore/geometry.py tests/test_geometry.py
git commit -m "feat: add Geometry with compatibility check

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Volume

`Volume` couples a 3D voxel array with its `Geometry`. Voxels are stored `[z, y, x]` (NumPy axis order returned by `SimpleITK.GetArrayFromImage`), while `Geometry` tuples are `(x, y, z)` — this asymmetry is deliberate and matches SimpleITK; document it and never silently reorder.

**Files:**
- Create: `src/stomcore/volume.py`
- Test: `tests/test_volume.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_volume.py`:

```python
import numpy as np
import pytest

from stomcore.geometry import Geometry
from stomcore.volume import Volume


def _voxels(shape=(4, 3, 2)):
    return np.arange(np.prod(shape), dtype=np.int16).reshape(shape)


def test_volume_exposes_voxels_and_geometry():
    vox = _voxels()
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    vol = Volume(vox, geo)
    assert vol.geometry is geo
    np.testing.assert_array_equal(vol.voxels, vox)


def test_shape_is_zyx_voxel_shape():
    vol = Volume(_voxels((4, 3, 2)), Geometry.identity(spacing=(1.0, 1.0, 1.0)))
    assert vol.shape == (4, 3, 2)


def test_rejects_non_3d_voxels():
    geo = Geometry.identity(spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        Volume(np.zeros((4, 4)), geo)


def test_equality_compares_voxels_and_geometry():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    a = Volume(_voxels(), geo)
    b = Volume(_voxels(), geo)
    assert a == b


def test_inequality_when_voxels_differ():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    a = Volume(_voxels(), geo)
    diff = _voxels()
    diff[0, 0, 0] += 1
    b = Volume(diff, geo)
    assert a != b
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_volume.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomcore.volume'`

- [ ] **Step 3: Write the implementation**

Create `src/stomcore/volume.py`:

```python
"""3D voxel volume bound to a spatial Geometry."""

from __future__ import annotations

import numpy as np

from .geometry import Geometry


class Volume:
    """Immutable-by-convention 3D volume.

    voxels: NumPy array indexed [z, y, x] (SimpleITK array order).
    geometry: spacing/origin/direction in (x, y, z) order.
    """

    def __init__(self, voxels: np.ndarray, geometry: Geometry) -> None:
        if voxels.ndim != 3:
            raise ValueError(f"voxels must be 3D [z, y, x], got ndim={voxels.ndim}")
        self._voxels = voxels
        self._geometry = geometry

    @property
    def voxels(self) -> np.ndarray:
        return self._voxels

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self._voxels.shape)  # type: ignore[return-value]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Volume):
            return NotImplemented
        return self._geometry == other._geometry and np.array_equal(
            self._voxels, other._voxels
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_volume.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomcore/volume.py tests/test_volume.py
git commit -m "feat: add Volume wrapping voxels and Geometry

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: SegmentationMask and LabelInfo

A `SegmentationMask` is a multi-label integer volume (`[z, y, x]`) plus a `label_map` describing each label. Label `0` is always background and is not described in `label_map`. The mask carries its own `Geometry` and can check compatibility against a `Volume` (shape *and* geometry).

**Files:**
- Create: `src/stomcore/mask.py`
- Test: `tests/test_mask.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mask.py`:

```python
import numpy as np
import pytest

from stomcore.geometry import Geometry
from stomcore.mask import LabelInfo, SegmentationMask
from stomcore.volume import Volume


def _label_map():
    return {
        11: LabelInfo(label_id=11, name="tooth-11", color=(255, 0, 0)),
        12: LabelInfo(label_id=12, name="tooth-12", color=(0, 255, 0), visible=False),
    }


def _labels(shape=(4, 3, 2)):
    arr = np.zeros(shape, dtype=np.uint16)
    arr[0, 0, 0] = 11
    arr[1, 1, 1] = 12
    return arr


def test_labelinfo_defaults_to_visible():
    info = LabelInfo(label_id=11, name="tooth-11", color=(255, 0, 0))
    assert info.visible is True


def test_mask_exposes_geometry_and_label_map():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    mask = SegmentationMask(_labels(), geo, _label_map())
    assert mask.geometry is geo
    assert mask.label_map[12].visible is False


def test_rejects_non_3d_labels():
    geo = Geometry.identity(spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        SegmentationMask(np.zeros((4, 4), dtype=np.uint16), geo, {})


def test_present_labels_excludes_background():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    mask = SegmentationMask(_labels(), geo, _label_map())
    assert mask.present_labels() == {11, 12}


def test_compatible_with_volume_when_shape_and_geometry_match():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    vol = Volume(np.zeros((4, 3, 2), dtype=np.int16), geo)
    mask = SegmentationMask(_labels((4, 3, 2)), geo, _label_map())
    assert mask.is_compatible_with(vol) is True


def test_incompatible_with_volume_when_shape_differs():
    geo = Geometry.identity(spacing=(0.3, 0.3, 0.3))
    vol = Volume(np.zeros((4, 3, 2), dtype=np.int16), geo)
    mask = SegmentationMask(_labels((2, 3, 4)), geo, _label_map())
    assert mask.is_compatible_with(vol) is False


def test_incompatible_with_volume_when_geometry_differs():
    vol = Volume(np.zeros((4, 3, 2), dtype=np.int16), Geometry.identity(spacing=(0.3, 0.3, 0.3)))
    mask = SegmentationMask(_labels((4, 3, 2)), Geometry.identity(spacing=(0.4, 0.3, 0.3)), _label_map())
    assert mask.is_compatible_with(vol) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_mask.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomcore.mask'`

- [ ] **Step 3: Write the implementation**

Create `src/stomcore/mask.py`:

```python
"""Multi-label segmentation mask aligned to a Volume's geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Geometry
from .volume import Volume

BACKGROUND_LABEL = 0


@dataclass(frozen=True)
class LabelInfo:
    """Describes one label: its id, human name, RGB color and visibility."""

    label_id: int
    name: str
    color: tuple[int, int, int]
    visible: bool = True


class SegmentationMask:
    """Integer label volume [z, y, x] plus a label_id -> LabelInfo map.

    Label 0 is background and is never listed in label_map.
    """

    def __init__(
        self,
        labels: np.ndarray,
        geometry: Geometry,
        label_map: dict[int, LabelInfo],
    ) -> None:
        if labels.ndim != 3:
            raise ValueError(f"labels must be 3D [z, y, x], got ndim={labels.ndim}")
        self._labels = labels
        self._geometry = geometry
        self._label_map = dict(label_map)

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def label_map(self) -> dict[int, LabelInfo]:
        return self._label_map

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self._labels.shape)  # type: ignore[return-value]

    def present_labels(self) -> set[int]:
        """Set of label ids actually present in the volume, excluding background."""
        present = set(int(v) for v in np.unique(self._labels))
        present.discard(BACKGROUND_LABEL)
        return present

    def is_compatible_with(self, volume: Volume, tol: float = 1e-4) -> bool:
        """True if this mask matches the volume in shape and geometry."""
        return self.shape == volume.shape and self._geometry.is_compatible(
            volume.geometry, tol=tol
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_mask.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomcore/mask.py tests/test_mask.py
git commit -m "feat: add SegmentationMask and LabelInfo

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: SimpleITK interop

This is the *only* module that bridges our pure types and `SimpleITK.Image`. `nifti_io` and `dicom_loader` build on it, so the conversion logic lives in exactly one place.

**Files:**
- Create: `src/stomcore/sitk_interop.py`
- Test: `tests/test_sitk_interop.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sitk_interop.py`:

```python
import numpy as np
import SimpleITK as sitk

from stomcore.geometry import Geometry
from stomcore.sitk_interop import (
    geometry_from_sitk,
    sitk_from_volume,
    volume_from_sitk,
)
from stomcore.volume import Volume


def _sitk_image():
    arr = np.arange(4 * 3 * 2, dtype=np.int16).reshape(4, 3, 2)  # [z, y, x]
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.3, 0.4, 0.5))  # (x, y, z)
    img.SetOrigin((1.0, 2.0, 3.0))
    return img


def test_geometry_from_sitk_reads_spacing_origin_direction():
    geo = geometry_from_sitk(_sitk_image())
    assert geo.spacing == (0.3, 0.4, 0.5)
    assert geo.origin == (1.0, 2.0, 3.0)
    assert len(geo.direction) == 9


def test_volume_from_sitk_preserves_voxels_in_zyx():
    vol = volume_from_sitk(_sitk_image())
    assert vol.shape == (4, 3, 2)
    assert vol.voxels[0, 0, 0] == 0
    assert vol.voxels[3, 2, 1] == 23


def test_round_trip_volume_through_sitk():
    geo = Geometry(spacing=(0.3, 0.4, 0.5), origin=(1.0, 2.0, 3.0),
                   direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    vox = np.arange(4 * 3 * 2, dtype=np.int16).reshape(4, 3, 2)
    vol = Volume(vox, geo)
    restored = volume_from_sitk(sitk_from_volume(vol))
    assert restored == vol
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_sitk_interop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomcore.sitk_interop'`

- [ ] **Step 3: Write the implementation**

Create `src/stomcore/sitk_interop.py`:

```python
"""The single bridge between stomcore types and SimpleITK images."""

from __future__ import annotations

import SimpleITK as sitk

from .geometry import Geometry
from .volume import Volume


def geometry_from_sitk(image: sitk.Image) -> Geometry:
    return Geometry(
        spacing=tuple(float(v) for v in image.GetSpacing()),
        origin=tuple(float(v) for v in image.GetOrigin()),
        direction=tuple(float(v) for v in image.GetDirection()),
    )


def volume_from_sitk(image: sitk.Image) -> Volume:
    voxels = sitk.GetArrayFromImage(image)  # [z, y, x]
    return Volume(voxels, geometry_from_sitk(image))


def sitk_from_volume(volume: Volume) -> sitk.Image:
    image = sitk.GetImageFromArray(volume.voxels)  # consumes [z, y, x]
    image.SetSpacing(volume.geometry.spacing)
    image.SetOrigin(volume.geometry.origin)
    image.SetDirection(volume.geometry.direction)
    return image
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_sitk_interop.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomcore/sitk_interop.py tests/test_sitk_interop.py
git commit -m "feat: add SimpleITK interop bridge

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: NIfTI I/O

`save_volume_nifti` / `load_volume_nifti` are thin wrappers over `sitk_interop` + SimpleITK read/write, using `.nii.gz` (the client↔cloud exchange format from the spec). The key behavior to lock down is a **round-trip that preserves geometry and voxels**.

**Files:**
- Create: `src/stomcore/nifti_io.py`
- Test: `tests/test_nifti_io.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nifti_io.py`:

```python
import numpy as np

from stomcore.geometry import Geometry
from stomcore.nifti_io import load_volume_nifti, save_volume_nifti
from stomcore.volume import Volume


def _volume():
    geo = Geometry(spacing=(0.3, 0.4, 0.5), origin=(1.0, 2.0, 3.0),
                   direction=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    vox = np.arange(5 * 4 * 3, dtype=np.int16).reshape(5, 4, 3)
    return Volume(vox, geo)


def test_save_then_load_round_trips_volume(tmp_path):
    vol = _volume()
    path = tmp_path / "study.nii.gz"
    save_volume_nifti(vol, path)
    assert path.exists()
    restored = load_volume_nifti(path)
    assert restored.shape == vol.shape
    np.testing.assert_array_equal(restored.voxels, vol.voxels)
    assert restored.geometry.is_compatible(vol.geometry)


def test_accepts_string_paths(tmp_path):
    vol = _volume()
    path = str(tmp_path / "study.nii.gz")
    save_volume_nifti(vol, path)
    restored = load_volume_nifti(path)
    assert restored.shape == vol.shape
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_nifti_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomcore.nifti_io'`

- [ ] **Step 3: Write the implementation**

Create `src/stomcore/nifti_io.py`:

```python
"""Read/write Volume as NIfTI (.nii.gz) — the client<->cloud exchange format."""

from __future__ import annotations

import os

import SimpleITK as sitk

from .sitk_interop import sitk_from_volume, volume_from_sitk
from .volume import Volume


def save_volume_nifti(volume: Volume, path: str | os.PathLike) -> None:
    sitk.WriteImage(sitk_from_volume(volume), str(path), useCompression=True)


def load_volume_nifti(path: str | os.PathLike) -> Volume:
    return volume_from_sitk(sitk.ReadImage(str(path)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_nifti_io.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/stomcore/nifti_io.py tests/test_nifti_io.py
git commit -m "feat: add NIfTI save/load with geometry round-trip

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: DICOM loader

`DicomLoader.load(directory)` reads a single DICOM series via SimpleITK's GDCM reader and returns a `Volume`. It raises `DicomError` with a clear message for the failure modes from the spec: no series found, multiple ambiguous series, or a degenerate single-slice volume. Tests use a synthetic DICOM series written by a fixture (no patient data).

**Files:**
- Create: `src/stomcore/dicom_loader.py`
- Create/Modify: `tests/conftest.py`
- Test: `tests/test_dicom_loader.py`

- [ ] **Step 1: Add the synthetic-DICOM fixture**

Create `tests/conftest.py`:

```python
import os

import numpy as np
import pytest
import SimpleITK as sitk


def _write_dicom_series(directory, n_slices, rows=16, cols=16, spacing=(0.3, 0.3, 0.3)):
    """Write a minimal valid CT DICOM series into `directory`. Returns the dir path."""
    arr = (np.arange(n_slices * rows * cols).reshape(n_slices, rows, cols) % 1000).astype(np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)

    series_uid = "1.2.826.0.1.3680043.2.1125.1.1234567890"
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
        writer.SetFileName(os.path.join(directory, f"slice_{i:03d}.dcm"))
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
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_dicom_loader.py`:

```python
import pytest

from stomcore.dicom_loader import DicomError, DicomLoader
from stomcore.volume import Volume


def test_loads_series_into_volume(dicom_series):
    vol = DicomLoader.load(dicom_series)
    assert isinstance(vol, Volume)
    assert vol.shape == (8, 16, 16)  # [z, y, x]
    assert vol.geometry.spacing[0] == pytest.approx(0.3)


def test_raises_when_no_series_found(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DicomError, match="no DICOM series"):
        DicomLoader.load(str(empty))


def test_raises_on_single_slice_volume(single_slice_series):
    with pytest.raises(DicomError, match="too few slices"):
        DicomLoader.load(single_slice_series)


def test_raises_when_directory_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(DicomError, match="not a directory"):
        DicomLoader.load(str(missing))
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_dicom_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomcore.dicom_loader'`

- [ ] **Step 4: Write the implementation**

Create `src/stomcore/dicom_loader.py`:

```python
"""Load a single DICOM CBCT series into a Volume."""

from __future__ import annotations

import os

import SimpleITK as sitk

from .sitk_interop import volume_from_sitk
from .volume import Volume

MIN_SLICES = 2


class DicomError(Exception):
    """Raised when a DICOM directory cannot be loaded as a single CBCT series."""


class DicomLoader:
    @staticmethod
    def load(directory: str | os.PathLike) -> Volume:
        directory = str(directory)
        if not os.path.isdir(directory):
            raise DicomError(f"not a directory: {directory}")

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(directory)
        if not series_ids:
            raise DicomError(f"no DICOM series found in {directory}")
        if len(series_ids) > 1:
            raise DicomError(
                f"multiple DICOM series found ({len(series_ids)}); expected exactly one"
            )

        file_names = reader.GetGDCMSeriesFileNames(directory, series_ids[0])
        if len(file_names) < MIN_SLICES:
            raise DicomError(
                f"too few slices ({len(file_names)}); expected a 3D volume"
            )

        reader.SetFileNames(file_names)
        image = reader.Execute()
        return volume_from_sitk(image)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_dicom_loader.py -v`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add src/stomcore/dicom_loader.py tests/conftest.py tests/test_dicom_loader.py
git commit -m "feat: add DicomLoader with validation and synthetic-series fixtures

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: CLI (`stom-dicom2nifti`)

A thin command-line entry that loads a DICOM directory and writes a `.nii.gz`. It only parses args and orchestrates `DicomLoader` + `save_volume_nifti`, mapping `DicomError` to a non-zero exit with a readable message.

**Files:**
- Create: `src/stomcore/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
from stomcore.cli import main
from stomcore.nifti_io import load_volume_nifti


def test_cli_converts_dicom_to_nifti(dicom_series, tmp_path, capsys):
    out = tmp_path / "out.nii.gz"
    code = main([dicom_series, str(out)])
    assert code == 0
    assert out.exists()
    vol = load_volume_nifti(out)
    assert vol.shape == (8, 16, 16)
    assert "out.nii.gz" in capsys.readouterr().out


def test_cli_reports_error_on_empty_dir(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out.nii.gz"
    code = main([str(empty), str(out)])
    assert code == 1
    assert not out.exists()
    assert "error" in capsys.readouterr().err.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stomcore.cli'`

- [ ] **Step 3: Write the implementation**

Create `src/stomcore/cli.py`:

```python
"""`stom-dicom2nifti` — convert a DICOM CBCT series to a NIfTI file."""

from __future__ import annotations

import argparse
import sys

from .dicom_loader import DicomError, DicomLoader
from .nifti_io import save_volume_nifti


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stom-dicom2nifti",
        description="Convert a DICOM CBCT series directory to a NIfTI (.nii.gz) file.",
    )
    parser.add_argument("dicom_dir", help="directory containing one DICOM series")
    parser.add_argument("output", help="output path, e.g. study.nii.gz")
    args = parser.parse_args(argv)

    try:
        volume = DicomLoader.load(args.dicom_dir)
    except DicomError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    save_volume_nifti(volume, args.output)
    print(f"saved volume {volume.shape} -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_cli.py -v`
Expected: `2 passed`

- [ ] **Step 5: Verify the installed console script works end-to-end**

Run (uses the fixture writer inline to avoid needing real data):

```bash
python -c "
import tempfile, os
from tests.conftest import _write_dicom_series
d = tempfile.mkdtemp()
_write_dicom_series(d, n_slices=8)
from stomcore.cli import main
raise SystemExit(main([d, os.path.join(tempfile.mkdtemp(), 'out.nii.gz')]))
"
```

Expected: prints `saved volume (8, 16, 16) -> .../out.nii.gz` and exits 0.

- [ ] **Step 6: Commit**

```bash
git add src/stomcore/cli.py tests/test_cli.py
git commit -m "feat: add stom-dicom2nifti CLI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Public API surface + full-suite green

Expose the core types from the package root so downstream plans (backend, client) import from one place, and confirm the whole suite passes together.

**Files:**
- Modify: `src/stomcore/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Extend the smoke test for public re-exports**

Replace `tests/test_smoke.py` with:

```python
import stomcore


def test_version_is_exposed():
    assert stomcore.__version__ == "0.1.0"


def test_public_types_are_reexported():
    from stomcore import (
        DicomError,
        DicomLoader,
        Geometry,
        LabelInfo,
        SegmentationMask,
        Volume,
        load_volume_nifti,
        save_volume_nifti,
    )

    assert all(
        obj is not None
        for obj in (
            DicomError,
            DicomLoader,
            Geometry,
            LabelInfo,
            SegmentationMask,
            Volume,
            load_volume_nifti,
            save_volume_nifti,
        )
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: FAIL with `ImportError: cannot import name 'Geometry' from 'stomcore'`

- [ ] **Step 3: Update `__init__.py` to re-export the public API**

Replace `src/stomcore/__init__.py` with:

```python
"""Core data layer for CBCT segmentation."""

from .dicom_loader import DicomError, DicomLoader
from .geometry import Geometry
from .mask import LabelInfo, SegmentationMask
from .nifti_io import load_volume_nifti, save_volume_nifti
from .volume import Volume

__version__ = "0.1.0"

__all__ = [
    "DicomError",
    "DicomLoader",
    "Geometry",
    "LabelInfo",
    "SegmentationMask",
    "Volume",
    "load_volume_nifti",
    "save_volume_nifti",
    "__version__",
]
```

- [ ] **Step 4: Run the FULL suite**

Run: `python -m pytest -v`
Expected: all tests pass (`test_smoke`, `test_geometry`, `test_volume`, `test_mask`, `test_sitk_interop`, `test_nifti_io`, `test_dicom_loader`, `test_cli`) — `25 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/stomcore/__init__.py tests/test_smoke.py
git commit -m "feat: re-export public stomcore API from package root

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review (completed by plan author)

**Spec coverage (against §4 data layer + §5 exchange format):**
- `Volume` → Task 3. `SegmentationMask`/`LabelInfo` → Task 4. `DicomLoader` (+ CBCT/integrity validation) → Task 7. NIfTI exchange format → Task 6. Geometric-invariance check (mask must match volume) → `Geometry.is_compatible` (Task 2) + `SegmentationMask.is_compatible_with` (Task 4). DICOM→NIfTI client-side conversion → CLI (Task 8). All covered.
- Out of scope for this plan (correctly deferred to later plans): VTK views, MaskEditor, MeasurementTool, CloudClient, backend, worker. This plan delivers the shared foundation all three subsystems consume.

**Placeholder scan:** No TBD/TODO/"handle edge cases" — every step has runnable code or an exact command + expected output.

**Type consistency:** `Geometry(spacing, origin, direction)` constructor and `.is_compatible(other, tol)` used identically across Tasks 2/4/5/6. `Volume(voxels, geometry)`, `.voxels/.geometry/.shape` consistent in Tasks 3/5/6/7. `SegmentationMask(labels, geometry, label_map)`, `.present_labels()`, `.is_compatible_with(volume, tol)` consistent in Task 4. `volume_from_sitk`/`sitk_from_volume`/`geometry_from_sitk` names identical in Tasks 5/6/7. `DicomLoader.load` / `DicomError` consistent in Tasks 7/8/9. `save_volume_nifti`/`load_volume_nifti` consistent in Tasks 6/8/9.
