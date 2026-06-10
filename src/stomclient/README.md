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
