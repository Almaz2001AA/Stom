# stomserver

Cloud backend for CBCT segmentation. Depends on `stomcore`.

## Run (dev)

Install: `pip install -e ".[dev,server]"` (and `".[nnunet]"` for real inference).

1. Create an account + token:
   `python scripts/create_account.py "Clinic A"`  → prints the token once.
2. Download model weights (once):
   `python scripts/download_weights.py`  → into `STOM_MODEL_DIR` (default `./models`).
3. Start Redis (native): `redis-server` (default `redis://localhost:6379/0`).
4. Start the API: `uvicorn "stomserver.api.app:create_app" --factory --reload`.
5. Start a worker: `rq worker segmentation` (with `STOM_*` env vars set).

## Config (env vars)

- `STOM_DB_URL` (default `sqlite:///stom.db`)
- `STOM_STORAGE_DIR` (default `./storage`)
- `STOM_REDIS_URL` (default `redis://localhost:6379/0`)
- `STOM_MODEL_DIR` (default `./models`)
- `STOM_MAX_UPLOAD_BYTES` (default 500 MB)

## API

- `POST /studies` (multipart `.nii.gz`) → `{study_id, shape, spacing}`
- `POST /studies/{id}/segment` → `{job_id, status}`
- `GET /jobs/{id}` → `{job_id, status, error?}`
- `GET /studies/{id}/masks` → `mask.nii.gz`
- `GET /studies/{id}/masks/labels` → `mask_labels.json`
- `GET /healthz`

All except `/healthz` require `Authorization: Bearer <token>`.
