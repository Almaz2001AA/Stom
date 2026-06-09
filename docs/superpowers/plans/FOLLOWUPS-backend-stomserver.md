# Follow-ups from Backend `stomserver` (Plan 2)

Surfaced by code reviews during Plan 2, **deliberately deferred** out of scope. Pick up in later plans / before the real model goes live.

## Before the real model (DentalSegmentator) goes live
- **Verify `DENTALSEGMENTATOR_LABELS` against the model's `dataset.json`.** Label ids/names/colors in `src/stomserver/segmentation/labels.py` are placeholders. After `download_weights.py` runs, inspect the extracted `dataset.json` and correct ids/names if they differ. (Self-documented TODO in `labels.py`.)
- **Verify the nnU-Net inference call.** `DentalSegmentatorRunner.predict` (`runner.py`) uses assumed `use_folds=("all",)`, `checkpoint_name="checkpoint_final.pth"`, and output filename `case.nii.gz`. Confirm against the actual extracted weights folder layout; adjust as needed. Run the `@pytest.mark.slow` `test_runner_real` with weights present.
- **Strengthen the geometry invariant.** `runner.predict` returns only `GetArrayFromImage(...)`, discarding the predicted image's geometry, so the worker's `mask.is_compatible_with(volume)` effectively checks only shape. Have `predict` return/verify the predicted geometry, or assert nnU-Net preserved the input geometry, to fully honor spec §4.
- **Checksum in `download_weights.py`.** Spec §6 calls for a checksum verification; the script currently only downloads + unzips. Add the Zenodo file's checksum once known.

## Reliability / robustness
- **Stuck `running` jobs on worker death.** If the worker process is OOM-killed/SIGKILLed mid-inference, the job stays `running` forever (no heartbeat/timeout). Add an RQ `on_failure` handler and/or a stale-`running` reaper.
- **Orphan `queued` job if Redis down.** `segment_study` commits the `Job` then enqueues; on enqueue failure it returns 503 but the `queued` row persists and never runs. Consider enqueue-then-commit, or rollback the job on enqueue failure.
- **Wire-level upload size limit.** `max_upload_bytes` is now enforced in the handler (413), but Starlette may spool the body to a temp file before the handler runs. For hardened deploys, add an ASGI/proxy-level body-size limit.
- **Streaming upload + orphaned storage cleanup.** Uploads are read into memory; large studies should stream. Worker partial-failure can leave a mask blob with no DB pointer — add cleanup.

## Productionization (spec §9)
- `S3Storage` implementation behind the existing `Storage` interface; migrate dev SQLite → PostgreSQL (Alembic migrations).
- Full accounts/login/roles (separate auth sub-project) — current MVP is static bearer tokens.
- Per-FDI tooth numbering (train on ToothFairy2, needs GPU).
- GPU inference + worker autoscaling under load.
- Legal review of DentalSegmentator/nnU-Net CC-BY attribution before commercial release.

## Minor polish
- `errors.py` validation handler discards `exc.errors()` detail; consider including a compact summary.
- `code` field in error responses equals the HTTP status int — decide if a stable machine-readable code string is wanted before clients depend on it.
- Add `WWW-Authenticate: Bearer` header on 401s.
- `auth.py` imports placed after `hash_token` (PEP8 E402) due to append-only construction; tidy if desired.
