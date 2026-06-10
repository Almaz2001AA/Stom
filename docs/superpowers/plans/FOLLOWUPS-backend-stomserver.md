# Follow-ups from Backend `stomserver` (Plan 2)

Surfaced by code reviews during Plan 2, **deliberately deferred** out of scope. Pick up in later plans / before the real model goes live.

## Before the real model (DentalSegmentator) goes live
- ✅ **DONE (2026-06-10).** ~~Verify `DENTALSEGMENTATOR_LABELS` against the model's `dataset.json`.~~ Weights downloaded (Zenodo 10829675); `dataset.json` labels are 1 Upper Skull / 2 Mandible / 3 Upper Teeth / 4 Lower Teeth / 5 Mandibular canal — ids matched, names updated to the official ones in `labels.py`.
- ✅ **DONE (2026-06-10).** ~~Verify the nnU-Net inference call.~~ Real weights ship a single `fold_0` (not `all`), so `runner.py` now uses `use_folds=(0,)`; `checkpoint_name="checkpoint_final.pth"` and output `case.nii.gz` confirmed correct. The model dir is `.../Dataset112_DentalSegmentator_v100/nnUNetTrainer__nnUNetPlans__3d_fullres`. Slow `test_runner_real` passes on CPU (~33s) and a full upload→segment→download e2e succeeded against the live server.
- ✅ **DONE (2026-06-10).** ~~Strengthen the geometry invariant.~~ `predict` now returns `(labels, geometry)`; the worker builds the mask from the *predicted* geometry, so `is_compatible_with(volume)` genuinely checks shape **and** geometry. `FakeRunner` returns the input geometry; `DentalSegmentatorRunner` returns `geometry_from_sitk(result)`. Test: `test_worker_marks_failed_on_geometry_drift`.
- **Checksum in `download_weights.py`.** Spec §6 calls for a checksum verification; the script currently only downloads + unzips. Add the Zenodo file's checksum once known.

## Reliability / robustness
- ✅ **DONE (2026-06-10).** ~~Stuck `running` jobs on worker death.~~ Two layers: (1) `reap_stale_jobs(session_factory, timeout_seconds)` in `worker.py` fails jobs stuck in `running` past `STOM_JOB_TIMEOUT_SECONDS` (default 1h) — covers OOM/SIGKILL where no in-process handler runs; exposed as `scripts/reap_stale_jobs.py` for cron. (2) RQ `on_failure=handle_job_failure` wired in `RqJobQueue.enqueue_segmentation` marks the DB job failed when the work-horse dies. Tests: `test_reap_stale_running_jobs_marks_only_old_running`, `test_mark_job_failed_*`, queue `failure_callback` assertion.
- ✅ **DONE (2026-06-10).** ~~Orphan `queued` job if Redis down.~~ `segment_study` now rolls back (deletes) the `Job` row on enqueue failure before returning 503, so no `queued` row lingers. Test: `test_segment_enqueue_failure_leaves_no_orphan_job`. (Kept commit-then-enqueue ordering so a real worker never races ahead of an uncommitted job.)
- **Wire-level upload size limit.** `max_upload_bytes` is now enforced in the handler (413), but Starlette may spool the body to a temp file before the handler runs. For hardened deploys, add an ASGI/proxy-level body-size limit.
- **Streaming upload + orphaned storage cleanup.** Uploads are read into memory; large studies should stream. Worker partial-failure can leave a mask blob with no DB pointer — add cleanup.

## Productionization (spec §9)
- `S3Storage` implementation behind the existing `Storage` interface; migrate dev SQLite → PostgreSQL (Alembic migrations).
- Full accounts/login/roles (separate auth sub-project) — current MVP is static bearer tokens.
- Per-FDI tooth numbering (train on ToothFairy2, needs GPU).
- GPU inference + worker autoscaling under load.
- Legal review of DentalSegmentator/nnU-Net CC-BY attribution before commercial release.

## Minor polish
- ✅ **DONE (2026-06-10).** ~~`errors.py` validation handler discards `exc.errors()`~~ — now returns a compact, non-leaky `"validation error: <loc>: <msg>; …"` summary (`_summarize_validation`, capped at 5). Test: `test_missing_file_returns_validation_summary`.
- `code` field in error responses equals the HTTP status int — decide if a stable machine-readable code string is wanted before clients depend on it.
- ✅ **DONE (2026-06-10).** ~~Add `WWW-Authenticate: Bearer` header on 401s.~~ `auth.py` 401s carry the header; the HTTP error handler now propagates `exc.headers`. Tests: `test_upload_requires_auth`, `test_invalid_token_advertises_bearer`.
- ✅ **DONE (2026-06-10).** ~~`auth.py` imports placed after `hash_token` (E402)~~ — imports moved to module top.
