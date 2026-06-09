# Follow-ups from Core Data Layer (Plan 1)

These were surfaced by the final holistic review of `stomcore` and **deliberately deferred** out of Plan 1's scope. They must be picked up in the next plan(s).

## For the Backend/Worker plan (Plan 2) — most urgent
- **SegmentationMask NIfTI I/O.** Add `save_mask_nifti` / `load_mask_nifti` so masks can be exchanged as `.nii.gz` per spec §5. The label array round-trips like a Volume, but the `label_map` (`label_id → {name, color, visible}`) needs a defined sidecar format (e.g. a JSON file alongside the `.nii.gz`). The worker returns masks; the client loads them — both need this.
- **Mask round-trip test** with a non-identity direction matrix (the riskiest axis; currently only `sitk_interop` covers non-identity direction).

## For the DICOM loader hardening (Plan 2 or a dedicated task)
- **Real CBCT / integrity validation** (spec §4, §6). Current loader only checks: directory exists, exactly one series, ≥2 slices. It does NOT verify modality is CT/CBCT, nor detect non-uniform spacing / missing slices (ITK only warns and loads anyway). Spec promises "не CBCT → понятная ошибка" — under-delivered. Add: modality tag check, slice-spacing uniformity check.

## Minor / consistency (pick up opportunistically)
- **Immutability is convention only.** `Volume.voxels` / `SegmentationMask.labels` return the live array; constructors don't copy. Either copy on construction, set `array.flags.writeable = False`, or document explicitly that callers must not mutate.
- **Equality/hash policy inconsistent.** `Geometry`/`LabelInfo` are frozen+hashable; `Volume` has `__eq__` but no `__hash__` (unhashable); `SegmentationMask` has neither. Choose one policy across the value types.
- **`SegmentationMask` coherence validation.** Optionally validate that `label_map` keys match `LabelInfo.label_id` and that background (0) is not described.
- `Volume.__eq__` is typed `-> bool` but can return `NotImplemented` (cosmetic).

## Done in Plan 1 (for reference)
- Fixed during review: SimpleITK `RuntimeError` from `GetGDCMSeriesIDs`/`Execute` is now wrapped as `DicomError` so the CLI fails cleanly (spec §6 "не падает"). Commit `31d3032`.
