# ToothFairy2 — GPU retraining pipeline

Retrain the per-tooth (FDI) ToothFairy2 model to push segmentation quality toward
the challenge **winning recipe** (mean Dice **0.9253** vs the default we ship),
then drop the result back into the Stom engine. No manual annotation needed — we
use the **public** ToothFairy2 dataset (CC BY-SA, commercial use OK; cite the
papers in the repo `NOTICE`).

Why retrain: the winning team's weights were **never published** — only the
recipe. Our shipped model is the *default* `nnUNetResEncUNetLPlans` ResEnc-L. The
winning model adds: the **torchres** planner, the **`onlyMirror01`** augmentation,
**1500 epochs**, a **larger patch**, and a 2-model ensemble. Stock `nnunetv2`
**2.8.0 already ships the exact trainer and planner**, so this is reproducible
with standard commands — no custom code.

> Pin **`nnunetv2==2.8.0`** for training. The Stom engine ships 2.8.0; a model
> trained on a different major version may not load. (`pip show nnunetv2` in this
> repo's `.venv` confirms the shipped version.)

---

## 0. What you need

- **A CUDA GPU.** Tier 1 fits **24 GB** (RTX 3090/4090, A5000). Tier 2 (winning
  large patch) wants **40–80 GB** (A100). Rent one (RunPod / Lambda / Vast.ai).
- ~**60 GB disk** (26 GB dataset + preprocessed + checkpoints).
- The dataset archive **`ToothFairy2_Dataset.zip`** (~26 GB, 480 image/label
  pairs). It is gitignored — copy it to the GPU box (it is in this project's
  `tf2_data/` on the dev machine). Source/login: see memory `toothfairy2-weights-license`.

**Time / cost (single GPU):** ~2 min/epoch × 1500 ≈ **2–3 days** on an A100;
roughly **\$50–150** at \$1–2/h. Tier 1 on a 4090 is comparable.

---

## 1. Environment (on the GPU box)

```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match the box's CUDA
pip install nnunetv2==2.8.0

# nnU-Net needs three directories; export them in every shell:
export nnUNet_raw=$PWD/nnUNet_raw
export nnUNet_preprocessed=$PWD/nnUNet_preprocessed
export nnUNet_results=$PWD/nnUNet_results
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
```

## 2. Data

```bash
unzip ToothFairy2_Dataset.zip -d "$nnUNet_raw"
# -> $nnUNet_raw/Dataset112_ToothFairy2/{dataset.json, imagesTr/*_0000.mha, labelsTr/*.mha}

# sanity-check: should be ~480 pairs and a dataset.json with channel_names + numTraining
ls "$nnUNet_raw/Dataset112_ToothFairy2/imagesTr" | wc -l
python -c "import json; d=json.load(open('$nnUNet_raw/Dataset112_ToothFairy2/dataset.json')); print(d['channel_names'], 'numTraining=', d['numTraining'], 'labels=', len(d['labels']))"
```

If `numTraining`/`channel_names` are missing (some mirrors ship a thin
`dataset.json`), fix them: `channel_names = {"0": "CBCT"}`, `numTraining =`
(number of pairs), `file_ending = ".mha"`, and the `labels` map from the dataset
page. The label set must match what the engine expects (FDI ids).

## 3. Plan + preprocess

```bash
nnUNetv2_extract_fingerprint -d 112 -np 8

# Tier 1 (recommended, 24 GB, best fit for our CPU/low-RAM deployment):
nnUNetv2_plan_experiment   -d 112 -pl nnUNetPlannerResEncL_torchres
nnUNetv2_preprocess        -d 112 -c 3d_fullres -plans_name nnUNetResEncUNetLPlans_torchres -np 8
```

**Tier 2 (winning leaderboard config, A100 40–80 GB).** Let the planner derive a
larger patch for the bigger VRAM budget (robust — the planner keeps the network
architecture valid for the patch, unlike hand-editing it):

```bash
nnUNetv2_plan_experiment -d 112 -pl nnUNetPlannerResEncL_torchres \
    -gpu_memory_target 40 -overwrite_plans_name nnUNetResEncUNetLPlans_torchres_40g
nnUNetv2_preprocess -d 112 -c 3d_fullres -plans_name nnUNetResEncUNetLPlans_torchres_40g -np 8
```

> The exact winning config is a hand-added `3d_fullres_torchres_ps160x320x320_bs2`
> (patch 160×320×320, bs 2, 7 stages, features `[32,64,128,256,320,320,320]`).
> See the upstream recipe and inference script:
> https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/competitions/Toothfairy2/readme.md
> The `-gpu_memory_target` route above captures most of the large-patch benefit
> without risking a mis-specified architecture.

## 4. Train

The winning augmentation + epoch count is the built-in trainer
**`nnUNetTrainer_onlyMirror01_1500ep`**. Train the `all` fold (all cases, like the
winners):

```bash
# Tier 1:
nnUNetv2_train 112 3d_fullres all -p nnUNetResEncUNetLPlans_torchres \
    -tr nnUNetTrainer_onlyMirror01_1500ep

# Tier 2: same, but -p nnUNetResEncUNetLPlans_torchres_40g
```

Resume after a preemption: append `--c`. Watch
`$nnUNet_results/Dataset112_ToothFairy2/…/fold_all/progress.png`.

**Optional 2-model ensemble (the winners' last +Dice, GPU-side only).** Train a
second `all` run into a different results dir (e.g. set `nnUNet_results` to a
second path, or copy `fold_all`→`fold_0`,`fold_1`) and let nnU-Net average them at
inference. Note: **our CPU engine runs a single model** (no ensemble inference),
so for shipping, export one model (below). The ensemble only helps if you also run
a server-side GPU inference path.

## 5. Export into the Stom engine

Bring the trained model into the flat layout the engine loads:

```bash
python export_for_engine.py \
    --results-dir "$nnUNet_results/Dataset112_ToothFairy2/nnUNetTrainer_onlyMirror01_1500ep__nnUNetResEncUNetLPlans_torchres__3d_fullres" \
    --fold all --checkpoint checkpoint_final.pth \
    --out ./engine_model --verify
```

This writes `engine_model/Dataset112_ToothFairy2/{dataset.json, plans.json,
fold_0/checkpoint_best.pth}` — exactly what `ToothFairy2Runner` expects
(`use_folds=(0,)`, `checkpoint_name="checkpoint_best.pth"`; the engine reads the
config name from the checkpoint, so no folder-name encoding is needed). `--verify`
loads it with nnU-Net 2.8.0 to catch incompatibilities before shipping.

## 6. Ship it (decoupled engine release)

```bash
cd engine_model && zip -r Dataset112_ToothFairy2.zip Dataset112_ToothFairy2   # top-level = Dataset112_ToothFairy2/
```

1. Upload the zip to a public URL (GitHub release asset or Zenodo).
2. Set the repo **variable `TF2_WEIGHTS_URL`** to that URL (replaces the old one).
3. **Actions → build-engine-pack → Run workflow**, `version` = e.g. `v0.5.0`.
   It rebuilds the engine-pack with the new model and updates the fixed `engine`
   prerelease in place. Users get the better model on the next engine update.
   (Client `v*` releases are unaffected — see the engine/client decoupling.)

---

## Notes & gotchas

- **VRAM**: if Tier 1 OOMs, the planner already targets 24 GB; drop to a smaller
  GPU only by lowering `-gpu_memory_target` (smaller patch, slightly lower Dice).
- **Same nnU-Net version both sides** — train and serve on `nnunetv2==2.8.0`.
- **Inference cost on the user's PC** is set by the model's patch/spacing, not by
  training length. Tier 1 (24 GB patch) keeps inference close to the current model
  — that is why it is the recommended tier for our 6–8 GB CPU target. Tier 2's
  larger patch raises inference RAM/time too; validate on a target PC before
  shipping it.
- **Post-processing** (per-class phantom cutoff, denoise) lives in the engine and
  applies to any model — no retrain needed to tune it.
- **License**: CC BY-SA 4.0. Keep the `NOTICE` (3 citations) with any shipped
  weights. Do **not** use ToothFairy3 data here (CC BY-NC-SA, non-commercial).
