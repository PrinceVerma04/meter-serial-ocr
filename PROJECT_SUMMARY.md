# Meter Serial-Number OCR — Complete Project Summary

**Goal:** Extract the alphanumeric serial number printed on electricity-meter
nameplates from already-cropped photos.

**Approach:** A **fine-tuned PaddleOCR PP-OCRv5 recognizer**, run **recognition-only**
(no text detector — each crop is already the serial ROI), with a **confidence gate**
that auto-accepts high-confidence reads and routes the rest to human review.

**Last updated:** 2026-07-15

---

## 0. Models & Quantization (quick reference)

A clean index of every model lives in **`MODELS/`** (clearly-named links → real paths).

**Naming scheme:** `meterocr_<version>_<precision>_<method>_<size>[.runtime]`
`v1`=1,144-img fine-tune · `v2`=6,709-img retrain · `fp32/int8`=precision ·
`original/ptq/qat/old`=method · size · `.onnx`/`paddle`=runtime.

**Shape (all models):** input `[N, 3, 48, W]` → output `[N, T, 18385]`
(N=batch, 3=RGB, 48=height, W=width≈320, T=time-steps, 18385=PP-OCRv5 char classes).

| Model (MODELS/) | Real path | Size | Precision | Method | Accuracy* | Runtime |
|---|---|---|---|---|---|---|
| `meterocr_v2_fp32_original_81mb` | `finetune/inference_v2/` | **81 MB** | FP32 | none | **94.7%** | Paddle |
| `meterocr_v2_int8_qat_22mb_paddle` ⭐ | `size_reduction/prune_qat/qat_inference/` | **22 MB** | INT8 | **QAT** | **92.8%** | Paddle |
| `meterocr_v2_int8_ptq_21mb.onnx` | `size_reduction/models/server_int8.onnx` | **21 MB** | INT8 | **PTQ** | 85.3% | ONNX |
| `meterocr_v1_fp32_old_81mb` | `finetune/inference/` | 81 MB | FP32 | none | ~87% (val) | Paddle |

*Accuracy = exact-match on the **414 readable** images of a 500-image TCPODL/TPSODL
sample (86 all-model-fail crops excluded). All 3 judged on the SAME images — see the
measured comparison below. On in-domain TPWODL data all three run higher (~95%+).

### Measured 3-model comparison (2026-07-15, 500 imgs, human-judged)
| Model | Size | On all 500 | On 414 readable | vs FP32 |
|---|---|---:|---:|---:|
| FP32 original | 81 MB | 78.4% | **94.7%** | — |
| **QAT INT8** ⭐ | 22 MB | 76.8% | **92.8%** | **−1.9 pp** |
| PTQ INT8 | 21 MB | 70.6% | **85.3%** | **−9.4 pp** |

**Conclusion: QAT ≫ PTQ.** For the same 4× size cut, QAT loses only ~2 pts vs the
original; PTQ loses ~9 pts. 86/500 (17%) crops are unreadable by *every* model
(input-quality limit, not a quantization issue).

**Quantization methodology**
- **PTQ (Post-Training Quantization)** → `ptq` model: convert Paddle→ONNX, fold constants
  (`quant_pre_process`), then **onnxruntime static INT8** calibrated on ~150 real crops
  (per-tensor, QOperator). Fast, no retraining, costs ~3–4 accuracy points.
- **QAT (Quantization-Aware Training)** → `qat` model: PaddleSlim inserts fake-quant on
  Conv2D+Linear layers, **fine-tune 10 epochs** on the 6,709-label dataset, export INT8.
  Recovers nearly all accuracy (0.91 vs FP32 0.917).
- Both needed an isolated **Paddle 3.0.0 venv** — the installed Paddle 3.3.1 exports a
  PIR-v4 format that `paddle2onnx` 2.1.0 can't convert (3.0.0 exports the convertible v1).
- **Result: 81 MB → 21–22 MB (4× smaller).** QAT→ONNX specifically is blocked by the old
  converter, so the QAT INT8 model deploys via Paddle; the PTQ INT8 model deploys via ONNX.

**Which to use:** best accuracy → `meterocr_v2_fp32_original_81mb` (81 MB); best small +
Paddle → `meterocr_v2_int8_qat_22mb_paddle` (22 MB); small + ONNX → the `ptq` model (21 MB).

**Deployment package (chosen: PTQ INT8 for ONNX portability):** `deploy/ptq_int8/`
— self-contained, ONNX-Runtime-only (no Paddle), bundles model + dict + `meterocr.py`
(preprocess + CTC decode + 0.95 confidence gate). ~21 MB. Verified standalone. Ship as-is.

---

## 1. Datasets (image counts)

| Folder | Images | Utility / type | Role |
|---|---:|---|---|
| `data/serial_roi_test` | **4,184** | TPWODL (LW / LM / WLT / TWSP prefixes) | Main labeled set — source of training + eval |
| `data/serialno_dev` | **501** | TPWODL | Dev slice — a **subset of** `serial_roi_test` (not separate data) |
| `data/tpddl_test` | **3,460** | TPDDL / Tata Power Delhi (pure-digit + SS prefixes) | Second utility — **evaluated only, never trained on** |
| `dataset for test/ROI` | **4,286** | TPWODL-family raw crops | Fresh blind set — **tested this session** |

**Distinct image pools = 3** (dev is inside serial_roi_test): TPWODL test, TPDDL, and ROI.
The 4 folders share the filename pattern `<sessionPrefix>_null_bigImg_<timestamp>.jpg`,
where the prefix is a **capture batch/agent, not a meter ID** → 1 image = 1 distinct serial
(no per-meter voting).

---

## 2. Fine-tuning (which dataset, how many images)

| Item | Value |
|---|---|
| Base model | PP-OCRv5 **server recognizer** (SVTR_HGNet, CTC+NRTR multi-head) |
| Mode | Recognition-only (no detector) |
| **Trained on** | **TPWODL only** — labeled crops from `data/serial_roi_test` |
| **Images used** | **1,144 labeled crops** |
| Split (by meter session, no leakage) | **773 train / 175 val / 196 test** |
| Learning rate | **2e-5** (3e-4 was tested and *hurt*) |
| Epochs | 20 (best at epoch 9) |
| Result | Held-out test **85.2% → 87.76%** (+2.55pp); fixed faint-prefix errors (LW→LN/LA/LH) |
| Not trained on | `data/tpddl_test` (0 images) and `dataset for test/ROI` (0 images) |

**Label bank:** `results/ground_truth.csv` = **4,148 labels / 3,387 readable** (TPWODL).
The deployed model was trained on the *earlier* 1,144-label subset; the remaining
~2,243 readable labels are available for a larger retrain but **not yet used**.

---

## 2A. Before vs After Fine-Tuning (dataset · model · accuracy)

Apples-to-apples: **same model family, same held-out test set** — only the weights changed.

| | **BEFORE fine-tuning** | **AFTER fine-tuning** |
|---|---|---|
| **Model** | PP-OCRv5 server recognizer — **stock/off-the-shelf weights** | PP-OCRv5 server recognizer — **fine-tuned on our labels** |
| **Mode** | Recognition-only | Recognition-only |
| **Trained on** | nothing (pretrained as shipped) | **1,144 TPWODL crops** (773 train / 175 val) |
| **Tested on** | 196 held-out TPWODL crops (session-split, no leakage) | **same 196 held-out crops** |
| **Exact-match accuracy** | **85.20%** | **87.76%** |
| **Improvement** | — | **+2.55 pp** |
| **Model size** | ~82 MB | ~81 MB (same architecture) |

- **Dataset (both stages):** TPWODL crops from `data/serial_roi_test`, 1,144 labeled
  images split **773 / 175 / 196** by meter session so the same meter never appears in
  both train and test (honest eval).
- **What fine-tuning fixed:** faint letter-prefix misreads (e.g. `LW → LN / LA / LH`) became
  correct, with **no regression** on already-clean crops. Best validation accuracy during
  training was **0.874 at epoch 9**.
- **Note on the 95.1% figure:** the "95.1% on unseen" number quoted for the deployed model
  is a *different, broader* measurement — NOT the same 196-image experiment. The controlled
  before/after is **85.2% → 87.76%**.
- **Scope:** this experiment was **TPWODL only**. TPDDL and ROI were never part of fine-tuning;
  they were tested later with the fine-tuned model only.

### How the fine-tuning was actually done (step by step)

1. **Collected labels** — hand-labeled TPWODL crops via the browser review/labeling page →
   `results/ground_truth.csv` (`image, true_label, verdict`).
2. **Built the dataset** — `scripts/prepare_finetune_data.py` converted the readable labels
   into a PaddleX text-recognition dataset under `finetune/dataset/` (`images/`, `train.txt`,
   `val.txt`, `test.txt`), **splitting by meter-session prefix** (70/15/15) to prevent
   train/test leakage. Label line format: `images/<file>\t<SERIAL>`.
3. **Config** — `finetune/rec_finetune.yml`: base = `PP-OCRv5_server_rec`, architecture
   SVTR_HGNet with a CTC + NRTR multi-head, `use_space_char: true`, `max_text_length: 25`,
   image shape 3×48×320, **Adam, Cosine LR = 2e-5**, 20 epochs, eval every 36 steps. Uses the
   stock ppocrv5 character dict.
4. **Environment fixes (this box)** — (a) upgraded `openpyxl` to kill the removed-`numpy.float`
   import error; (b) to train you need the PaddleOCR repo plugin: `mkdir` the missing
   `repo_manager/repos` dir, then `pip install -e <PaddleOCR repo> --no-deps` (this temporarily
   replaces pip's `paddleocr` with repo HEAD — restore with `pip install paddleocr==3.6.0
   --no-deps` afterwards for stable inference).
5. **Trained** — `PYTHONPATH=<PaddleOCR repo> python <repo>/tools/train.py -c
   finetune/rec_finetune.yml` on the GPU. Key lessons: **LR 2e-5 works; LR 3e-4 HURT**
   (dropped below baseline); set `save_epoch_step` high or checkpoints fill the disk.
   Best model at **epoch 9** (val acc 0.874); log in `finetune/train.log`.
6. **Evaluated** — scored stock vs fine-tuned on the held-out 196-image test split →
   **85.20% → 87.76%**, faint-prefix errors fixed, no clean-crop regression.
7. **Exported for inference** — converted the best checkpoint to a static inference model in
   `finetune/inference/` (`inference.json` + `.pdiparams` + `.yml`, ~81 MB). `extract_serials.py`
   auto-loads it (`--base-model` forces the stock recognizer). Large re-derivable artifacts
   (`finetune/output` ~1.7 GB, `pretrained` ~205 MB) were deleted afterward.

---

## 2B. Next fine-tune — dataset REBUILT (2026-07-10, ready to train)

The ROI review labels were merged in and the fine-tune dataset was rebuilt. **Training
not yet run** — dataset is staged and verified.

| | Old (current deployed model) | **New (rebuilt, staged)** |
|---|---:|---:|
| Readable labels | 1,144 | **6,709** (5.9×) |
| Meter sessions | — | 730 |
| Train / Val / Test | 773 / 175 / 196 | **4,643 / 1,098 / 968** |
| of which from ROI | 0 | **2,262 / 579 / 484** (3,325 total) |

**What was done:**
1. Backed up `results/ground_truth.csv` → `results/ground_truth.csv.bak` (original 4,148 rows).
2. Merged `roi_training_labels.csv` (4,286 rows) into `ground_truth.csv` → now **8,434 rows /
   6,709 readable**; mapped ROI verdict `not_visible → unreadable`. 0 duplicate images.
3. Patched `scripts/prepare_finetune_data.py` `IMG_DIRS` to add `dataset for test/ROI`.
4. Rebuilt `finetune/dataset/` (cleared old 1,144-image set first).

**Integrity verified:** 0 session leakage across splits · all 3,325 ROI labels included
(0 missing) · label format `images/<file>\t<SERIAL>` OK · charset 34 (0–9, A–X). The 309
human corrections (`verdict=fixed`) are the highest-value signal in the new data.

**Before training:** free disk (was at 99% / ~5 GB) and/or keep `save_epoch_step` high so
checkpoints don't fill the drive. Then train per the §2A step-5 recipe (LR 2e-5) and export
to `finetune/inference/`.

---

## 3. Deployed model (what production uses)

| File (in `finetune/inference/`) | Size | Purpose |
|---|---:|---|
| `inference.pdiparams` | 81 MB | trained weights |
| `inference.json` | 316 KB | network graph |
| `inference.yml` | 148 KB | preprocessing / config |
| **Total** | **~81 MB** | ship the **whole folder** (all 3 files) |

- Loaded via `TextRecognition(model_dir="finetune/inference")`.
- Runtime deps: `paddlepaddle(-gpu) 3.3.x`, `paddleocr==3.6.0`, `opencv-python`, `numpy`.
- **GPU is available on this box** (CUDA Paddle build, 1 GPU) — full 4,286-image run
  completes in minutes. (Earlier notes said CPU-only; that is now outdated.)
- Shrink options (optional): ONNX FP16 ≈ 40 MB (≈zero loss), INT8 ≈ 20 MB (~1–3% loss).

**Production essentials:** the `finetune/inference/` folder + a ~15-line recognition-only
loader + the **0.95 confidence gate** (the gate is what delivers the high auto-accept precision).

---

## 4. Latest inference — ROI blind test (this session, 2026-07-10)

Ran the deployed model on all **4,286** `dataset for test/ROI` crops (GPU, gate @0.95),
then **100% human-reviewed** via the review page.

**Model gate output:** 2,750 accepted · 1,048 review · 488 blank.

**Human verdicts (all 4,286):**

| Verdict | Count |
|---|---:|
| ✓ correct (model right) | **3,016** |
| ✗ wrong (human corrected) | **309** |
| 🚫 not visible (bad photo) | **961** |

### Accuracy (measured on the 3,325 *readable* crops; bad photos excluded)

> **Overall exact-match = 3,016 / 3,325 = 90.7%**

| Model status | Readable | Correct | **Accuracy** |
|---|---:|---:|---:|
| **accepted** (score ≥ 0.95) | 2,704 | 2,618 | **96.8%** ✅ |
| review (< 0.95) | 572 | 398 | 69.6% |
| blank (no read) | 49 | 0 | 0% |

**Interpretation:** the gate works as designed — when the model says *accepted*
(64% of all images / 81% of readable ones) it is right **96.8%** of the time and needs
no checking. The 90.7% overall is lower only because it folds in the low-confidence
*review* tier, which is meant for human eyes.

### Review output files
- `roi_results.csv` — full audit, 4,286 rows: `image, image_path, model_reading, score, model_status, verdict, correct_serial`
- `roi_training_labels.csv` — retraining labels, 4,286 rows: `image, true_label, verdict` (3,325 usable serials incl. **309 human corrections**; 961 blank/not-visible auto-excluded)
- `results/roi_serials.csv` — raw model output · `results/roi_review.html` — the review page

---

## 5. Accuracy by dataset (which dataset gives how much accuracy)

**Headline — one row per dataset:**

| Dataset | Images | Trained on? | Overall exact-match | Auto-accept precision (score ≥ 0.95) |
|---|---:|:--:|---:|---:|
| **TPWODL** (`data/serial_roi_test`) | 4,184 | ✅ 1,144 | **~95.1%** (on unseen) | **~95.4%** |
| **TPDDL** (`data/tpddl_test`) | 3,460 | ❌ | **~95.4%** | **~98.0%** |
| **ROI** (`dataset for test/ROI`) | 4,286 | ❌ | **90.7%** | **96.8%** |

> "Overall exact-match" = correct reads ÷ *readable* images (unreadable/bad-photo
> crops excluded). "Auto-accept precision" = of the reads the model is confident
> about (status = accepted), how many are correct — this is the number that matters
> for production, because those are the ones you trust without review.

### 5a. ROI — full breakdown (measured this session, 100% human-reviewed)

- Images: **4,286** total → **3,325 readable** · 961 not-visible (bad photos, excluded).

| Model confidence tier | Images | Readable | Correct | **Accuracy** |
|---|---:|---:|---:|---:|
| accepted (score ≥ 0.95) | 2,750 | 2,704 | 2,618 | **96.8%** ✅ |
| review (score < 0.95) | 1,048 | 572 | 398 | 69.6% |
| blank (no read) | 488 | 49 | 0 | 0% |
| **Overall** | **4,286** | **3,325** | **3,016** | **90.7%** |

### 5b. TPWODL & TPDDL — prior measured results

- **TPWODL** (`serial_roi_test`, 4,184 imgs, the training family): fine-tuned model
  scored **95.1% exact-match on unseen images**; accept-tier precision **~95.4%** @0.95.
  Fine-tuning lifted held-out test from 85.2% → 87.76%.
- **TPDDL** (`tpddl_test`, 3,460 imgs, second utility, **never trained on**): **~95.4%
  overall**, accept-tier precision **~98.0%** @0.95. Verdict: no retraining needed
  (only random hard-digit misreads, no systematic errors).

### 5c. Reading the numbers

- **The gate is the story.** Across every dataset, when the model reports `accepted`
  it is right **95–98%** of the time. That tier (≈64–80% of images) can be trusted
  without human review.
- **ROI's lower 90.7% overall** is not a weaker model — it's because ROI was scored
  across the *entire* confidence range including the hard `review` tier, whereas the
  TPWODL "95.1%" figure was on a cleaner unseen slice. Compare like-for-like via the
  accept-tier column (96.8% vs 95.4% vs 98.0%) — all consistent.

---

## 6. Repository map

```
scripts/
  extract_serials.py        # MAIN: recognition-only + 0.95 gate -> serials CSV
  build_review_page.py      # generic CSV -> review HTML
  roi_review_page.py        # ROI verification page (Correct/Wrong/Not-visible + correction box + exports)
  move_not_visible.py       # move 'not_visible' flagged crops to a separate folder
  score_accuracy.py         # score predictions vs ground_truth.csv
  make_labeling_page.py     # stratified labeling page
  prepare_finetune_data.py  # ground_truth.csv -> PaddleX rec dataset (session-split)
  compare_all_models.py     # (optional) 4-model comparison
data/                       # serialno_dev (501) · serial_roi_test (4,184) · tpddl_test (3,460)
dataset for test/ROI/       # 4,286 blind-test crops
finetune/
  inference/                # THE DEPLOYED MODEL (81 MB, 3 files)
  dataset/                  # PaddleX rec dataset (773/175/196)
  rec_finetune.yml          # training config (LR 2e-5)
  train.log                 # training run (best epoch 9)
results/                    # ground_truth.csv (4,148/3,387) + all output CSVs/HTML
report/                     # LaTeX methodology report
```

---

## 7. Next steps (optional)

1. **Retrain on the larger label set** — ✅ dataset merged & rebuilt (see §2B). **Remaining:**
   free disk, run `tools/train.py` (LR 2e-5), export to `finetune/inference/`, re-measure.
2. **Set aside bad photos** — `python3 scripts/move_not_visible.py --csv roi_results.csv`
   (961 not-visible ROI crops).
3. **Shrink for deployment** — export ONNX FP16 (~40 MB) via ONNX Runtime and confirm
   reads unchanged.
4. **One combined model** — fold in TPDDL labels (`results/feedback_tpddl.csv`) if a single
   model for both utilities is wanted.

---

## 8. Progress log (chronological)

| Date | Step | Outcome |
|---|---|---|
| 2026-07-06 | Pre-fine-tune model comparison on TPWODL (Paddle/TrOCR/EasyOCR) | Paddle best (65% / 89.1%); TrOCR 41% → dropped |
| 2026-07-07 | Fine-tuned PP-OCRv5 server rec on 1,144 TPWODL labels | 85.2% → 87.76% held-out; exported to `finetune/inference/` (81 MB) |
| 2026-07-07 | TPDDL cross-utility test (3,460 imgs) | ~95.4% overall / ~98% accept — no retrain needed |
| 2026-07-10 | ROI blind test (4,286 imgs), 100% human-reviewed | 90.7% overall / 96.8% accept; 309 corrections captured |
| 2026-07-10 | Built ROI review tooling | `scripts/roi_review_page.py` (Correct/Wrong/Not-visible + correction box + CSV export), `scripts/move_not_visible.py` |
| 2026-07-10 | **Merged ROI labels + rebuilt fine-tune dataset** | 1,144 → **6,709** readable (4,643/1,098/968), 0 leakage |
| 2026-07-13 | **Retrained** PP-OCRv5 server rec on 6,709 labels (LR 2e-5, 20 epochs) | best **val acc 0.874 → 0.9107** (+3.7pp), best epoch 7. Checkpoint at `finetune/output/`. Old model backed up at `finetune/inference_backup_2026-07-13/`. **Pending: export to inference + score on 968-img held-out test, then decide swap.** |

| 2026-07-13 | Mobile model (Path B) trained + tested for size | 7.5 MB but only **9%** on new data → rejected |
| 2026-07-15 | **Quantized the SERVER model** (Paddle 3.0.0 venv → PIR v1 → ONNX → static INT8) | **`server_int8.onnx` = 21 MB** (4× smaller). PTQ INT8. Key unlock: `quant_pre_process` folded 660 Constant nodes into initializers. |
| 2026-07-15 | **QAT** (quantization-aware training, PaddleSlim, patched slim tools for NRTR) | **`qat_inference/` = 22 MB INT8**, val 0.9117 ≈ FP32. QAT→ONNX blocked by paddle2onnx 2.1.0 → QAT deploys via Paddle. |
| 2026-07-15 | **3-model head-to-head** (500 imgs, human-judged) | FP32 94.7% / QAT 92.8% / PTQ 85.3% (on 414 readable). QAT ≫ PTQ for the same size. |
| 2026-07-15 | **Deployment package built** | `deploy/ptq_int8/` — self-contained ONNX package (model+dict+`meterocr.py`), Paddle-free, verified standalone. |

*(Keep appending rows here as the project progresses.)*

## 9. Current model artifacts & deploy (2026-07-15)
- **4 models** (see §0): FP32 81MB, QAT 22MB, PTQ 21MB, old-v1 81MB. Index in `MODELS/`.
- **Deployment (chosen):** `deploy/ptq_int8/` (PTQ INT8, ONNX-only, ~21 MB, portable).
- **Quantization pipeline** (reproducible): `size_reduction/compat_env/` (Paddle 3.0.0 venv).
  Comparison tooling: `size_reduction/compare_3models.py` + `build_3model_page.py`.
- **Disk note:** tight (~2 GB free). Reclaimable when done: `size_reduction/compat_env/venv`
  (~9.5 GB), training checkpoints in `compat_env/output` + `prune_qat/qat_output` (~0.4 GB
  each, keep `best_accuracy.pdparams` only if you may re-quantize).

### Quantization — final recipe (works, reproducible)
The server model **can** be quantized, but NOT with the default Paddle 3.3.1 tooling. The working chain:
1. venv with **Paddle 3.0.0** (exports PIR **v1**; 3.3.1 exports v4 which `paddle2onnx` can't convert).
2. Retrain the server in that venv → export (v1) → `paddle2onnx` → ONNX FP32.
3. **`quant_pre_process`** (skip_symbolic_shape=True) to fold Constant nodes → initializers.
4. **`quantize_static`**, `per_channel=False`, `QuantFormat.QOperator`, weight=QInt8/act=QUInt8, calibrate on ~100 real crops.
Result: **21.2 MB INT8**, ~80% accuracy. Full code in `size_reduction/compat_env/`.
Deploy via ONNX Runtime + `size_reduction/onnx_reader.py`. (Note: the venv is ~9.5 GB — `rm -rf` it to reclaim disk once the model is saved.)
