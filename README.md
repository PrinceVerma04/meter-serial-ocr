# Meter Serial-Number OCR

Reads the **serial number** printed on electricity-meter nameplates from already-cropped photos of the serial area. It combines a **fine-tuned PaddleOCR PP-OCRv5 recognizer** (recognition only, no text detector), a **confidence gate**, and a **≈21 MB INT8 ONNX model** that runs on CPU without Paddle.

![input crop](assets/sample_meter_crop.jpg)  →  ![extracted result](assets/result_after.png)

```
input : a 232×50 crop of a meter nameplate
output: ('<serial number>', 0.98, 'accepted')
```

## Status

| Model | Status |
|---|---|
| **v3** (PTQ INT8) | **In production**: `deploy/ptq_int8/` and the Python 3.8 Flask server package |
| **v4** (v3 retrained on human-reviewed data) | Candidate, not deployed. Decision pending (see "How to read these numbers") |
| v4.2 (v4 recipe + 727 reviewed 4-Nov crops) | Tried 2026-11-05 and dropped: no better than v4 after INT8; models deleted |

## Confidence gate

Every read gets a confidence score. Reads with confidence ≥ 0.95 are **auto-accepted**; the rest go to **human review**.
[CONFIRM from `deploy/ptq_int8/meterocr.py`: whether a third outcome, **recapture**, exists and what triggers it. Describe it here exactly as the code does.]

Gate results below are from the first fine-tuned model (**v1**, July 2026) and were not re-measured on v3 or v4:

| Test | Images | Result |
|---|---|---|
| Blind test, every image human-reviewed | 4,286 (3,325 readable, 961 not visible) | 90.7% exact match on readable images; **96.8% precision on accepted reads** |
| Second utility, never trained on | 3,460 | ~95.4% overall; ~98.0% precision on accepted reads |

[CONFIRM: share of the 4,286 blind-test images that were auto-accepted at ≥ 0.95 (the "64%" figure), and the file in `results/` that shows it.]

For the current models, accept precision on `test_old` is 93.48% (v3 PTQ) and 92.38% (v4 PTQ).

## Versions

| Version | Started from | Training data | test_old (1,024) | test_new (1,680) | test_new_hard (171) | Status |
|---|---|---|---|---|---|---|
| v1, v2 | stock | 1,144 / 6,709 labels | — | — | — | replaced |
| **v3** | stock | 6,733 labels (`datasets/v3_train`) | PTQ 87.21% | PTQ 91.01% | PTQ 11.70% | **deployed** |
| **v4** | v3 FP32 weights | 23,707 labels (`datasets/v4_train`) | PTQ 90.04% | PTQ 98.81% | PTQ 90.64% | candidate |
| v4.2 | v3 FP32 weights | 24,434 labels (v4 + 727 reviewed 4-Nov crops) | PTQ 90.82% | PTQ 98.27% | PTQ 92.40% | tried, dropped |

Exact match, INT8 PTQ, deploy pipeline (`results/2026-11-02_v4/`; v4.2: `results/2026-11-05_v4.2/`).

### How to read these numbers

- **`test_old` is the neutral comparison.** v4 PTQ gets 29 more images right than v3 PTQ (90.04% vs 87.21%), but it also makes more confidently wrong reads (74 vs 58). Check those 74 before deploying v4.
- **`test_new` and `test_new_hard` were built around v3's behavior.** The 1,509 easy images were chosen because v3 read them right, and the 171 hard images because v3 misread them. v3's score on the hard set is therefore low by construction, and the jump from 11.70% to 90.64% overstates the typical gain. (v3 FP32 scores 35.09% on the hard set, so part of v3's 11.70% is quantization loss.)
- **v4 survives quantization better than v3.** FP32-to-INT8 agreement is **98.6%** for v4 against **93.3%** for v3.

### v3 against utility records (large clean test set)

`datasets/test/clean_test_oct27/`: 21,042 sharp crops when scored, all of them misread by the department's old OCR (72 were deleted on 2026-11-05, see `CLEANUP_LOG.md`; 20,970 remain).

| Scoring | v3 FP32 | v3 PTQ (deployed) |
|---|---|---|
| Exact match vs department serial (21,042) | 83.0% | 79.6% |
| After manual review, scorable images only (18,105) | 93.33% | 91.04% |
| Accept precision after review | 94.29% | 93.49% |

Every image either model got wrong (4,744) was reviewed by people: 1,771 department serials were confirmed, 36 were corrected, and 2,937 (62%) were unreadable. Most apparent errors were photos that people cannot read either.

## v4: how it was trained

- **Data (23,707):** v3's 6,733 labels plus 16,974 new crops from the clean test set: 1,807 human-reviewed images that v3 misread, and 15,167 where the department serial, EasyOCR, v3 FP32 and v3 PTQ all agreed with confidence ≥ 0.95. New images were split 80/10/10 by capture session.
- **Start:** v3 FP32 (its inference export was mapped back to training weights; all 458 weights match by layer name).
- **Recipe** (`training/train_v4.py`): 20 epochs, batch 128, LR 4e-5, RTX 5080, **22 minutes**. Best validation accuracy **96.23% at epoch 16**.

## v4.2: v4 retrained on the 4-Nov clean crops, tried and dropped (2026-11-05)

The 2,454 clean crops of the 30k vision-model run were reviewed down to 2,250; 1,523 of them were already in v4's training data and the other **727** were added (`label_source = nov4_reviewed`). v4 was retrained on that (v4 recipe: start from v3, 20 epochs, new crops ×3; `training/train_v4_2.py`) and quantized with a recipe search (`quantization/quantize_v4_2_ptq.py`: CTC head and attention kept FP32, percentile calibration on 1,000 crops, chosen on the validation split). Deploy pipeline (`results/2026-11-05_v4.2/`):

| All 8,931 test images (old + new + 4-Nov + TPDDL July) | v3 INT8 (deployed) | **v4 INT8** | v4.2 INT8 |
|---|---|---|---|
| Right | 8,185 | **8,674** | 8,663 |
| Wrong but auto-accepted (≥ 0.95) | 410 | 200 | 179 |
| Sent to review | 674 | **247** | 426 |
| Size / CPU time per crop | 21.4 MB / 28 ms | **21.4 MB / 28 ms** | 28.0 MB / 29 ms |

v4.2's 21 fewer wrong auto-accepts come only from lower confidence: at the same review workload v4 INT8 lets through as few or fewer wrong reads (200 vs 204; 169 vs 179). FP32 against FP32 the gain is 11 images (8,698 vs 8,687). **Decision: keep v4, delete v4.2** (models and checkpoints removed; scripts, configs, logs and results kept as the record). An earlier fine-tune *from* v4 (4 Nov) was worse than v4 on every set.

Training uses ~14.4 of the RTX 5080's 16 GB; if the monitor is plugged into that card the desktop lags until training ends, so use the motherboard port.

## Quantization and deployment

**PTQ INT8 recipe** (works for v3 and v4):
1. Export the training weights with a **Paddle 3.0.0** environment (PIR v1). Paddle 3.3.1 exports PIR v4, which `paddle2onnx 2.1.0` cannot read.
2. `paddle2onnx` → ONNX FP32, then fold constants in Python with `quant_pre_process` (a space in the project path breaks paddle2onnx's own folding). This also drops the unused NRTR head (205 → 84 MB).
3. `quantize_static`: QDQ, **per-tensor** scaling, Conv/MatMul/Gemm, **random** calibration crops (200 crops, not one capture batch). Per-channel scaling truncated long serials (~47% FP32/INT8 agreement), which is why per-tensor is used.

**Size and speed:** ≈21 MB INT8 model; ≈18 ms per crop on one CPU thread against ≈253 ms for FP32.
[CONFIRM: these are the figures in the project summary. Your CV states 81 MB → 21 MB and 304 ms → 20 ms per image. Take the exact values from the logs in `results/2026-11-02_v4/` and make this line match, naming the CPU and batch size.]

**Deployment**
- **In production:** v3 PTQ INT8. `deploy/ptq_int8/` contains the model, dictionary and `meterocr.py` (preprocessing, CTC decoding and the 0.95 gate), and needs only ONNX Runtime. `deploy/meterocr_ptq_py38_v3.zip` is the Python 3.8 server variant: pinned onnxruntime 1.16.3, Flask app, systemd unit, Docker, port 8080.
- [CONFIRM: health-check endpoint name and how it is used in the systemd/Docker setup.]
- **To switch to v4:** use `models/v4/meterocr_v4_int8_ptq.onnx` in the same package (same input, output, dictionary and gate). Not done yet; this is a decision for the team.

## Scope

This repository covers **recognition only**: the input is already a crop of the serial region, so no text detector is used.
[CONFIRM: if the YOLO symbol detector (fine-tuned YOLO variants benchmarked under varied lighting) and the combined detection + OCR pipeline live in another repository, link it here and say how crops reach this recognizer.]

## Layout
```
.
├── README.md · PROJECT_SUMMARY.md · CLEANUP_LOG.md
├── datasets/        training data per version + test sets, each images/ + labels.csv  (README inside)
├── models/          base/ · v3/ · v4/  — FP32 Paddle, FP32 ONNX, INT8 PTQ ONNX, training weights (README inside)
├── deploy/          ptq_int8/ (v3 in production) · meterocr_ptq_py38_v3.zip (Python 3.8 server)
├── training/        train_v4.py · train_v4_2.py (v4.2 record) · rec_v3/v4/v4.2.yml · logs/
├── quantization/    quantize_v4_ptq.py · quantize_v4_2_ptq.py (INT8 recipe search) · compat_env/
├── results/         2026-10-27_v3_clean_test/ · 2026-11-02_v4/ · 2026-11-04_all_models_test/ · 2026-11-05_v4.2/ · 2026-11-05_review_21425/
├── dashboard/       app.py + index.html — local web page: image/folder in, every model's read out
├── scripts/         infer · run_ptq_folder · step1–3 (all-model test) · review_round2/3 servers · build_v4_retrain_data · accepted_summary · build_review_dashboard · export_all_same
├── docs/            internal notes and report zips (local only)
└── assets/          README images
```
Datasets and model binaries are git-ignored (too large); the deploy package is committed. Which version used which data, and where every label came from, is in `datasets/README.md`.

## Run the OCR
```bash
# deployed model (ONNX, no Paddle)
cd deploy/ptq_int8 && pip install -r requirements.txt
python3 -c "from meterocr import MeterOCR; print(MeterOCR().predict('crop.jpg'))"
# -> ('<serial number>', 0.98, 'accepted')

# v4 with the same code
python3 -c "from meterocr import MeterOCR; print(MeterOCR(model_path='../../models/v4/meterocr_v4_int8_ptq.onnx').predict('crop.jpg'))"

# dashboard: upload one image or a folder, compare all 6 models side by side, download CSV
.venv/bin/python dashboard/app.py          # then open http://127.0.0.1:8095

# a whole folder on all CPU cores -> CSV
.venv/bin/python scripts/run_ptq_folder.py --images <crops_dir> --out reads.csv --model v4
```

## Train / quantize
```bash
python3 training/train_v4.py             # how v4 was trained (refuses to overwrite models/v4)
python3 quantization/quantize_v4_ptq.py  # how v4 was quantized + scored (re-prints the summary)
```
Both switch to the right Python environment by themselves. The GPU training environment (Paddle 3.3.1, CUDA 13), the Paddle 3.0.0 export environment and the PaddleOCR repository live outside this repository (see `PROJECT_SUMMARY.md`).

## Open items
- Finish cleaning the 5-Nov review set with `scripts/build_review_dashboard.py` (filter by acceptance, model agreement and correct serial; confirmed crops go to `all_models_same/`).
- Decide on deploying v4 PTQ; inspect its 74 confidently wrong reads on `test_old` first.
- Test v4 on sets it never saw beyond its own test split: `datasets/test/tpddl/`, `datasets/test/tpcodl_human_verified/` and `datasets/test/tpddl_july_2026/` (6,153 images, unseen by v3 and v4; labels are easy-biased).
