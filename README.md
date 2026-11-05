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
| **v4.2** (v4 recipe + 727 reviewed 4-Nov crops) | FP32 trained (2026-11-05); PTQ INT8 and deploy-pipeline scoring next |

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
| **v4.2** | v3 FP32 weights | 24,434 labels (v4 + 727 reviewed 4-Nov crops) | FP32 91.31%* | FP32 98.81%* | FP32 92.40%* | FP32 done, PTQ next |

Exact match, INT8 PTQ, deploy pipeline (`results/2026-11-02_v4/`).
*v4.2 numbers are FP32 from the training-eval pipeline (fixed 48×320 resize) and are not comparable with the PTQ rows until v4.2 is quantized and scored.

### How to read these numbers

- **`test_old` is the neutral comparison.** v4 PTQ gets 29 more images right than v3 PTQ (90.04% vs 87.21%), but it also makes more confidently wrong reads (74 vs 58). Check those 74 before deploying v4.
- **`test_new` and `test_new_hard` were built around v3's behavior.** The 1,509 easy images were chosen because v3 read them right, and the 171 hard images because v3 misread them. v3's score on the hard set is therefore low by construction, and the jump from 11.70% to 90.64% overstates the typical gain. (v3 FP32 scores 35.09% on the hard set, so part of v3's 11.70% is quantization loss.)
- **v4 survives quantization better than v3.** FP32-to-INT8 agreement is **98.6%** for v4 against **93.3%** for v3.

### v3 against utility records (large clean test set)

`datasets/test/clean_test_oct27/`: 21,042 sharp crops, all of them misread by the department's old OCR.

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

## v4.2: FP32 trained (2026-11-05), PTQ next

The v4 recipe on v4's data plus 727 reviewed 4-Nov crops (new train crops repeated ×3). Best validation accuracy 96.00% at epoch 10. FP32, training-eval pipeline (`training/output_v4.2/eval_v3_v4_v4.2.txt`):

| test set | v3 | v4 | v4.2 |
|---|---|---|---|
| test_old (1,024, 74% letter serials) | 91.02% | 90.82% | **91.31%** |
| test_new (1,680) | 93.39% | **98.99%** | 98.81% |
| test_new_hard (171) | 35.09% | 91.81% | **92.40%** |
| test_nov4 (74, new 4-Nov crops) | 98.65% | 98.65% | **100.00%** |

An earlier attempt (fine-tuning from v4 for 8 epochs) was slightly worse than v4 on every set and is kept locally as `models/v4.2_try1/`.

Steps, one command each (1–4 done):
1. `.venv/bin/python scripts/review_round3_server.py` — decide the doubtful labels (port 8097)
2. `.venv/bin/python scripts/build_v4_retrain_data.py` — apply all review rounds
3. `.venv/bin/python datasets/merge_into_v4_train.py` — merge into `datasets/v4_train`
4. `python3 training/train_v4_2.py` — v4.2 FP32 → `models/v4.2/`
5. `python3 quantization/quantize_v4_2_ptq.py` — v4.2 PTQ INT8 + scores v3 / v4 / v4.2
6. `python3 quantization/quantize_v4_2_ptq.py --remove-old-v4` — after reading the scores

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
├── models/          base/ · v3/ · v4/ · v4.2/  — FP32 Paddle, FP32 ONNX, INT8 PTQ ONNX, training weights (README inside)
├── deploy/          ptq_int8/ (v3 in production) · meterocr_ptq_py38_v3.zip (Python 3.8 server)
├── training/        train_v4.py · train_v4_2.py (auto-resuming) · rec_v3/v4/v4.2.yml · logs/
├── quantization/    quantize_v4_ptq.py · quantize_v4_2_ptq.py (export → ONNX → INT8 → scoring) · compat_env/
├── results/         2026-10-27_v3_clean_test/ · 2026-11-02_v4/ · 2026-11-04_all_models_test/
├── dashboard/       app.py + index.html — local web page: image/folder in, every model's read out
├── scripts/         infer · run_ptq_folder · step1–3 (all-model test) · review_round2/3 servers · build_v4_retrain_data
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
- Run `quantization/quantize_v4_2_ptq.py`, then decide between v4 and v4.2.
- Decide on deploying v4 PTQ; inspect its 74 confidently wrong reads on `test_old` first.
- Test v4 on sets it never saw beyond its own test split: `datasets/test/tpddl/`, `datasets/test/tpcodl_human_verified/` and `datasets/test/tpddl_july_2026/` (6,153 images, unseen by v3 and v4; labels are easy-biased).
