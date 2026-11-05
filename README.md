# Meter Serial-Number OCR

Reads the **serial number** printed on electricity-meter nameplates from cropped photos, using a
**fine-tuned PaddleOCR PP-OCRv5 recognizer** and a **21 MB INT8 model** that runs on CPU without Paddle.

![input crop](assets/sample_meter_crop.jpg)  →  ![extracted result](assets/result_after.png)

```
input : a 232×50 crop of a meter nameplate
output: TPCODL TS0319159   (confidence 0.98, status: accepted)
```

Reads with confidence ≥ 0.95 are **accepted** automatically; the rest go to **review** by a person.

## Versions

| Version | Trained on | old test (1,024) | new test (1,680) | hard (171) | Status |
|---|---|---|---|---|---|
| v1 | 1,144 labels | — | — | — | replaced |
| v2 | 6,709 labels | — | — | — | replaced |
| **v3** | 6,733 labels (`datasets/v3_train`) | PTQ 87.21% | PTQ 91.01% | PTQ 11.70% | **deployed** |
| **v4** | 23,707 labels (`datasets/v4_train`), started from v3 | PTQ 90.04% | PTQ 98.81% | PTQ 90.64% | candidate |
| **v4.2** | 24,434 labels (v4 + 727 reviewed 4-Nov crops), started from v3 | FP32 91.31%* | FP32 98.81%* | FP32 92.40%* | FP32 done, PTQ next |

PTQ INT8 numbers, exact match, deploy pipeline (`results/2026-11-02_v4/`). *v4.2: FP32, training-eval
pipeline (fixed 48×320 resize) — not directly comparable until its PTQ is scored. History and caveats:
**`PROJECT_SUMMARY.md`**.

## v4.2 — FP32 trained (2026-11-05), PTQ next
The v4 recipe (start from v3, 20 epochs, batch 128) on v4's data + 727 reviewed 4-Nov crops
(new train crops repeated ×3). Best val 96.00% at epoch 10. FP32, training-eval pipeline
(`training/output_v4.2/eval_v3_v4_v4.2.txt`, local; training log in `training/logs/v4.2_train.log`):

| test set | v3 | v4 | v4.2 |
|---|---|---|---|
| test_old (1,024, 74% letter serials) | 91.02% | 90.82% | **91.31%** |
| test_new (1,680) | 93.39% | **98.99%** | 98.81% |
| test_new_hard (171) | 35.09% | 91.81% | **92.40%** |
| test_nov4 (74, new 4-Nov crops) | 98.65% | 98.65% | **100.00%** |

Try 1 (2026-11-04, fine-tuned *from v4* for 8 epochs) was slightly worse than v4 everywhere; kept
locally in `models/v4.2_try1/` (config `training/rec_v4.2_try1.yml`, log `training/logs/v4.2_try1_train.log`).

Steps, each one command, run in order (1–4 done):
1. `.venv/bin/python scripts/review_round3_server.py` — decide the doubtful labels (port 8097)
2. `.venv/bin/python scripts/build_v4_retrain_data.py` — apply all review rounds
3. `.venv/bin/python datasets/merge_into_v4_train.py` — merge into `datasets/v4_train`
4. `python3 training/train_v4_2.py` — v4.2 FP32 → `models/v4.2/`
5. `python3 quantization/quantize_v4_2_ptq.py` — v4.2 PTQ INT8 + scores v3 / v4 / v4.2
6. `python3 quantization/quantize_v4_2_ptq.py --remove-old-v4` — after reading the scores

Training uses ~14.4 of the RTX 5080's 16 GB: if the monitor is plugged into that card the desktop
lags until training ends — plug it into the motherboard port instead.

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
├── docs/            interview Q&A and report zips (local only)
└── assets/          README images
```
Datasets and model binaries are git-ignored (too large); the deploy package is committed.

## Run the OCR
```bash
# deployed model (ONNX, no Paddle)
cd deploy/ptq_int8 && pip install -r requirements.txt
python3 -c "from meterocr import MeterOCR; print(MeterOCR().predict('crop.jpg'))"
# -> ('TPCODL TS0319159', 0.98, 'accepted')

# v4 with the same code
python3 -c "from meterocr import MeterOCR; print(MeterOCR(model_path='../../models/v4/meterocr_v4_int8_ptq.onnx').predict('crop.jpg'))"

# dashboard: upload one image or a folder, compare all 6 models side by side, download CSV
.venv/bin/python dashboard/app.py          # then open http://127.0.0.1:8095

# a whole folder on all CPU cores -> CSV
.venv/bin/python scripts/run_ptq_folder.py --images <crops_dir> --out reads.csv --model v4
```

## Train / quantize
```bash
python3 training/train_v4.py           # how v4 was trained (refuses to overwrite models/v4)
python3 quantization/quantize_v4_ptq.py  # how v4 was quantized + scored (re-prints the summary)
```
Both switch to the right Python env by themselves. The GPU training env (Paddle 3.3.1, CUDA 13),
the Paddle 3.0.0 export env and the PaddleOCR repo are in
`/home/coral/Desktop/Prince/Prince number detection in meter/` (outside this repo).
