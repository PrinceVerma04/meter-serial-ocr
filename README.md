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

PTQ INT8 numbers, exact match, deploy pipeline (`results/2026-11-02_v4/`). History and caveats:
**`PROJECT_SUMMARY.md`**.

## Layout
```
.
├── README.md · PROJECT_SUMMARY.md · CLEANUP_LOG.md
├── datasets/        training data per version + test sets, each images/ + labels.csv  (README inside)
├── models/          base/ · v3/ · v4/  — FP32 Paddle, FP32 ONNX, INT8 PTQ ONNX, training weights (README inside)
├── deploy/          ptq_int8/ (v3 in production) · meterocr_ptq_py38_v3.zip (Python 3.8 server)
├── training/        train_v4.py (one auto-resuming command) · rec_v3.yml · rec_v4.yml · logs/
├── quantization/    quantize_v4_ptq.py (export → ONNX → INT8 → scoring) · quantize_v3_ptq.py · compat_env/
├── results/         2026-10-27_v3_clean_test/ · 2026-11-02_v4/
├── scripts/         infer · extract_serials · run_ptq_folder · compare_fp32_ptq · score_vs_department
├── docs/            interview Q&A and report zips
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
