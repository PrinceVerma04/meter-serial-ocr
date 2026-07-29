# Meter Serial-Number OCR

Extract the alphanumeric **serial number** printed on electricity-meter nameplates
from cropped photos — using a **fine-tuned PaddleOCR PP-OCRv5 recognizer**, with
INT8-quantized models for lightweight deployment.

---

## Demo — before → after

**Input** (a cropped meter-nameplate serial region):

![input crop](assets/sample_meter_crop.jpg)

**Output** (the model reads the serial):

![extracted result](assets/result_after.png)

```
input : a 232×50 crop of a meter nameplate
output: TPCODL TS0319159   (confidence 0.98, status: accepted)
```

---

## What this is

- **Task:** read the serial number from an already-cropped meter-nameplate image
  (recognition-only — the crop is the serial ROI, so no text detector is needed).
- **Model:** PP-OCRv5 server recognizer, **fine-tuned on 6,709 hand-labeled meter serials**.
- **Confidence gate:** high-confidence reads are auto-accepted; the rest route to
  a human review queue.

## Models (`MODELS/README.md` for the full index)

| Model | Size | Precision | Accuracy* | Runtime | Deployment package |
|---|---|---|---|---|---|
| **FP32 (original)** | 81 MB | 32-bit | **94.7%** | Paddle | — |
| **QAT INT8** ⭐ | 22 MB | 8-bit (QAT) | **92.8%** | Paddle | `deploy/qat_int8/` |
| **PTQ INT8** | 21 MB | 8-bit (PTQ) | 85.3% | ONNX | `deploy/ptq_int8/` |

*Exact-match on readable images of an unseen out-of-distribution utility (TCPODL/TPSODL).
On in-domain data all run ~95%+. See `PROJECT_SUMMARY.md` for the full comparison.

## Quick start — run the OCR (deployment package)

**ONNX (portable, no Paddle):**
```bash
cd deploy/ptq_int8
pip install -r requirements.txt
python3 -c "from meterocr import MeterOCR; print(MeterOCR().predict('sample.jpg'))"
# -> ('TPCODL TS0319159', 0.98, 'accepted')
```

**Paddle (higher accuracy):** same API in `deploy/qat_int8/` (needs `paddlepaddle`).

Each package bundles the model + dictionary + `meterocr.py` (preprocess + CTC decode +
confidence gate) + a `benchmark.py` to measure CPU/GPU latency. See each folder's README.

## Repository layout

```
.
├── README.md · PROJECT_SUMMARY.md · MODELS/README.md   # docs
├── deploy/                      # ready-to-ship deliverable packages
│   ├── ptq_int8/               #   ONNX INT8 package  (21 MB)
│   ├── qat_int8/               #   Paddle INT8 package (22 MB)
│   └── ptq_py38/               #   Python-3.8 / Docker service variant
├── scripts/                     # extraction, review pages, scoring, data prep, batch inference
├── analysis/                    # model-comparison review dashboards (fp32_ptq, fp32_qat, …)
├── webapp/                      # Flask inference-timing demo (CPU/GPU)
├── size_reduction/             # quantization pipeline (PTQ + QAT) + comparison tooling
│   └── compat_env/             # Paddle 3.0.0 recipe (exports the ONNX-convertible format)
├── finetune/                   # training config + log
├── results/                    # ground-truth + comparison CSVs
└── datasets/                    # working image sets for testing (git-ignored)
```
> Datasets, the training venv, and 200 MB+ checkpoints are **not** in the repo
> (`.gitignore`) — they're re-derivable; see `PROJECT_SUMMARY.md`.

## Methodology & full story
- **`PROJECT_SUMMARY.md`** — datasets, fine-tuning (85.2% → 87.8%), retrain, the full
  quantization journey (PTQ vs QAT), measured accuracy, latency, and the progress log.
- **`MODELS/README.md`** — the model index + naming scheme.
- **Quantization pipeline** — `size_reduction/compat_env/` (reproducible).

## Pipeline scripts (for the full dataset, not the deploy package)
```bash
python3 scripts/extract_serials.py  --data <crops_dir> --out results/out.csv --threshold 0.95
python3 scripts/build_review_page.py --csv results/out.csv --data <crops_dir> --out results/review.html
python3 scripts/score_accuracy.py    --preds results/out.csv
```
