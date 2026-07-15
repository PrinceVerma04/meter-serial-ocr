# Meter Serial Number Detection

Extract the alphanumeric serial number printed on electricity-meter nameplates
from cropped photos, using a **fine-tuned PaddleOCR PP-OCRv5 recognizer**.

## Current state (measured on 4,148 hand-labeled images)
- **Model:** PP-OCRv5 server recognizer, **fine-tuned on our labels**, recognition-only
  (no detector — the crops are already the serial ROI). Runs on GPU.
  Lives in `finetune/inference/`; `extract_serials.py` uses it automatically.
- **Accuracy:** **95.1% exact-match on unseen images** (held-out test earlier: 85.2% → 87.8% after fine-tuning).
- **Confidence gate @0.95:** auto-accept ~76% of images at **95%+ precision**; the rest go to a review queue.
- TrOCR/EasyOCR were evaluated and **dropped** — the fine-tuned recognizer beats them
  (in disagreements, it is right 93% vs TrOCR 1%).

## Folder structure
```
.
├── README.md
├── scripts/
│   ├── extract_serials.py       # MAIN: fine-tuned recognizer + confidence gate -> serials CSV
│   ├── build_review_page.py     # CSV -> review/feedback HTML (zoom, copy-path, live labeling)
│   ├── score_accuracy.py        # score predictions vs results/ground_truth.csv
│   ├── make_labeling_page.py    # stratified sample -> labeling page
│   ├── prepare_finetune_data.py # ground_truth.csv -> PaddleX rec dataset (session-split)
│   └── compare_all_models.py    # (optional) 4-model comparison utility
├── data/
│   ├── serialno_dev/            # 501 dev crops
│   └── serial_roi_test/         # 4,184 test crops (deduplicated)
├── finetune/
│   ├── inference/               # the fine-tuned recognizer (deployed model)
│   ├── dataset/                 # PaddleX rec dataset (train/val/test, session-split)
│   └── rec_finetune.yml         # training config
├── results/
│   ├── ground_truth.csv         # 4,148 hand labels (3,387 readable) — eval + retrain
│   ├── test_serials.csv         # MAIN OUTPUT (fine-tuned reads + status)
│   ├── test_review.html         # review/feedback viewer
│   └── feedback_test_finetuned.csv  # raw review record
└── report/                      # LaTeX methodology report
```

## Output columns (`test_serials.csv`)
`image, meter_id, display_text, serial_only, score, final_serial, status`
- **`final_serial`** — the answer (a serial, or `NEEDS_REVIEW`, or blank)
- **`status`** — `accepted` (score ≥ 0.95, ~95% correct) · `review` · `blank`

## Run the pipeline
```bash
cd "/home/ganesh/Desktop/Prince number detection in meter"
python3 scripts/extract_serials.py --data data/serial_roi_test --out results/test_serials.csv --threshold 0.95
python3 scripts/build_review_page.py --csv results/test_serials.csv --data data/serial_roi_test --out results/test_review.html
```
- Uses the fine-tuned model automatically (`--base-model` forces the stock one).
- Open `results/test_review.html` to review; it doubles as a labeling tool
  (✓ correct / fix / ✗ unreadable, "Download feedback.csv").

## Accuracy check
```bash
python3 scripts/score_accuracy.py --preds results/test_serials.csv
```

## Retraining (optional, to go past 95%)
`results/ground_truth.csv` now has 3,387 readable labels. To retrain:
1. `python3 scripts/prepare_finetune_data.py`  (rebuilds `finetune/dataset/`)
2. Train via the PaddleOCR repo: `PYTHONPATH=<repo> python3 <repo>/tools/train.py -c finetune/rec_finetune.yml` (use LR ~2e-5; see project notes).
3. Export to `finetune/inference/` and re-run the pipeline.
Free disk space first — the drive runs near full.
