# Meter Serial-Number OCR — Project Report

**Project:** Extract the alphanumeric serial number from each cropped meter-nameplate image in `serialno/`
**Report date:** 2026-07-06
**Dataset:** ~501 cropped nameplate images (no ground-truth labels yet)

---

## 1. Tasks COMPLETED

### 1.1 OCR pipeline built — `ocr_serials.py`
- Preprocess (2× upscale + CLAHE) → PaddleOCR PP-OCRv5 (run on raw + preprocessed) →
  `best_serial()` picks the most serial-like token (drops labels like Sr.No./TPWODL, strips fused brand) →
  `choose_serial()` ensemble picker.
- Output: `serials.csv` (image, session_id, predicted_serial, score).
- `make_review_page.py` → `review.html` visual review grid, color-coded by confidence.

### 1.2 Error classes fixed
1. **Suffix truncation** (10200 → 10200735) — FIXED. Raw-primary ensemble: trust the raw read, use preprocessed only to complete it.
2. **Dropped letter-prefix** (TWNX 519026 → 519026) — FIXED. Joins short 2–4 char alpha prefix to its digit group; recovered ~14 serials.
3. **Brand `TPWODL` fused to digits** — FIXED. `strip_fused_brand()` drops a leading alpha run 5+ chars or containing "ODL".
4. Faint leading letters (LW220305 → 20305) — data-quality limit, needs manual review.
5. Blur false-positives (confident but wrong) — NOT yet gated.
6. Multi-number crops — selector may pick wrong number — NOT yet gated.

### 1.3 Multi-model comparison — `compare_models.py` / `models_compare.html`
Tested PaddleOCR mobile vs server vs EasyOCR vs TrOCR. On all 501: 184 all-agree, 227 disagree, 90 all-blank.
Example (single image):

| model | prediction | confidence | verdict |
|---|---|---|---|
| paddle_server | TW8650378 | 0.947 | ✅ correct, honest confidence |
| trocr | TW8650378 | 1.000 | ✅ correct but confidence is fake (always 1.0, hallucinates) |
| paddle_mobile | TWB650378 | 0.916 | ❌ B/8 slip |
| easyocr | NUU50378 | 0.266 | ❌ wrong (self-flags low) |

**Conclusion:** PaddleOCR **server** is the production winner — correct AND honestly confident.

### 1.4 Detector root-cause diagnostic (2026-07-04)
Question: is the **detector** or the **recognizer** losing serials?
Ran `server_det` vs `mobile_det` (recognizer held constant) on 10 images that `mobile_det+server_rec` lost.
**Result: `server_det` recovered 6/10 → the TEXT DETECTOR is the cause.**
The other 4 return `[]` from both detectors (separate detection-hard class:
g_1782008352 / 08425 / 08507 / 09626).
Full table: `DIAGNOSTIC_detector_cause_summary.md`.

### 1.5 Model size analysis (2026-07-06)
Actual on-disk sizes measured:

| model | components | size |
|---|---|---|
| paddle_mobile | mobile_det 4.8 + en_mobile_rec 7.7 | **12.5 MB** |
| paddle_server | server_det 85 + server_rec 82 | **167 MB** |
| easyocr | CRAFT 80 + english_g2 15 | **95 MB** |
| trocr (base-printed) | recognition-only | **~1.3 GB** |
| **all bundled** | | **~1.57 GB** |

### 1.6 Deliverables
- `serials.csv`, `review.html`, `models_compare.html`
- `meter_ocr_report.zip` (Overleaf/LaTeX methodology report with figures)

---

## 2. Production DECISIONS made
- **Production config = PP-OCRv5 `server_det + server_rec` (167 MB), 368/501 reads.** User chose the server recognizer for accuracy.
- **Do NOT swap `server_det` → `mobile_det`** to save size: mobile_det fails to locate text on ~20–30 faint/low-contrast crops (verified: server_det recovers 6/10).
- EasyOCR (worst) and TrOCR (1.3 GB, no usable confidence, hallucinates) are dropped from production.

### Size-reduction path (from 167 MB, no meaningful accuracy loss)
| method | size | accuracy cost |
|---|---|---|
| Ship only the winner (drop other 3 families) | 1.57 GB → **167 MB** | zero |
| FP16 / ONNX FP16 | ~**84 MB** | ~zero (recommended zero-loss target) |
| INT8 dynamic quant | ~**42 MB** | ~1–3% (a few reads) |
| 4 MB hard target | recognition-only mobile INT8 (~3.5–4 MB) | real drop (~300–330 reads est.) |

**Env caveat:** oneDNN/mkldnn bug on this CPU build — FP16/INT8 should go via **ONNX Runtime**, not Paddle's oneDNN path. (See `paddleocr-env-quirks`.)

---

## 3. Tasks IN PROGRESS / PENDING

| # | Task | Status |
|---|---|---|
| 1 | **Hand-label 30–50 images as a test set** — no ground truth exists; current 368/501 is a CONFIDENCE count, not verified accuracy | HIGHEST PRIORITY, not started |
| 2 | Export `paddle_server` → **ONNX FP16** and confirm ~84 MB with unchanged reads | proposed, awaiting go-ahead |
| 3 | Build **4 MB recognition-only INT8** model and benchmark real read count on 501 | proposed, awaiting go-ahead |
| 4 | Investigate the **4 detection-hard images** (empty on both detectors) | open |
| 5 | Blur/sharpness gate (variance-of-Laplacian) to kill confident-but-wrong reads | not built |
| 6 | Validated serial-format regex for multi-number crops | not built |
| 7 | Final decision on size approach (FP16 vs INT8 vs 4 MB) | pending from user |

---

## 4. Recommended next step
Hand-label a 30–50 image test set (Task #1). Every accuracy/size claim above is currently
measured in *confidence*, not verified accuracy — a small labeled set converts all of this
into real precision numbers and lets the FP16/INT8/4 MB tradeoffs be judged objectively.
