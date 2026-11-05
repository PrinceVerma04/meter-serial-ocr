# Meter Serial-Number OCR — Project Summary

**Project:** Serial Meter Detection (Coral Innovation, IT and Development) · **Owner:** Prince Verma
**Last updated:** 2026-11-02

**Goal:** read the serial number printed on an electricity-meter nameplate from an
already-cropped photo of the serial area.

**Approach:** fine-tune the **PaddleOCR PP-OCRv5 server recognizer** (recognition only — the
crop is already the serial region, so no text detector), add a **confidence gate** (reads with
confidence ≥ 0.95 are auto-accepted, the rest go to human review), and ship a **PTQ INT8 ONNX**
model that runs on CPU without Paddle.

---

## 1. Versions at a glance

| Version | Date | Start weights | Training data | Key result | Status |
|---|---|---|---|---|---|
| v1 | 2026-07-07 | stock | 1,144 TPWODL (773/175/196) | held-out 85.20% → **87.76%** | deleted |
| v2 | 2026-07-13 | stock | 6,709 (4,643/1,098/968) | val 0.874 → **0.9107**; FP32 94.7% / QAT 92.8% / PTQ 85.3% on 414 readable | deleted |
| **v3** | 2026-07-24 (PTQ 07-25) | stock | **6,733** (4,793/916/1,024) — `datasets/v3_train` | PTQ **93.14%** on the 18,544-image consensus set | **deployed (PTQ INT8)** |
| **v4** | 2026-11-02 | **v3** | **23,707** (18,402/2,601/2,704) — `datasets/v4_train` | old test 87.2% → **90.0%**, hard images 11.7% → **90.6%** (PTQ vs PTQ) | candidate |
| **v4.2** | 2026-11-05 | **v3** | **24,434** (19,008/2,648/2,778) — `datasets/v4_train` incl. 727 `nov4_reviewed` | FP32 vs v4 FP32: old test 90.82 → **91.31%**, hard 91.81 → **92.40%**, new 98.99 → 98.81% | FP32 done, PTQ not yet |

Where everything is: models `models/`, data `datasets/` (with `datasets/README.md` = which
version used which data and where every label came from), results `results/`.

---

## 2. History

### v1 — first fine-tune (July)
- Pre-study 2026-07-06 on TPWODL: Paddle best (65% / 89.1%), TrOCR 41% (dropped), EasyOCR compared.
- 1,144 hand-labelled TPWODL crops, split by capture session 773/175/196; Adam, cosine LR **2e-5**
  (3e-4 hurt), 20 epochs, best epoch 9 (val 0.874).
- Same 196 held-out crops: stock **85.20% → 87.76%**; fixed faint-prefix misreads (LW → LN/LA/LH).
- TPDDL cross-utility test (3,460, never trained on): ~95.4% overall, ~98.0% accept precision.
- ROI blind test (4,286, all human-reviewed): 90.7% on 3,325 readable, **96.8% accept precision**;
  961 not-visible photos; 309 human corrections captured.

### v2 — retrain on 6,709 labels (July)
- ROI review labels merged into the label bank (`datasets/v1_v2_labels/ground_truth.csv`,
  8,434 rows / ~6,709 readable) → 6,709 training labels, 0 session leakage.
- Retrained 2026-07-13: val 0.874 → **0.9107** (best epoch 7). A mobile model (7.5 MB) scored
  only 9% on new data and was rejected.
- Quantization 2026-07-15: PTQ INT8 21 MB and QAT INT8 22 MB (val 0.9117). Head-to-head on
  500 TPCODL/TPSODL crops (414 readable): **FP32 94.7%, QAT 92.8%, PTQ 85.3%**.
  First deploy package `deploy/ptq_int8/`.

### v3 — retrain on 6,733 labels (late July) — deployed
- 6,733 labels (4,793/916/1,024; 74% serials with letters), from stock weights, LR 2e-5, 20 epochs
  (`training/rec_v3.yml`).
- PTQ redone 2026-07-25 with the fixed recipe (§4): per-tensor, 200 random calibration crops.
  v3 PTQ scored **93.14%** exact match on the 18,544-image corrected consensus set.
- Deployed: `deploy/ptq_int8/` and `deploy/meterocr_ptq_py38_v3.zip` (Python 3.8 Flask server,
  systemd, port 8080).

### Clean test set and review (27–30 October)
- **Clean test set** (`datasets/test/clean_test_oct27`): 21,042 sharp crops where EasyOCR read the
  department serial exactly — all of them misread by the department's old OCR.
- v3 vs the department serial: **FP32 83.0%, PTQ 79.6%**; accepted reads 85.0% / 84.6% right.
  FP32 and PTQ gave the same read on 18,803 (89.4%); 793 confident disagreements.
- **Filter test** (contrast stretch, Sobel): no gain (contrast 82.2% / 78.4%, Sobel 71.4% / 63.6%)
  → keep raw crops; errors come from the model, not image quality.
- **Manual review** (30 Oct): every image FP32 or PTQ got wrong (4,744) reviewed on a LAN dashboard
  by Prince Verma (1,756), Sathwik (1,557), Srinath Battu (1,431):
  1,771 department serial confirmed, 36 corrected, **2,937 unreadable (62%)**.
  → department serials are ~98% right; most "wrong" reads are images people can't read either.
  On the 18,105 scorable images: **v3 FP32 93.33%, v3 PTQ 91.04%** (accept precision 94.29% / 93.49%).
- Results: `results/2026-10-27_v3_clean_test/` (CSVs + `v3_model_test_results.pptx`).

### v4 — v3 retrained on the reviewed data (2 November)
- **Data** (`datasets/v4_train`, 23,707): v3's 6,733 + 16,974 new crops from the clean test set =
  1,807 human-reviewed (the images v3 misread) + 15,167 where department serial, EasyOCR, v3 FP32
  and v3 PTQ all agreed with confidence ≥ 0.95. New images split 80/10/10 by session.
- **Start:** v3 FP32 — its inference export was mapped back to training weights (all 458 weights
  match by layer name; checked identical reads on 64 images) → `models/v3/meterocr_v3_train_weights.pdparams`.
- **Training** (`training/train_v4.py`, one auto-resuming command): 20 epochs, batch 128, LR 4e-5
  (2e-5 × √(128/32)), 16 loader workers; RTX 5080, **22 min**, 14.3/16 GB GPU memory.
  Best val acc **96.23% at epoch 16** (94.85% after epoch 1).
- **PTQ** (`quantization/quantize_v4_ptq.py`): same recipe as v3, calibrated on 200 v4 training
  crops → 21.4 MB. v4 survives quantization better: FP32/PTQ agreement **98.6% vs v3's 93.3%**.

| Test (deploy pipeline, exact match) | v3 FP32 | v3 PTQ (deployed) | v4 FP32 | v4 PTQ |
|---|---|---|---|---|
| test_old — 1,024 v3 test images | 91.02% | 87.21% | 90.82% | **90.04%** |
| test_new — 1,680 new held-out | 93.39% | 91.01% | 98.99% | **98.81%** |
| test_new_hard — 171 human-reviewed | 35.09% | 11.70% | 91.81% | **90.64%** |
| accept precision on test_old | 93.92% | 93.48% | 93.06% | 92.38% |

Caveats: `test_new`'s 1,509 easy images were selected because v3 read them right, and the 171 hard
ones because v3 misread them, so `test_old` is the neutral comparison (v4 PTQ +29 right, but 74 vs
58 confidently-wrong reads). Details: `results/2026-11-02_v4/`.

---

### v4.2 — v4 recipe + reviewed 4-Nov crops (4–5 November)
- **Test (4 Nov):** all models on `new testing data 4 nov/clean` (2,409 crops, local only). 1,528 of them
  turned out to be in v4's training data, so only the 881 unseen ones are a fair test. Scripts
  `scripts/step1_run_all_models.py` (reads only) → `step2_check_accuracy.py` (scoring) → `step3_build_dashboard.py`;
  reports in `results/2026-11-04_all_models_test/` (dashboard HTML local only, it embeds the images).
- **Review:** three rounds on local review pages (`scripts/review_round2_server.py`, `review_round3_server.py`),
  applied by `scripts/apply_review_2026_11_04.py` + `build_v4_retrain_data.py`; 727 crops merged into
  `datasets/v4_train` with `label_source = nov4_reviewed` (train 606 / val 47 / test 74, split by meter).
- **Try 1** (4 Nov): fine-tuned from v4, 8 epochs → best val at epoch 1, then drifted; slightly worse than v4
  on every test set. Kept locally as `models/v4.2_try1/`.
- **v4.2** (5 Nov, `training/train_v4_2.py`): the v4 recipe — start from v3, 20 epochs, batch 128,
  LR 4e-5, new train crops ×3. Best val **96.00% at epoch 10**, ~25 min on the RTX 5080.
  FP32, training-eval pipeline (`training/output_v4.2/eval_v3_v4_v4.2.txt`):

| test set | v3 | v4 | v4.2 |
|---|---|---|---|
| test_old (1,024, 74% letter serials) | 91.02% | 90.82% | **91.31%** |
| test_new (1,680) | 93.39% | **98.99%** | 98.81% |
| test_new_hard (171) | 35.09% | 91.81% | **92.40%** |
| test_nov4 (74, new 4-Nov crops) | 98.65% | 98.65% | **100.00%** |

- Not yet: PTQ INT8 + deploy-pipeline scoring (`quantization/quantize_v4_2_ptq.py`).

## 3. Data (summary — full detail in `datasets/README.md`)

| Folder | Images | Role |
|---|---|---|
| `datasets/v3_train/` | 6,733 | v3 training data (exact labels + split) |
| `datasets/v4_train/` | 23,707 | v4 training data (exact labels + split + label source) |
| `datasets/v1_v2_labels/` | labels only | 8,434-row label bank used by v1/v2 |
| `datasets/test/clean_test_oct27/` | 21,042 | department serial + human review; 16,974 later used to train v4 |
| `datasets/test/final_test/` | 26,787 | labelled test set (3,014 also in v3/v4 training) |
| `datasets/test/tpddl/` | 3,460 | second utility, never trained on |
| `datasets/test/tpcodl_human_verified/` | 712 | human-verified TPCODL, never trained on |
| `datasets/test/tpddl_july_2026/` | 6,153 | hand-reviewed TPDDL July crops, never trained on by v3/v4 |

---

## 4. Quantization recipe (PTQ INT8, works for v3 and v4)
1. Export the training weights with a **Paddle 3.0.0** env (PIR v1). Paddle 3.3.1 exports PIR v4,
   which `paddle2onnx 2.1.0` can't read.
2. `paddle2onnx` → ONNX FP32, then fold constants in Python with `quant_pre_process` (the space in
   the project path breaks paddle2onnx's own folding). Drops the unused NRTR head: 205 → 84 MB.
3. `quantize_static`: QDQ, **per-tensor** (per-channel truncated long serials, ~47% agreement),
   Conv/MatMul/Gemm, **random** calibration crops (not one capture batch). → 21 MB, ~18 ms/crop on
   one CPU thread vs 253 ms FP32.

---

## 5. Deployment
- **In production:** v3 PTQ INT8 — `deploy/ptq_int8/` (ONNX Runtime only: model + dictionary +
  `meterocr.py` with preprocessing, CTC decode and the 0.95 gate) and `deploy/meterocr_ptq_py38_v3.zip`
  (Python 3.8 server variant: pinned onnxruntime 1.16.3, Flask app, systemd, Docker).
- **To switch to v4:** use `models/v4/meterocr_v4_int8_ptq.onnx` in the same package (same input,
  output, dictionary and gate). Not done yet — a decision for the team.

---

## 6. Lessons learned
- LR 2e-5 for fine-tuning this recognizer; 3e-4 made it worse.
- The confidence gate carries the precision: accepted reads are 93–98% right on every set.
- Image filters (contrast, Sobel) don't help; the remaining errors are the model's or unreadable photos.
- ~60% of crops the models "get wrong" against department records are unreadable to people too —
  human review before counting errors changes the picture (83% → 93% for v3 FP32).
- Fine-tuning from the previous version with its data mixed in keeps old skills (letter serials)
  while learning the new data.
- PTQ: per-tensor, random calibration, fold constants in Python.

---

## 7. Open items
- v4.2: run `quantization/quantize_v4_2_ptq.py` (PTQ INT8 + deploy-pipeline scores), then decide v4 vs v4.2;
  remove old v4 only with `--remove-old-v4` after that.
- Decide on deploying v4 PTQ; look first at its 74 confidently-wrong reads on `test_old`.
- Test v4 on sets it never saw beyond its own test split: `datasets/test/tpddl/`, `tpcodl_human_verified/`,
  and `datasets/test/tpddl_july_2026/` (6,153, unseen by v3 and v4; easy-biased labels).
- The training/quantization envs (Paddle 3.3.1 GPU and Paddle 3.0.0) and the PaddleOCR repo live
  outside this project, in `/home/coral/Desktop/Prince/Prince number detection in meter/` — keep that folder.
