# Training / Dataset Ledger

Tracks every retrain: which dataset, how many images, when, and what came out.
Update this file each time a new model version is trained. This is the single
source of truth for "what was this model trained on" — don't rely on memory
or scattered chat history for that.

---

## v1 — first fine-tune
- **Date:** pre-2026-07-06
- **Dataset:** 1,144 labeled images (initial manual set)
- **Result:** ~87% val accuracy
- **Artifact:** `finetune/inference/` (FP32, Paddle) — **deleted** in the
  2026-07-16 cleanup (superseded by v2, regenerable only by retraining from
  scratch — no checkpoint kept).

## v2 — retrain on expanded set
- **Date:** ~2026-07-15
- **Dataset:** `finetune/dataset/` — 6,709 labeled images
- **Result:** 94.7% val accuracy (FP32)
- **Checkpoint kept:** `size_reduction/compat_env/output/best_accuracy.pdparams`
  (214 MB, dated 2026-07-15)
- **Artifacts:**
  - FP32 Paddle (`finetune/inference_v2/`) — **deleted** in disk cleanup, not
    regenerable without re-exporting from the checkpoint above
  - PTQ INT8 ONNX: `size_reduction/models/server_int8.onnx` (21 MB) — **present**
  - Ranked 72.2% agreement vs FP32 reference on the 2026-07-25 26,788-image test

## v3 — latest trained model (current production baseline)
- **Date:** 2026-07-24/25
- **Dataset:** `finetune/dataset_retrain/` — 6,733 labeled images, built by
  `scripts/prepare_retrain_data.py` from `retraining dataset with label/`
  (session-grouped 70/15/15 split: 4,793 train / 916 val / 1,024 test)
- **Config:** `finetune/rec_retrain.yml` — Adam, LR 2e-5 cosine, 20 epochs,
  PP-OCRv5_server_rec (SVTR_HGNet backbone + CTC/NRTR multi-head)
- **Result:** ranked best in the 2026-07-25 26,788-image agreement test
- **Artifacts:**
  - FP32 ONNX: `size_reduction/models/meterocr_v3_fp32.onnx` (84 MB) — **present**
  - PTQ INT8 ONNX: `size_reduction/models/meterocr_v3_int8_ptq.onnx` (21 MB),
    also shipped at `deploy/ptq_int8/model/meterocr_ptq_int8.onnx` — **present**
  - QAT INT8 (Paddle): existed at `deploy/qat_int8/` — **deleted** 2026-08-17,
    not regenerable without retraining QAT from scratch
  - FP32 Paddle checkpoint (`finetune/inference_v3/`) — **deleted**, not
    regenerable without a checkpoint (only the ONNX export above survives)

## Data-collection work between v3 and v4 (2026-08-17 → 2026-08-19)
New TPDDL "1st–4th July 2026" set (7,078 raw crops) hand-reviewed via three
purpose-built local web tools (ports 8811/8812/8813 — see session history),
cross-checked against 3 ONNX models + the bot's own annotation:
- 6,153 images landed in `Some_analysis_part/all3_agree_eq_bot_tpddl/`
- 270 discarded (bad crops / unreadable)
- 655 never finished review — **status unknown, folder was deleted 2026-08-19
  before those were resolved; not counted in v4**

Separately, a TPCODL "1–4 Aug 2026" set (2,193 images, `restart training/`)
was manually reviewed to 1,222 annotations (419 correct + 309 corrected =
728 usable labels) via a local Flask app. **This entire folder was deleted
from disk on 2026-08-19 before being merged into any training set — those
728 labels are lost unless a backup exists.** Do not report this data as
available for v4 or any future version.

## v4 — trained 2026-08-19
- **Dataset:** `retraining dataset with label/` — **12,886 images**
  (v3's 6,733 + the 6,153 newly merged from
  `Some_analysis_part/all3_agree_eq_bot_tpddl/`), split via
  `scripts/prepare_retrain_data.py` (session-grouped 70/15/15): **1,187
  sessions -> 830 train / 178 val / 179 test sessions, 9,123 / 1,840 / 1,923
  images**
- **Merge verified:** 0 filename collisions, every CSV row has a matching
  image file, every image file has a CSV row (see chat log 2026-08-19)
- **Not included:** the 655 still-unreviewed TPDDL images, and the TPCODL
  `restart training/` data (deleted before merge — see above)
- **Trained with:** `scripts/retrain_all.sh` — pretrained base
  `PP-OCRv5_server_rec_pretrained` (same base as v1/v2/v3, not continued from
  v3's weights), config `finetune/rec_retrain_v4.yml` (same recipe as v3:
  Adam, LR 2e-5 cosine, 20 epochs), trained on an RTX 5080 in a fresh venv
  (`finetune/train_env/`, Paddle 3.3.1 GPU from the official cu130 index —
  plain PyPI's `paddlepaddle-gpu` resolves to a stale 2.6.2 build with no
  Blackwell/cudnn support, don't use it)
- **Result:** best checkpoint at **epoch 10, val acc 94.08%**
  (`norm_edit_dis` 0.9900) — close to v2's reported 94.7%, needs an actual
  test-set / cross-model comparison against v3 before treating it as better
  (val accuracy alone isn't the same metric v3 was ranked on)
- **Artifacts:**
  - FP32 Paddle checkpoint: `finetune/output_retrain_v4/best_accuracy.pdparams` (214 MB)
  - FP32 Paddle export: `finetune/inference_v4/` (84 MB)
  - PTQ INT8 ONNX, quantized 2026-08-19 via `scripts/quantize_v4_ptq.sh` +
    `size_reduction/quantize_v4_ptq.py` (same recipe as v3 — re-export via
    Paddle 3.0.0 CPU compat venv for a paddle2onnx-convertible format, fold
    constants in-Python, per-tensor static INT8 calibrated on 200 random
    crops): `size_reduction/models/meterocr_v4_fp32.onnx` (84.1 MB, CTC-only)
    and `size_reduction/models/meterocr_v4_int8_ptq.onnx` (**21.4 MB** —
    matches v3's PTQ size almost exactly)
  - QAT: not yet run
- **Status: not yet evaluated against v3 or deployed.** Still shipping v3
  (`deploy/ptq_int8/`) in production. Next step: score
  `meterocr_v4_int8_ptq.onnx` against v3 on held-out/agreement data before
  considering a swap.

---

## How to add the next entry
When v4 (or later) finishes training, append a new `## vN — ...` section above
with: date, dataset path + image count (from the `prepare_retrain_data.py`
console output), config used, result (val/test accuracy), and where the
exported artifacts landed. Keep old entries — this file is a history, not a
snapshot.
