# Session handoff — Saturday 2026-07-25 → continue Monday 2026-07-27

Task record for today's work so we can pick up cleanly on Monday.

---

## What we did today (all ✅ done unless marked)

1. **Model inventory & confusion cleared.** Established there was no working v3 PTQ —
   only the old v2 PTQ (Jul 15). Yesterday's `scratch_quant/server_v3_int8.onnx` was
   fake-quant (205 MB, FP32 compute).

2. **Re-quantized the latest v3 model correctly → real 21 MB INT8.** Fixed three
   stacked bugs (Paddle-3.0.0 re-export, space-in-path breaks folding → fold in-Python,
   per-tensor not per-channel). Full detail in `WORKLOG_2026-07-25_v3_ptq_requant.md`
   and memory `ptq-quantization-gotchas`. Script: `size_reduction/quantize_v3_ptq.py`.

3. **Cleanup** (~1 GB freed): removed broken `scratch_quant/` v3 artifacts. Kept
   `deploy/ptq_py38/`.

4. **Nomenclature + deploy.** `MODELS/meterocr_v3_int8_ptq_22mb.onnx`,
   `MODELS/meterocr_v3_fp32_original_81mb`; deployed v3 INT8 into `deploy/ptq_int8/`.
   Updated `MODELS/README.md` + `deploy/ptq_int8/README.md`.

5. **Restored the QAT model** from git (`deploy/qat_int8/`, `size_reduction/prune_qat/`).

6. **Ran a 4-model test** on the 26,787-image set `26788 images for test/`
   (no ground-truth labels → agreement vs retrained FP32). Result
   `four_model_results.csv`, columns renamed for clarity:
   `retrained_fp32, ptq_int8_new_v3, ptq_int8_old_v2, qat_int8`.
   Ranking (exact-match vs FP32): **QAT 77.7% > new PTQ 74.1% > old PTQ 72.2%**.

7. **Consensus set** — images where FP32 + old PTQ + QAT all agree (new PTQ excluded):
   **18,262 (68.2%)**. Folder `26788 images for test/agreed_fp32_oldptq_qat/` with
   `accepted_by_3models.csv`, `result.csv` (filename,serial = the ground truth),
   `images/` (18,262 copies), `summary.txt`, and `dashboard.html` (local gallery).

8. **✅ DONE — scored the NEW PTQ vs that consensus ground truth (18,262 imgs).**
   - Script: `scripts/score_new_ptq_vs_groundtruth.py`; auto-saved by `save_new_ptq_results.py`.
   - **Exact-match accuracy: 87.29% (15,940/18,262)** · within 1 char 96.81% · within 2 char 99.28% · wrong 3+ chars 0.72% · blank 2.
   - Results: `agreed_fp32_oldptq_qat/new_ptq_accuracy_summary.txt` + `new_ptq_vs_groundtruth.csv`.
   - Nothing running in background anymore — session fully complete.

## Model reference (sizes / datasets)

| Model | Size | Format | Trained on | Images |
|---|---|---|---|---|
| Retrained FP32 (v3) | 81 MB | Paddle | `finetune/dataset_retrain/` | 6,733 (4793/916/1024) |
| New PTQ INT8 (v3) | 21 MB | ONNX | quantized from v3 FP32 | — |
| Old PTQ INT8 (v2) | 21 MB | ONNX | quantized from v2 | dataset 6,709 |
| QAT INT8 | 21 MB | Paddle | v2-era lineage (pre-v3) | — |

Test set: `26788 images for test/` = **26,787 images**, no labels.

## Where to resume Monday

- **First:** read `new_ptq_accuracy_summary.txt` (auto-saved by then) for the final
  new-PTQ accuracy vs ground truth. Confirm it landed and matches the ~89% preliminary.
- Open the dashboard: `xdg-open "26788 images for test/agreed_fp32_oldptq_qat/dashboard.html"`.
- **Possible next steps discussed but NOT done:**
  - Fresh **v3 QAT** retrain (current QAT is v2-era, not apples-to-apples).
  - Disagreement report on the ~6% where new PTQ differs 3+ chars from FP32.
  - Add the original pre-retrain FP32 as a comparison column if desired.
  - Optionally commit today's changes (nothing committed yet) + purge/keep QAT in git.

## Key files created today
- `WORKLOG_2026-07-25_v3_ptq_requant.md` — detailed technical log
- `SESSION_HANDOFF_2026-07-25.md` — this file
- `scripts/`: `quantize_v3_ptq.py` (in size_reduction/), `test_26788_3models.py`,
  `run_qat_26788.py`, `score_new_ptq_vs_groundtruth.py`, `save_new_ptq_results.py`
- `26788 images for test/`: `four_model_results.csv`, `three_model_results.csv`,
  `qat_results.csv`, and `agreed_fp32_oldptq_qat/` (dashboard + consensus set)
