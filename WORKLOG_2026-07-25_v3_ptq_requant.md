# Work log — 2026-07-25 · v3 PTQ re-quantization, cleanup, 3-model test

Track record of the session that fixed the v3 quantization, reorganized the model
files under the `meterocr_*` nomenclature, and launched the 26,788-image 3-model test.

---

## 0. Starting question — "which models do we have?"

Full inventory taken (three families):

| Family | Backbone | Example | Notes |
|---|---|---|---|
| Paddle FP32 | mobile/server rec, 81 MB | `finetune/inference_v3/` (v3, latest), `finetune/inference_v2/` | full precision |
| ONNX INT8 21 MB | server, true INT8 | `deploy/ptq_int8/…` (was **v2**, Jul 15) | the shipped/deployable model |
| ONNX server v3 205 MB | server | `scratch_quant/server_v3_*.onnx` (Jul 24) | **yesterday's broken quant** |

**Key finding:** yesterday (Jul 24) did **not** produce a working PTQ. The only real
PTQ was the **Jul-15 / v2** one. Yesterday's `scratch_quant/server_v3_int8.onnx` was
QDQ *fake-quant* — still 205 MB, still FP32 compute (`Conv`, not `QLinearConv`).

---

## 1. Backup — already done

`deploy/ptq_int8_OLD_backup_2026-07-15/` holds the old v2 21 MB PTQ, byte-identical
to what was shipped. Left untouched.

## 2. Re-quantized v3 correctly (the real work)

Yesterday's failure had **three stacked causes**, all fixed and documented in
`size_reduction/quantize_v3_ptq.py`:

1. **Wrong source/format** — `finetune/inference_v3/` is exported by system Paddle
   3.3.1, whose format `paddle2onnx` can't read (SIGABRT). → Re-exported the
   checkpoint (`finetune/output_retrain/best_accuracy`) via **compat_env Paddle 3.0.0**
   → PIR-v1 model that converts.

2. **Space in the project path** (`Prince number detection in meter`) broke
   paddle2onnx's constant-folding subprocess:
   `[W] Constant folding pass failed … No such file or directory: '.../Prince'`.
   So weights stayed as ONNX **Constant nodes**, not initializers → `quantize_static`
   had nothing to quantize → never shrank (root cause of the 205 MB "int8").
   → Fold constants **in-Python** with `onnxruntime … quant_pre_process`
   (`skip_symbolic_shape=True`). Folding also pruned the unused **NRTR head**:
   **205 MB → 81 MB, CTC-only.**

3. **`per_channel=True` corrupted the folded graph** (wrong quant axis → long serials
   collapsed, e.g. `SS11135228 → SS1`; only ~47 % agreement with FP32).
   → **per-tensor** (`per_channel=False`), `QUInt8` activations, `QInt8` weights,
   random-sampled calibration crops → **78 % agreement, long serials preserved.**

**Result:** genuine **21 MB INT8** (real INT8 weights). Reproduce with
`size_reduction/compat_env/venv/bin/python size_reduction/quantize_v3_ptq.py`
(after the one-time export in that script's STEP 0 header).

## 3. Removed unnecessary models (~1 GB freed)

- `scratch_quant/server_v3_int8.onnx` (205 MB, broken fake-quant)
- `scratch_quant/server_v3_fp32.onnx` (205 MB)
- `scratch_quant/inference_v3_v1/` (205 MB, wrong training-format export)
- `scratch_quant/quantize_v3.py` (buggy script) + removed the now-empty `scratch_quant/`
- large regenerable intermediates (unfolded ONNX, convertible Paddle export)
- **Kept** `deploy/ptq_py38/` (a deploy build — not deleted without an explicit yes)

## 4. Nomenclature + references

Scheme: `meterocr_<version>_<precision>_<method>_<size>[.runtime]`

- `MODELS/meterocr_v3_int8_ptq_22mb.onnx` → `size_reduction/models/meterocr_v3_int8_ptq.onnx`
- `MODELS/meterocr_v3_fp32_original_81mb` → `finetune/inference_v3/`
- Deployed v3 INT8 → `deploy/ptq_int8/model/meterocr_ptq_int8.onnx`
- Updated `MODELS/README.md` and `deploy/ptq_int8/README.md`

Produced files:
- `size_reduction/models/meterocr_v3_fp32.onnx` (81 MB, CTC-only FP32, reference)
- `size_reduction/models/meterocr_v3_int8_ptq.onnx` (21 MB, real INT8, deployed)

## 5. QAT status

**Not removed this session — already gone.** `deploy/qat_int8/` and
`size_reduction/prune_qat/` were deleted from the working tree *before* today
(git status `RD`/`D`). Still in git history if ever needed. Today's cleanup only
touched the `scratch_quant/` v3 artifacts.

---

## 6. 3-model test — 26,788 images (no ground truth → agreement comparison)

- Script: `scripts/test_26788_3models.py` (ONNX/CPU, resumable)
- Models: **v3 FP32** (reference) · **v3 INT8** (new PTQ) · **v2 INT8** (old PTQ)
- Output: `26788 images for test/three_model_results.csv`
- Runtime: ~2.5–3 h on CPU (FP32 server model dominates)

### FINAL 4-model results (all 26,787 images; QAT added afterward)

`26788 images for test/four_model_results.csv` (columns: v3_fp32, v3_int8, v2_int8, qat_int8)
Reference = v3 FP32. Figures are agreement with FP32 (no ground-truth labels).

| Model | Exact | ≤1 char | ≤2 char | 3+ char | Blank | Avg len |
|---|---|---|---|---|---|---|
| **QAT INT8 (Paddle)** 🥇 | **77.7%** | 91.1% | 96.1% | 3.9% | 0.5% | 7.91 |
| **v3 INT8 (new PTQ)** 🥈 | 74.1% | 88.6% | 94.0% | 6.0% | 1.5% | 7.85 |
| v2 INT8 (old PTQ) 🥉 | 72.2% | 86.4% | 91.9% | 8.1% | 2.7% | 7.61 |
| v3 FP32 (ref) | — | — | — | — | 0.5% | 7.96 |

- All four identical: 63.5%
- **Ranking QAT > v3 PTQ > v2 PTQ** on every metric.
- New v3 PTQ clearly beats old v2 PTQ (requant paid off). QAT is most accurate but
  Paddle-only + ~2× slower (1.8 vs 3.1 img/s) — v3 PTQ stays the practical deploy choice.
- QAT run: `scripts/run_qat_26788.py` (system Paddle 3.3.1), 242 min → `qat_results.csv`.


---

## New PTQ vs consensus ground truth (auto-saved)

Ground truth = the 18,262 images where FP32 + old PTQ + QAT agreed (`agreed_fp32_oldptq_qat/result.csv`). New PTQ v3 run on those images:

- **Exact-match accuracy: 87.29%** (15940/18262)
- within 1 char: 96.81% · within 2: 99.28% · blank: 2
- Files: `new_ptq_vs_groundtruth.csv`, `new_ptq_accuracy_summary.txt`
