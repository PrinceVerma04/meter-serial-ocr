# Cleanup Log — deleted files reference

**Date:** 2026-07-16
**Reason:** free disk space by removing regenerable / redundant files.
**Result:** freed ~11 GB (disk 3.2 GB → 14 GB free; project 11 GB → 637 MB).

All deleted items are **regenerable** or **redundant** — nothing unique was lost, and
**nothing on GitHub was affected** (the repo only tracked code + docs + deploy packages).

---

## What was deleted (and how to regenerate it)

| # | Deleted path | Size | What it was | How to regenerate |
|---|---|---|---|---|
| 1 | `size_reduction/compat_env/venv/` | 9.0 GB | Paddle 3.0.0 GPU virtualenv (used for quantization/export) | `bash size_reduction/compat_env/1_setup_venv.sh` (then re-add GPU/onnx libs — see notes below) |
| 2 | `size_reduction/prune_qat/qat_output/` | 1.7 GB | QAT training checkpoints (`best_accuracy.pdparams`, `.pdopt`, `latest.*`, `best_model/`) | `bash size_reduction/prune_qat/2_qat.sh` (retrains QAT). **The deployable QAT model was already exported** to `deploy/qat_int8/` and `size_reduction/prune_qat/qat_inference/`, so this is only needed to re-train. |
| 3 | `finetune/inference/` | 81 MB | OLD v1 model (fine-tuned on 1,144 labels, ~87%) — superseded by `inference_v2` | Re-export from a checkpoint, or use `inference_v2` (the better model) |
| 4 | `finetune/inference_backup_2026-07-13/` | 81 MB | Backup copy of the OLD v1 model | redundant (was a copy of #3) |
| 5 | `finetune/inference_v2_backup_2026-07-13/` | 81 MB | Backup copy of the retrained FP32 model | redundant — `finetune/inference_v2/` (the original) is kept |
| 6 | `results/*.html` | ~3.5 MB | Review pages (`roi_review.html`, `tpddl_review.html`, `model_compare.html`, `clear500_compare.html`) | re-run the `scripts/build_*_page.py` / `size_reduction/build_3model_page.py` on the matching CSV |
| 7 | `__pycache__/` (project dirs) | tiny | Python bytecode cache | auto-regenerated on next run |
| 8 | `MODELS/meterocr_v1_fp32_old_81mb` | link | Dangling symlink (pointed to deleted #3) | n/a |

---

## What was KEPT (deliverables & essentials — NOT deleted)

- `deploy/ptq_int8/` — PTQ INT8 ONNX deploy package (21 MB) ✅
- `deploy/qat_int8/` — QAT INT8 Paddle deploy package (22 MB) ✅
- `finetune/inference_v2/` — the **FP32 original** model (81 MB) ✅
- `size_reduction/compat_env/output/best_accuracy.pdparams` — retrained checkpoint (205 MB, needed to re-quantize) ✅
- `size_reduction/models/server_int8.onnx` — PTQ INT8 model (21 MB) ✅
- `size_reduction/prune_qat/qat_inference/` — QAT INT8 inference model (22 MB) ✅
- All `scripts/`, `size_reduction/*.py|*.sh|*.md`, docs, and `results/*.csv` ✅
- The git repo + GitHub remote (unaffected) ✅

---

## Notes for regenerating the venv (item #1), if you ever re-quantize

The venv wasn't just `1_setup_venv.sh` — it also needed these extra manual steps that were
added during the work (for GPU export + ONNX quantization):

1. `bash size_reduction/compat_env/1_setup_venv.sh` — creates venv + Paddle 3.0.0 (CPU) + repo deps.
2. GPU Paddle (for fast retrain/export):
   `venv/bin/pip install --no-deps paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/`
   (plus the `nvidia-*-cu11` libs it depends on).
3. PaddleSlim (for QAT): `venv/bin/pip install paddleslim==2.6.0`.
4. ONNX tools: `venv/bin/pip install paddle2onnx onnx onnxruntime` (for benchmarking on
   CUDA 11.8, use `onnxruntime-gpu==1.18.1` + set `LD_LIBRARY_PATH` to the venv's nvidia libs).
5. Patched slim tools (for PP-OCRv5 NRTR head) live in the PaddleOCR repo at
   `deploy/slim/quantization/quant_v5.py`, `export_v5.py`, `export_qat_standalone.py`.

Full methodology: `PROJECT_SUMMARY.md` (§2A, §0, progress log) and `size_reduction/compat_env/README.md`.
