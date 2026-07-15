# Pruning + QAT — get below 21 MB while recovering accuracy

Goal: a model **smaller than the 21 MB INT8** (pruning removes parameters) **and**
as accurate as possible (QAT recovers the quantization loss).

**Nothing here touches your originals.** Protected & untouched:
- `finetune/inference_v2/` (81 MB original, 6,709-img model)
- `finetune/inference_v2_backup_2026-07-13/`
- `size_reduction/models/server_int8.onnx` (21 MB INT8)
- all training logs + `size_reduction/compat_env/output/` checkpoint

All new outputs land in `size_reduction/prune_qat/`.

## ⚠️ Prerequisites before running
1. **Free disk to ≥ 5 GB.** You're at ~2 GB (100% full). Pruning + QAT need:
   PaddleSlim (~0.4 GB) + pruned checkpoints (~0.6 GB) + QAT checkpoints (~0.6 GB)
   + temp. The 320 GB drive is 98% full with non-project data — free several GB there.
2. GPU Paddle 3.0.0 venv (already built at `size_reduction/compat_env/venv`).

## The pipeline (run in order, in the compat venv)

```bash
cd "/home/ganesh/Desktop/Prince number detection in meter"
ENV=size_reduction/compat_env/venv
$ENV/bin/pip install --no-cache-dir paddleslim==2.6.0     # QAT+prune toolkit

# STEP 1 — PRUNE: sensitivity analysis, then retrain a slimmed backbone
bash size_reduction/prune_qat/1_prune.sh                  # ~20-30 min GPU

# STEP 2 — QAT: quantization-aware fine-tune of the PRUNED model
bash size_reduction/prune_qat/2_qat.sh                    # ~15 min GPU
#   -> exports a quantized inference model directly

# STEP 3 — convert + measure
bash size_reduction/prune_qat/3_export_convert.sh         # -> ONNX INT8 + size
```

## Honest expectations
- **Size:** pruning 30–50% of channels → ~**10–14 MB** INT8.
- **Accuracy:** QAT recovers most of the quant loss, but *pruning itself* costs
  accuracy. Net likely **−5 to −15%** vs the 81 MB model, depending on prune ratio.
- **Uncertainty:** PaddleOCR's slim tools were built for older versions; the PP-OCRv5
  HGNet/SVTR backbone + Paddle 3.0.0 may need tweaks. Expect some iteration.

## If it doesn't converge / tooling fails
Your 21 MB INT8 and 81 MB original are the safe fallbacks — nothing is lost.
