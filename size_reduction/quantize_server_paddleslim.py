#!/usr/bin/env python3
"""
Quantize the accurate SERVER model (finetune/inference_v2, 81 MB) to INT8 using
PaddleSlim — the in-Paddle route (no ONNX), since paddle2onnx crashes on this
architecture in this environment.

  finetune/inference_v2  (FP32, 81 MB)  ->  server_int8_paddle/  (~20 MB, INT8)

IMPORTANT — this is the LAST untried route and may still fail:
  * PaddleSlim may not support PP-OCRv5's new PIR (inference.json) format.
  * Paddle INT8 inference on THIS box's CPU hits the oneDNN bug; run on GPU.
Your model is backed up at finetune/inference_v2_backup_2026-07-13/ (untouched).

Install first:  pip install paddleslim
Run:            python3 size_reduction/quantize_server_paddleslim.py
"""
import os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SRC = "finetune/inference_v2"
BK  = "finetune/inference_v2_backup_2026-07-13"
OUT = "size_reduction/models/server_int8_paddle"

# --- safety: ensure a backup of the source model exists before we touch anything ---
if not os.path.isdir(BK):
    shutil.copytree(SRC, BK)
    print(f"created backup: {BK}")
else:
    print(f"backup already present: {BK}")

# --- numpy shim ---
import numpy as np
for _a in ("float", "int", "bool", "object", "str"):
    if not hasattr(np, _a):
        setattr(np, _a, getattr(__builtins__, _a, None))

import paddle
paddle.enable_static()          # quant_post_dynamic uses the static-graph loader
try:
    from paddleslim.quant import quant_post_dynamic
except Exception as e:
    sys.exit(f"paddleslim not available: {e}\n  -> pip install paddleslim")

os.makedirs(OUT, exist_ok=True)
mb = lambda p: os.path.getsize(p) / 1e6 if os.path.exists(p) else 0
print(f"source FP32 weights: {mb(SRC + '/inference.pdiparams'):.1f} MB")

# Dynamic post-training quantization: weights -> int8, no calibration data needed.
try:
    quant_post_dynamic(
        model_dir=SRC,
        save_model_dir=OUT,
        model_filename="inference.json",       # PIR format (may be unsupported)
        params_filename="inference.pdiparams",
        weight_bits=8,
        # PaddleSlim dynamic quant supports conv2d + mul (FC) only; matmul (the
        # SVTR/NRTR attention ops) is NOT quantizable and must be excluded.
        quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
    )
except Exception as e:
    sys.exit(f"\nquant_post_dynamic FAILED (likely PIR-format unsupported): {str(e)[:300]}\n"
             f"=> The server model cannot be quantized in this environment. "
             f"Your original stays intact at {SRC} and {BK}.")

# report resulting size
tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 1e6
print(f"\n=== DONE ===\n  INT8 model dir: {OUT}  ({tot:.1f} MB total)")
print("  Verify accuracy before using it — INT8 can lose 1-3% (and more if the "
      "op coverage is partial).")
