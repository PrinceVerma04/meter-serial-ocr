#!/usr/bin/env python3
"""
Path A — quantize the retrained SERVER recognizer to INT8 via ONNX.

  finetune/inference_v2  (Paddle, 81 MB FP32)
    -> ONNX FP32  (paddle2onnx)          ~81 MB
    -> ONNX INT8  (onnxruntime dynamic)  ~20 MB   <- deployable, ONNX Runtime

Dynamic quantization needs no calibration data (weights -> int8, activations
quantized at runtime). Runs on this box's ONNX Runtime path, avoiding the Paddle
oneDNN INT8 bug noted in the project.

Usage:
  python3 size_reduction/path_a_server_int8/quantize.py
"""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

SRC = "finetune/inference_v2"                       # paddle inference model dir
OUT = "size_reduction/models"
FP32_ONNX = f"{OUT}/server_fp32.onnx"
INT8_ONNX = f"{OUT}/server_int8.onnx"
os.makedirs(OUT, exist_ok=True)


def sh(cmd):
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def mb(p):
    return os.path.getsize(p) / 1e6 if os.path.exists(p) else 0


# 1) Paddle inference model -> ONNX FP32
#    PP-OCRv5 uses the new JSON inference format; paddle2onnx reads model+params.
P2O = os.path.expanduser("~/.local/bin/paddle2onnx")
sh([P2O,
    "--model_dir", SRC,
    "--model_filename", "inference.json",
    "--params_filename", "inference.pdiparams",
    "--save_file", FP32_ONNX,
    "--opset_version", "13",
    "--enable_onnx_checker", "True"])
print(f"ONNX FP32: {mb(FP32_ONNX):.1f} MB")

# 2) ONNX FP32 -> ONNX INT8 (dynamic weight quantization)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(FP32_ONNX, INT8_ONNX, weight_type=QuantType.QInt8)
print(f"ONNX INT8: {mb(INT8_ONNX):.1f} MB")

print("\n=== Path A result ===")
print(f"  server FP32 (paddle) : {mb(SRC+'/inference.pdiparams'):.1f} MB (weights)")
print(f"  server FP32 (onnx)   : {mb(FP32_ONNX):.1f} MB")
print(f"  server INT8 (onnx)   : {mb(INT8_ONNX):.1f} MB   <- deploy this")
print(f"  saved to {INT8_ONNX}")
