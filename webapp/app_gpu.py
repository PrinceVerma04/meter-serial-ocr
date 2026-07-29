#!/usr/bin/env python3
"""
GPU build of the meter-OCR inference-time web app.

Runs the models pinned to the GPU. Reality (honestly reported in the UI):
  * FP32 original  -> runs on GPU, ~25-30x faster than CPU.
  * INT8 QAT       -> Paddle INT8 kernels need TensorRT; marked unavailable on GPU.
  * INT8 PTQ ONNX  -> ONNX Runtime falls back to CPU (INT8 CUDA unsupported).
INT8 quantization is a CPU-deployment optimization, so for the INT8 models use
app_cpu.py. This file is the counterpart of app_cpu.py — the ONLY difference is
the device below.

Run:
    python3 webapp/app_gpu.py                 # http://127.0.0.1:5002
    python3 webapp/app_gpu.py --port 8002
    python3 webapp/app_gpu.py --host 0.0.0.0  # reachable from LAN
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import create_app

DEVICE = "gpu"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5002)
    args = ap.parse_args()

    print(f"Loading all 3 models on {DEVICE.upper()} ...", flush=True)
    app = create_app(DEVICE)
    print(f"\n  GPU benchmark ready ->  http://{args.host}:{args.port}\n", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)
