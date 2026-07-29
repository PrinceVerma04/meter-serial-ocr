#!/usr/bin/env python3
"""
CPU build of the meter-OCR inference-time web app.

Runs all THREE models (FP32, INT8 QAT, INT8 PTQ-ONNX) pinned to the CPU, so the
inference times you see are pure-CPU numbers. This is the counterpart of
app_gpu.py — the ONLY difference between the two files is the device below.

Run:
    python3 webapp/app_cpu.py                 # http://127.0.0.1:5001
    python3 webapp/app_cpu.py --port 8001
    python3 webapp/app_cpu.py --host 0.0.0.0  # reachable from LAN
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import create_app

DEVICE = "cpu"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5001)
    args = ap.parse_args()

    print(f"Loading all 3 models on {DEVICE.upper()} ...", flush=True)
    app = create_app(DEVICE)
    print(f"\n  CPU benchmark ready ->  http://{args.host}:{args.port}\n", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)
