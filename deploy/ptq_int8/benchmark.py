#!/usr/bin/env python3
"""
Benchmark the PTQ INT8 (ONNX) model on one image — result + CPU/GPU timing.

Usage:
    python benchmark.py <image.jpg> [--runs 50]

Reports, for CPU and (if available) GPU:
  * the OCR result
  * first-call latency (includes one-time init)
  * steady-state inference latency (avg over N runs)
  * end-to-end latency including cv2.imread (disk load + inference)
  * throughput (images/sec)

GPU needs `onnxruntime-gpu` installed and a CUDA GPU.
"""
import sys, time, argparse
import cv2
import onnxruntime as ort
from meterocr import MeterOCR


def bench(providers, label, img_path, runs):
    avail = ort.get_available_providers()
    if providers[0] not in avail:
        print(f"\n[{label}] not available (provider {providers[0]} missing; "
              f"have: {avail}). Skipping.")
        return
    ocr = MeterOCR(providers=providers)
    bgr = cv2.imread(img_path)

    t0 = time.time(); text, score = ocr.read(bgr); first = (time.time()-t0)*1000
    t0 = time.time()
    for _ in range(runs): ocr.read(bgr)
    infer = (time.time()-t0)/runs*1000
    t0 = time.time()
    for _ in range(runs): ocr.read(cv2.imread(img_path))
    full = (time.time()-t0)/runs*1000

    print(f"\n[{label}]  result: '{text}'  (score {score:.3f})")
    print(f"   first call        : {first:8.1f} ms")
    print(f"   inference only    : {infer:8.1f} ms/img   ({1000/infer:.0f} img/s)")
    print(f"   load + inference  : {full:8.1f} ms/img")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    print(f"image: {args.image}")
    print(f"onnxruntime providers available: {ort.get_available_providers()}")
    bench(["CPUExecutionProvider"], "CPU", args.image, args.runs)
    bench(["CUDAExecutionProvider", "CPUExecutionProvider"], "GPU (CUDA)", args.image, args.runs)


if __name__ == "__main__":
    main()
