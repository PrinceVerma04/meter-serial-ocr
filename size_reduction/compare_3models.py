#!/usr/bin/env python3
"""
Run all three models on the same random sample and write one CSV:
  FP32 original (ONNX) · QAT INT8 (Paddle) · PTQ INT8 (ONNX)

Output columns:
  image, image_path,
  fp32_reading, fp32_score, qat_reading, qat_score, ptq_reading, ptq_score

Run in the compat venv (has onnxruntime + paddle inference).
"""
import os, re, csv, glob, random, argparse, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_reader import OnnxRec
from paddle_reader import PaddleRec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
MIN_LEN = 5

FP32_ONNX = "size_reduction/models/_fp32_tmp.onnx"
PTQ_ONNX  = "size_reduction/models/server_int8.onnx"
QAT_DIR   = "size_reduction/prune_qat/qat_inference"


def disp(t, s):
    n = re.sub(r"[^A-Z0-9]+", " ", (t or "").upper()).strip()
    return (n, round(s, 3)) if sum(c.isalnum() for c in n) >= MIN_LEN else ("", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="new dataset for test/tpcodl_tpsodl_roi_images")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", default="size_reduction/models_compare.csv")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.jpg")))
    random.seed(args.seed)
    full = random.sample(files, min(args.n, len(files)))
    end = len(full) if not args.count else min(len(full), args.start + args.count)
    sample = full[args.start:end]
    print(f"{len(files)} imgs; sample {len(full)} (seed {args.seed}); batch [{args.start}:{end}] = {len(sample)}")

    fp32 = OnnxRec(FP32_ONNX)
    ptq  = OnnxRec(PTQ_ONNX)
    qat  = PaddleRec(QAT_DIR)
    print("loaded FP32(onnx), PTQ(onnx), QAT(paddle)")

    rows = []
    for i, f in enumerate(sample, 1):
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        ft, fs = disp(*fp32.read(bgr))
        qt, qs = disp(*qat.read(bgr))
        pt, ps = disp(*ptq.read(bgr))
        rows.append((os.path.basename(f), os.path.abspath(f), ft, fs, qt, qs, pt, ps))
        if i % 25 == 0:
            print(f"  {i}/{len(sample)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a" if args.append else "w", newline="") as fh:
        w = csv.writer(fh)
        if not args.append:
            w.writerow(["image", "image_path", "fp32_reading", "fp32_score",
                        "qat_reading", "qat_score", "ptq_reading", "ptq_score"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
