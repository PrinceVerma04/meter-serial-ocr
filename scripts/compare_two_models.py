#!/usr/bin/env python3
"""
Run TWO recognizers (old vs new/retrained) on the same random sample of crops
and write one CSV with both readings per image, for a head-to-head review.

Output CSV columns:
  image, image_path, old_reading, old_score, new_reading, new_score

Usage:
  python3 scripts/compare_two_models.py \
      --data "new dataset for test/tpcodl_tpsodl_roi_images" \
      --old finetune/inference --new finetune/inference_v2 \
      --n 500 --seed 42 --out results/model_compare.csv
"""
import os, re, csv, glob, random, argparse

# --- numpy alias shim (old openpyxl on this box) ---
import numpy as np
for _a in ('float', 'int', 'bool', 'object', 'str'):
    if not hasattr(np, _a):
        setattr(np, _a, getattr(__builtins__, _a, None) if not isinstance(__builtins__, dict)
                else __builtins__.get(_a))
import cv2
import paddle
from paddleocr import TextRecognition

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MIN_LEN = 5
_ON_GPU = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
_kw = dict(device="gpu") if _ON_GPU else dict(enable_mkldnn=False)


def sharpness(path):
    """Variance of the Laplacian — higher = sharper/clearer, lower = blurrier."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return -1.0
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def preprocess(bgr):
    up = cv2.resize(bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def recog(engine, img):
    res = engine.predict(img)
    r = res[0] if res else {}
    d = r if isinstance(r, dict) else getattr(r, "res", r)
    return d.get("rec_text", ""), float(d.get("rec_score", 0))


def display_text(text, score):
    """Same normalization as extract_serials: keep letters/digits + single spaces."""
    norm = re.sub(r"[^A-Z0-9]+", " ", (text or "").upper()).strip()
    if sum(c.isalnum() for c in norm) < MIN_LEN:
        return "", 0.0
    return norm, score


def read(engine, bgr):
    """Production-faithful read: raw first, fall back to preprocessed if raw is empty."""
    t, s = display_text(*recog(engine, bgr))
    if t:
        return t, s
    return display_text(*recog(engine, preprocess(bgr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--old", default="finetune/inference")
    ap.add_argument("--new", default="finetune/inference_v2")
    ap.add_argument("--old-name", default=None, help="model_name for old (e.g. PP-OCRv5_mobile_rec)")
    ap.add_argument("--new-name", default=None, help="model_name for new (e.g. PP-OCRv5_mobile_rec)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sharp", action="store_true",
                    help="select the N CLEAREST images (highest sharpness) instead of a random sample")
    ap.add_argument("--min-sharp", type=float, default=0.0,
                    help="drop images below this sharpness (variance-of-Laplacian) before selecting")
    ap.add_argument("--out", default="results/model_compare.csv")
    ap.add_argument("--start", type=int, default=0, help="process sample[start:start+count]")
    ap.add_argument("--count", type=int, default=0, help="0 = all remaining (for batching)")
    ap.add_argument("--append", action="store_true", help="append to --out without header")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.jpg")))
    if args.sharp:
        print(f"scoring sharpness of {len(files)} images to pick the {args.n} clearest ...", flush=True)
        scored = [(sharpness(f), f) for f in files]
        scored = [sf for sf in scored if sf[0] >= args.min_sharp]
        scored.sort(reverse=True)                          # sharpest (clearest) first
        full = [f for _s, f in scored[:args.n]]
        if scored:
            print(f"  kept {len(scored)} above min-sharp={args.min_sharp}; "
                  f"selected {len(full)} clearest (sharpness {scored[0][0]:.0f} .. "
                  f"{scored[min(len(full),len(scored))-1][0]:.0f})")
    else:
        random.seed(args.seed)
        full = random.sample(files, min(args.n, len(files)))   # the fixed 500-image sample
    end = len(full) if not args.count else min(len(full), args.start + args.count)
    sample = full[args.start:end]
    print(f"{len(files)} images; fixed sample {len(full)} (seed {args.seed}); "
          f"this batch [{args.start}:{end}] = {len(sample)}")

    print("loading OLD model:", args.old)
    old = TextRecognition(model_dir=args.old, **({"model_name": args.old_name} if args.old_name else {}), **_kw)
    print("loading NEW model:", args.new)
    new = TextRecognition(model_dir=args.new, **({"model_name": args.new_name} if args.new_name else {}), **_kw)

    rows = []
    for i, f in enumerate(sample, 1):
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        ot, os_ = read(old, bgr)
        nt, ns_ = read(new, bgr)
        rows.append((os.path.basename(f), os.path.abspath(f),
                     ot, round(os_, 3), nt, round(ns_, 3)))
        if i % 50 == 0:
            print(f"  {i}/{len(sample)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    mode = "a" if args.append else "w"
    with open(args.out, mode, newline="") as fh:
        w = csv.writer(fh)
        if not args.append:
            w.writerow(["image", "image_path", "old_reading", "old_score", "new_reading", "new_score"])
        w.writerows(rows)

    agree = sum(1 for r in rows if r[2].upper() == r[4].upper())
    print(f"\nDone. {len(rows)} images. models agree on {agree} "
          f"({agree/len(rows)*100:.1f}%), differ on {len(rows)-agree}. Wrote {args.out}")


if __name__ == "__main__":
    main()
