#!/usr/bin/env python3
"""
Side-by-side test of the FULL-size model vs the reduced (INT8) models on the
new dataset. Runs every model that exists, on the same random sample, and writes
one CSV plus a summary (size, agreement with the full model, speed).

Models tested (skipped if missing):
  full_server   finetune/inference_v2            (Paddle, 81 MB)  -- reference
  int8_server   size_reduction/models/server_int8.onnx  (~20 MB, Path A)
  int8_mobile   size_reduction/models/mobile_int8.onnx  (~4 MB,  Path B)

Usage:
  python3 size_reduction/test_compare.py --n 500 --seed 42
"""
import os, re, csv, glob, time, random, argparse

import numpy as np
for _a in ("float", "int", "bool", "object", "str"):
    if not hasattr(np, _a):
        setattr(np, _a, getattr(__builtins__, _a, None))
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
MIN_LEN = 5


def norm(s):
    return re.sub(r"[^A-Z0-9]+", " ", (s or "").upper()).strip()


def disp(text, score):
    n = norm(text)
    return (n, score) if sum(c.isalnum() for c in n) >= MIN_LEN else ("", 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="new dataset for test/tpcodl_tpsodl_roi_images")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="size_reduction/size_test.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.jpg")))
    random.seed(args.seed)
    sample = random.sample(files, min(args.n, len(files)))
    print(f"{len(files)} images; sample {len(sample)} (seed {args.seed})")

    # ---- build the readers that exist ----
    readers = {}          # name -> (read_fn, size_mb)
    # full server (Paddle)
    import paddle
    from paddleocr import TextRecognition
    kw = dict(device="gpu") if (paddle.is_compiled_with_cuda() and
                                paddle.device.cuda.device_count() > 0) else dict(enable_mkldnn=False)
    eng = TextRecognition(model_dir="finetune/inference_v2", **kw)

    def read_full(bgr):
        r = eng.predict(bgr)
        d = r[0] if r else {}
        d = d if isinstance(d, dict) else getattr(d, "res", d)
        return disp(d.get("rec_text", ""), float(d.get("rec_score", 0)))
    readers["full_server"] = (read_full, os.path.getsize("finetune/inference_v2/inference.pdiparams") / 1e6)

    # onnx reduced models
    from size_reduction.onnx_reader import OnnxRec
    for name, path in [("int8_server", "size_reduction/models/server_int8.onnx"),
                       ("int8_mobile", "size_reduction/models/mobile_int8.onnx")]:
        if os.path.exists(path):
            r = OnnxRec(path)
            readers[name] = ((lambda rr: (lambda bgr: disp(*rr.read(bgr))))(r),
                             os.path.getsize(path) / 1e6)

    names = list(readers)
    print("testing models:", ", ".join(f"{n}({readers[n][1]:.1f}MB)" for n in names))

    # ---- run ----
    rows, times = [], {n: 0.0 for n in names}
    for k, f in enumerate(sample, 1):
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        rec = {"image": os.path.basename(f)}
        for n in names:
            t0 = time.time()
            txt, sc = readers[n][0](bgr)
            times[n] += time.time() - t0
            rec[n] = txt
            rec[n + "_score"] = round(sc, 3)
        rows.append(rec)
        if k % 50 == 0:
            print(f"  {k}/{len(sample)}", flush=True)

    # ---- write CSV ----
    cols = ["image"] + sum(([n, n + "_score"] for n in names), [])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- summary: size, speed, agreement with full_server ----
    print("\n================ SIZE-REDUCTION TEST ================")
    print(f"{'model':14}{'size(MB)':>10}{'ms/img':>9}{'agree w/ full':>15}")
    ref = "full_server"
    for n in names:
        agree = sum(1 for r in rows if r[n].upper() == r[ref].upper())
        ms = times[n] / len(rows) * 1000
        pct = f"{agree/len(rows)*100:.1f}%" if n != ref else "(reference)"
        print(f"{n:14}{readers[n][1]:>10.1f}{ms:>9.1f}{pct:>15}")
    print(f"\n{len(rows)} images. CSV: {args.out}")
    print("'agree w/ full' = how often the reduced model outputs the SAME serial "
          "as the full model (higher = less quality lost to compression).")


if __name__ == "__main__":
    main()
