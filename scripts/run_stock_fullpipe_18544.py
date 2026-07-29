#!/usr/bin/env python3
"""Run the STOCK PP-OCRv5 full pipeline (server_det + server_rec, NO fine-tuning)
on the 18,544 corrected-consensus images, scored against corrected GT.

Reuses extract_serials.py's "full" engine (stock server_det + server_rec) and its
best_serial() distiller so the serial extraction matches the project pipeline.
Resumable. Runs foreground; supports --limit N for a quick sample.

Reads : 26788 images for test/agreed_fp32_oldptq_qat/result.csv  (corrected GT)
Images: 26788 images for test/images/<filename>
Writes: 26788 images for test/agreed_fp32_oldptq_qat/stock_fullpipe_scored.csv
"""
import csv, os, sys, time, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "26788 images for test")
CDIR = os.path.join(BASE, "agreed_fp32_oldptq_qat")
SRC = os.path.join(BASE, "images")
GT = os.path.join(CDIR, "result.csv")
OUT = os.path.join(CDIR, "stock_fullpipe_scored.csv")

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cv2                       # noqa: E402
import extract_serials as E      # noqa: E402


def norm(s): return (s or "").strip().upper().replace(" ", "")


def lev(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N images (sample)")
    args = ap.parse_args()

    items = [(r["filename"], r["result"]) for r in csv.DictReader(open(GT))]
    if args.limit:
        items = items[:args.limit]

    done = set()
    if os.path.exists(OUT) and not args.limit:
        done = {r["filename"] for r in csv.DictReader(open(OUT))}
    todo = [(f, g) for f, g in items if f not in done]
    print(f"total={len(items)} done={len(done)} todo={len(todo)}", flush=True)

    E.build_engine("full")       # stock server_det + server_rec (no fine-tuning)

    write_header = not os.path.exists(OUT) or args.limit
    mode_a = "w" if args.limit else "a"
    t0 = time.time()
    with open(OUT, mode_a, newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["filename", "ground_truth", "stock_pred", "score", "edit_distance", "match"])
        for i, (fn, gt) in enumerate(todo, 1):
            bgr = cv2.imread(os.path.join(SRC, fn))
            if bgr is None:
                serial, score = "", 0.0
            else:
                serial, score = E.best_serial(E.run_ocr(bgr))
            d = lev(norm(gt), norm(serial))
            w.writerow([fn, gt, serial, round(float(score), 4), d, 1 if d == 0 else 0])
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  {rate:.2f} img/s  ETA {(len(todo)-i)/rate/60:.1f} min", flush=True)
                f.flush()
    dt = time.time() - t0
    print(f"\ndone {len(todo)} imgs in {dt:.1f}s  ({len(todo)/dt:.2f} img/s)")


if __name__ == "__main__":
    main()
