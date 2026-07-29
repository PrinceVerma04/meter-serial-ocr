#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline batch OCR: read every image in a folder, write one CSV. PYTHON 3.8 COMPATIBLE.

    python3 batch_predict.py --images /path/to/crops --out results.csv
    python3 batch_predict.py --images /path/to/crops --out results.csv --workers 4

Output columns: filename, serial, confidence, status, ms

Use this instead of the HTTP API for bulk jobs (S3 sync -> run -> upload CSV).
"""
from __future__ import absolute_import, division, print_function

import os
import csv
import glob
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

from meterocr import MeterOCR

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of crops")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel threads (ORT run() is thread-safe); 1 = sequential")
    ap.add_argument("--threads", type=int, default=2, help="ORT intra-op threads")
    args = ap.parse_args()

    files = sorted(set(f for pat in EXTS for f in glob.glob(os.path.join(args.images, pat))))
    if not files:
        raise SystemExit("no images found in %s" % args.images)
    print("%d images from %s" % (len(files), args.images))

    ocr = MeterOCR(threads=args.threads)

    def work(path):
        t0 = time.time()
        serial, score, status = ocr.predict(path, threshold=args.threshold)
        return [os.path.basename(path), serial, score, status,
                round((time.time() - t0) * 1000, 1)]

    t0 = time.time()
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "serial", "confidence", "status", "ms"])
        if args.workers > 1:
            pool = ThreadPoolExecutor(args.workers)
            try:
                rows = list(pool.map(work, files))
            finally:
                pool.shutdown()
        else:
            rows = []
            for i, f in enumerate(files, 1):
                rows.append(work(f))
                if i % 200 == 0:
                    el = time.time() - t0
                    print("  %d/%d  (%.0f ms/img)" % (i, len(files), el / i * 1000))
        w.writerows(rows)

    el = time.time() - t0
    acc = sum(1 for r in rows if r[3] == "accepted")
    rev = sum(1 for r in rows if r[3] == "review")
    bln = sum(1 for r in rows if r[3] == "blank")
    print("\nDone in %.1f min (%.0f ms/img)" % (el / 60, el / len(files) * 1000))
    print("  accepted %d  review %d  blank %d" % (acc, rev, bln))
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
