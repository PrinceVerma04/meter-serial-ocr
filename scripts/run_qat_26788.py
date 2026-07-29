#!/usr/bin/env python3
"""
Run the QAT INT8 model (Paddle) on the 26,788-image test set and write
qat_results.csv (filename,qat_int8), resumable. A separate merge step then adds
this as a 4th column to three_model_results.csv.

Run with the SYSTEM python (paddle 3.3.1):
    python3 scripts/run_qat_26788.py
"""
import os, sys, glob, csv, time
sys.path.insert(0, "deploy/qat_int8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

from meterocr import MeterOCR

IMGDIR = "26788 images for test/images"
OUTCSV = "26788 images for test/qat_results.csv"


def main():
    ocr = MeterOCR()
    files = sorted(glob.glob(os.path.join(IMGDIR, "*.jpg")))

    done = set()
    if os.path.exists(OUTCSV):
        with open(OUTCSV) as f:
            done = {r[0] for r in csv.reader(f) if r and r[0] != "filename"}
    new_file = not os.path.exists(OUTCSV)

    t0 = time.time()
    with open(OUTCSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["filename", "qat_int8"])
        n = 0
        for path in files:
            name = os.path.basename(path)
            if name in done:
                continue
            text, _score, _st = ocr.predict(path)
            w.writerow([name, text])
            n += 1
            if n % 200 == 0:
                rate = n / (time.time() - t0)
                left = (len(files) - len(done) - n) / rate if rate else 0
                print(f"  {n} done | {rate:.1f} img/s | ~{left/60:.1f} min left", flush=True)
                f.flush()
    print(f"DONE: {n} new rows in {(time.time()-t0)/60:.1f} min -> {OUTCSV}")


if __name__ == "__main__":
    main()
