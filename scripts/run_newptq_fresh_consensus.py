#!/usr/bin/env python3
"""FRESH foreground re-run of the New PTQ (v3) INT8 model on the corrected
consensus set, scored against the corrected ground truth.

Uses the deployed MeterOCR class (deploy/ptq_int8/meterocr.py) so predictions
match production exactly. Resumable: skips filenames already in the output CSV.

Reads : agreed_fp32_oldptq_qat/result.csv  (filename, corrected GT)
Images: 26788 images for test/images/<filename>
Writes: agreed_fp32_oldptq_qat/new_ptq_scored.csv
        agreed_fp32_oldptq_qat/new_ptq_summary.txt
"""
import csv, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "26788 images for test")
CDIR = os.path.join(BASE, "agreed_fp32_oldptq_qat")
SRC = os.path.join(BASE, "images")
GT = os.path.join(CDIR, "result.csv")           # corrected ground truth
OUT = os.path.join(CDIR, "new_ptq_scored.csv")
SUMMARY = os.path.join(CDIR, "new_ptq_summary.txt")

sys.path.insert(0, os.path.join(ROOT, "deploy", "ptq_int8"))
from meterocr import MeterOCR   # noqa: E402


def norm(s):
    return (s or "").strip().upper().replace(" ", "")


def lev(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def main():
    with open(GT) as f:
        items = [(r["filename"], r["result"]) for r in csv.DictReader(f)]
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {r["filename"] for r in csv.DictReader(f)}
    todo = [(fn, gt) for fn, gt in items if fn not in done]
    print(f"total={len(items)}  done={len(done)}  todo={len(todo)}", flush=True)

    ocr = MeterOCR()
    write_header = not os.path.exists(OUT)
    t0 = time.time()
    with open(OUT, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["filename", "ground_truth", "new_ptq_pred", "score", "edit_distance", "match"])
        for i, (fn, gt) in enumerate(todo, 1):
            p = os.path.join(SRC, fn)
            text, score, _ = ocr.predict(p)
            d = lev(norm(gt), norm(text))
            w.writerow([fn, gt, text, score, d, 1 if d == 0 else 0])
            if i % 500 == 0:
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / rate / 60
                print(f"  {i}/{len(todo)}  {rate:.1f} img/s  ETA {eta:.1f} min", flush=True)
                f.flush()

    # summarize full output
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total = 0
    with open(OUT) as f:
        for r in csv.DictReader(f):
            total += 1
            d = int(r["edit_distance"]); counts[d if d < 3 else 3] += 1
    def pct(n): return 100.0 * n / total if total else 0.0
    s = (
        f"NEW PTQ (v3) — FRESH foreground re-run vs corrected consensus GT\n"
        f"(space-insensitive, leading zeros preserved)\n\n"
        f"total images : {total}\n"
        f"exact match  : {counts[0]:6d}  ({pct(counts[0]):.2f}%)\n"
        f"1 char off    : {counts[1]:6d}  ({pct(counts[1]):.2f}%)   cum <=1: {pct(counts[0]+counts[1]):.2f}%\n"
        f"2 char off    : {counts[2]:6d}  ({pct(counts[2]):.2f}%)   cum <=2: {pct(counts[0]+counts[1]+counts[2]):.2f}%\n"
        f"wrong 3+      : {counts[3]:6d}  ({pct(counts[3]):.2f}%)\n"
    )
    with open(SUMMARY, "w") as f:
        f.write(s)
    print("\n" + s)
    print("Wrote:", OUT, "and", SUMMARY)


if __name__ == "__main__":
    main()
