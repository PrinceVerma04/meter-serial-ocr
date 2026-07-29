#!/usr/bin/env python3
"""Re-score the New PTQ (v3) model over the 18,544 corrected-consensus dataset.

No model inference — it scores the New PTQ predictions already produced
(column `ptq_int8_new_v3` in four_model_results.csv) against the corrected
ground truth (result.csv), using the fixed rules:
  * comparison is SPACE-INSENSITIVE ('TPWODL BLE' == 'TPWODLBLE')
  * leading zeros are PRESERVED ('02542761' != '2542761')

Reads : 26788 images for test/four_model_results.csv   (new PTQ predictions)
        26788 images for test/agreed_fp32_oldptq_qat/result.csv   (corrected GT)
Writes: 26788 images for test/agreed_fp32_oldptq_qat/new_ptq_vs_groundtruth_corrected.csv
        26788 images for test/agreed_fp32_oldptq_qat/new_ptq_accuracy_summary_corrected.txt
"""
import csv, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "26788 images for test")
CDIR = os.path.join(BASE, "agreed_fp32_oldptq_qat")
FOUR = os.path.join(BASE, "four_model_results.csv")
GT = os.path.join(CDIR, "result.csv")                       # corrected ground truth
OUT = os.path.join(CDIR, "new_ptq_vs_groundtruth_corrected.csv")
SUMMARY = os.path.join(CDIR, "new_ptq_accuracy_summary_corrected.txt")


def norm(s):
    """Comparison key: trim, uppercase, drop spaces. Leading zeros KEPT."""
    return (s or "").strip().upper().replace(" ", "")


def lev(a, b):
    """Levenshtein edit distance between two strings."""
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


# New PTQ predictions keyed by filename
pred = {}
with open(FOUR) as f:
    for r in csv.DictReader(f):
        pred[r["filename"]] = r.get("ptq_int8_new_v3", "")

# score every consensus image against corrected GT
counts = {0: 0, 1: 0, 2: 0, 3: 0}
rows = []
missing = 0
with open(GT) as f:
    for r in csv.DictReader(f):
        fn = r["filename"]; gt = r["result"]
        if fn not in pred:
            missing += 1
            continue
        p = pred[fn]
        d = lev(norm(gt), norm(p))
        counts[d if d < 3 else 3] += 1
        rows.append([fn, gt, p, d, 1 if d == 0 else 0])

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "ground_truth", "new_ptq_pred", "edit_distance", "match"])
    w.writerows(rows)

total = len(rows)
def pct(n): return 100.0 * n / total if total else 0.0
summary = (
    "NEW PTQ (v3) — re-scored vs CORRECTED 3-model consensus ground truth\n"
    "(space-insensitive comparison, leading zeros preserved)\n\n"
    f"total images  : {total}\n"
    f"exact match   : {counts[0]:6d}  ({pct(counts[0]):.2f}%)\n"
    f"1 char off     : {counts[1]:6d}  ({pct(counts[1]):.2f}%)   cumulative <=1: {pct(counts[0]+counts[1]):.2f}%\n"
    f"2 char off     : {counts[2]:6d}  ({pct(counts[2]):.2f}%)   cumulative <=2: {pct(counts[0]+counts[1]+counts[2]):.2f}%\n"
    f"wrong (3+)     : {counts[3]:6d}  ({pct(counts[3]):.2f}%)\n"
)
if missing:
    summary += f"\n(note: {missing} GT rows had no New PTQ prediction and were skipped)\n"

with open(SUMMARY, "w") as f:
    f.write(summary)

print(summary)
print("Wrote:")
print("  ", OUT)
print("  ", SUMMARY)
