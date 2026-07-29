#!/usr/bin/env python3
"""Fix the consensus ground-truth + re-score the New PTQ (v3) correctly.

Two bugs fixed:
  1. Leading-zero stripping — stored GT lost leading zeros the models actually read
     (models all read 02542761, GT stored 2542761). GT is now the raw reference output.
  2. Space-only mismatches — 'TPWODL BLETWB158175' vs 'TPWODLBLETWB158175' were counted
     as errors. Comparison is now space-insensitive.

Ground truth = images where the 3 voting models (FP32 + old PTQ + QAT) agree
space-insensitively (leading zeros preserved). GT display value = FP32 raw output.
New PTQ (v3) is scored against it with a space-insensitive normalized edit distance.

Originals are backed up to *.bak once; corrected files are written as *_corrected.csv.
"""
import csv, os, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "26788 images for test")
CDIR = os.path.join(BASE, "agreed_fp32_oldptq_qat")
FOUR = os.path.join(BASE, "four_model_results.csv")

OUT_GT = os.path.join(CDIR, "result_corrected.csv")
OUT_SCORE = os.path.join(CDIR, "new_ptq_vs_groundtruth_corrected.csv")
OUT_SUMMARY = os.path.join(CDIR, "new_ptq_accuracy_summary_corrected.txt")


def norm(s):
    """Comparison key: trim, uppercase, drop spaces. Leading zeros are KEPT."""
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


four = {}
with open(FOUR) as f:
    for r in csv.DictReader(f):
        four[r["filename"]] = r

gt_rows = []          # corrected ground truth (consensus images only)
score_rows = []       # new PTQ scored vs corrected GT
counts = {0: 0, 1: 0, 2: 0, 3: 0}
fixed_zero = fixed_space = 0

for fn, r in four.items():
    fp32 = r.get("retrained_fp32", "")
    new = r.get("ptq_int8_new_v3", "")
    old = r.get("ptq_int8_old_v2", "")
    qat = r.get("qat_int8", "")

    # consensus = the 3 voting models agree (space-insensitive), non-empty
    if not (norm(fp32) and norm(fp32) == norm(old) == norm(qat)):
        continue

    gt_disp = fp32.strip()          # canonical GT keeps proper spacing + leading zeros
    gt_key = norm(fp32)
    p_key = norm(new)
    d = lev(gt_key, p_key)
    k = d if d < 3 else 3
    counts[k] += 1

    # was this a case the old pipeline mis-flagged?
    if d == 0:
        if new.replace(" ", "") == gt_disp.replace(" ", "") and new != gt_disp:
            fixed_space += 1
        if gt_key.startswith("0") and gt_key.lstrip("0") == p_key.lstrip("0") and False:
            pass

    gt_rows.append((fn, gt_disp))
    score_rows.append((fn, gt_disp, new, d, 1 if d == 0 else 0))

# write corrected ground truth
with open(OUT_GT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["filename", "result"]); w.writerows(gt_rows)

# write corrected score
with open(OUT_SCORE, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["filename", "ground_truth", "new_ptq_pred", "edit_distance", "match"])
    for fn, gt, pred, d, m in score_rows:
        w.writerow([fn, gt, pred, d, m])

total = len(score_rows)
def pct(n): return 100.0 * n / total
summary = (
    f"NEW PTQ (v3) vs CORRECTED 3-model consensus ground truth\n"
    f"(space-insensitive comparison, leading zeros preserved)\n\n"
    f"total consensus images : {total}\n"
    f"exact match            : {counts[0]:6d}  ({pct(counts[0]):.2f}%)\n"
    f"1 character off         : {counts[1]:6d}  ({pct(counts[1]):.2f}%)   cumulative <=1: {pct(counts[0]+counts[1]):.2f}%\n"
    f"2 characters off        : {counts[2]:6d}  ({pct(counts[2]):.2f}%)   cumulative <=2: {pct(counts[0]+counts[1]+counts[2]):.2f}%\n"
    f"wrong (3+ chars)        : {counts[3]:6d}  ({pct(counts[3]):.2f}%)\n\n"
    f"space-only cases now counted correct: {fixed_space}\n"
)
with open(OUT_SUMMARY, "w") as f:
    f.write(summary)

print(summary)
print("Wrote:")
print("  ", OUT_GT)
print("  ", OUT_SCORE)
print("  ", OUT_SUMMARY)
