#!/usr/bin/env python3
"""Waits for the new-PTQ scoring run to finish, then auto-saves a results summary
(summary .txt in the folder + appends to the work log). No manual step needed."""
import os, csv, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
D = "26788 images for test/agreed_fp32_oldptq_qat"
CSV = os.path.join(D, "new_ptq_vs_groundtruth.csv")
GT = os.path.join(D, "result.csv")
TARGET = sum(1 for _ in open(GT)) - 1        # expected rows

# wait until the CSV stops growing and reaches the target
last, stable = -1, 0
while True:
    n = (sum(1 for _ in open(CSV)) - 1) if os.path.exists(CSV) else 0
    if n >= TARGET:
        break
    stable = stable + 1 if n == last else 0
    if stable >= 6 and n > 0:      # no growth for ~3 min -> assume finished/stalled
        break
    last = n
    time.sleep(30)

rows = list(csv.DictReader(open(CSV)))
n = len(rows)
correct = sum(int(r["match"]) for r in rows)
acc = 100 * correct / n if n else 0

# character-closeness of the misses
def lev(a, b):
    if a == b: return 0
    m, k = len(a), len(b)
    if not m: return k
    if not k: return m
    prev = list(range(k + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * k
        for j in range(1, k + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[k]
d = [lev(r["ground_truth"], r["new_ptq_pred"]) for r in rows]
within1 = sum(1 for x in d if x <= 1)
within2 = sum(1 for x in d if x <= 2)
blank = sum(1 for r in rows if r["new_ptq_pred"].strip() == "")

summary = f"""NEW PTQ INT8 (v3) — accuracy vs consensus ground truth
======================================================
Ground truth : result.csv  (serial agreed by FP32 + old PTQ v2 + QAT)
Model tested : new PTQ INT8 v3  (size_reduction/models/meterocr_v3_int8_ptq.onnx)
Images scored: {n}

EXACT-MATCH ACCURACY : {acc:.2f}%   ({correct}/{n})
within 1 character   : {100*within1/n:.2f}%
within 2 characters  : {100*within2/n:.2f}%
wrong by 3+ chars    : {100*(n-within2)/n:.2f}%
blank/no-read        : {blank} ({100*blank/n:.2f}%)

Per-image results    : new_ptq_vs_groundtruth.csv
Saved                : {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
open(os.path.join(D, "new_ptq_accuracy_summary.txt"), "w").write(summary)

with open("WORKLOG_2026-07-25_v3_ptq_requant.md", "a") as f:
    f.write("\n\n---\n\n## New PTQ vs consensus ground truth (auto-saved)\n\n")
    f.write(f"Ground truth = the 18,262 images where FP32 + old PTQ + QAT agreed "
            f"(`agreed_fp32_oldptq_qat/result.csv`). New PTQ v3 run on those images:\n\n")
    f.write(f"- **Exact-match accuracy: {acc:.2f}%** ({correct}/{n})\n")
    f.write(f"- within 1 char: {100*within1/n:.2f}% · within 2: {100*within2/n:.2f}% · blank: {blank}\n")
    f.write(f"- Files: `new_ptq_vs_groundtruth.csv`, `new_ptq_accuracy_summary.txt`\n")

print("SAVED. accuracy = %.2f%% (%d/%d)" % (acc, correct, n))
print(summary)
