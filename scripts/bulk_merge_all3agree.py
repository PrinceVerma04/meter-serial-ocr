#!/usr/bin/env python3
"""
Bulk-merge every remaining image in the "All 3 agree" filter (fp32==ptq_v3==ptq_v2)
into Some_analysis_part/all3_agree_eq_bot_tpddl/, using the models' shared value
as the label (same rule as the Keep button on all3agree_live_server.py).

Usage:
    python3 scripts/bulk_merge_all3agree.py
"""
import csv
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "Some_analysis_part")
SRC_CSV = os.path.join(BASE, "merged_3models_vs_bot_tpddl.csv")
IMG_DIR = os.path.join(BASE, "1stjuly_4thjuly_2026_tpddl")
TAKEN_OUT_DIR = os.path.join(BASE, "all3_agree_eq_bot_tpddl")
TAKEN_OUT_IMG_DIR = os.path.join(TAKEN_OUT_DIR, "images")
TAKEN_OUT_CSV = os.path.join(TAKEN_OUT_DIR, "ground_truth.csv")
DISCARDED_CSV = os.path.join(BASE, "discarded_tpddl.csv")


def main():
    with open(SRC_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(TAKEN_OUT_CSV, newline="", encoding="utf-8") as f:
        taken_out = {r["filename"] for r in csv.DictReader(f)}
    with open(DISCARDED_CSV, newline="", encoding="utf-8") as f:
        discarded = {r["filename"] for r in csv.DictReader(f)}

    todo = [r for r in rows if r["all_agree"] == "yes"
            and r["image_name"] not in taken_out
            and r["image_name"] not in discarded]
    print(f"to merge: {len(todo)}")

    added = 0
    with open(TAKEN_OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in todo:
            name = r["image_name"]
            label = r["fp32"]  # == ptq_v3 == ptq_v2 by definition of all_agree
            src = os.path.join(IMG_DIR, name)
            if not os.path.exists(src):
                continue
            shutil.copy(src, os.path.join(TAKEN_OUT_IMG_DIR, name))
            w.writerow([name, label])
            added += 1

    print(f"merged: {added}")


if __name__ == "__main__":
    main()
