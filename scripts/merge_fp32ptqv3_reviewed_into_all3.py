#!/usr/bin/env python3
"""
Merge a hand-reviewed CSV (from the tpddl_fp32ptqv3_review.html tool — already
filtered to Keep/Written rows only, Discards excluded by the tool's own export)
into the retrain-ready folder Some_analysis_part/all3_agree_eq_bot_tpddl/.

Input:  tpddl_fp32ptqv3_reviewed.csv   (filename, ground truth)
Source images: Some_analysis_part/1stjuly_4thjuly_2026_tpddl/
Target: Some_analysis_part/all3_agree_eq_bot_tpddl/images/ + ground_truth.csv

Usage:
    python3 scripts/merge_fp32ptqv3_reviewed_into_all3.py [path/to/reviewed.csv]
"""
import csv
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "Some_analysis_part")

REVIEWED_CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "tpddl_fp32ptqv3_reviewed.csv")
SRC_IMG_DIR = os.path.join(BASE, "1stjuly_4thjuly_2026_tpddl")

TARGET_DIR = os.path.join(BASE, "all3_agree_eq_bot_tpddl")
TARGET_IMG_DIR = os.path.join(TARGET_DIR, "images")
TARGET_CSV = os.path.join(TARGET_DIR, "ground_truth.csv")


def main():
    with open(REVIEWED_CSV, newline="", encoding="utf-8") as f:
        reviewed = list(csv.DictReader(f))
    print(f"reviewed rows (already keep/written only): {len(reviewed)}")

    with open(TARGET_CSV, newline="", encoding="utf-8") as f:
        existing = list(csv.DictReader(f))
    existing_names = {r["filename"] for r in existing}
    print(f"already in all3_agree_eq_bot_tpddl: {len(existing_names)}")

    added, skipped_dupe, missing = 0, 0, 0
    new_rows = []
    for r in reviewed:
        name = r["filename"]
        label = r["ground truth"]
        if name in existing_names:
            skipped_dupe += 1
            continue
        src = os.path.join(SRC_IMG_DIR, name)
        if not os.path.exists(src):
            missing += 1
            continue
        shutil.copy(src, os.path.join(TARGET_IMG_DIR, name))
        new_rows.append((name, label))
        existing_names.add(name)
        added += 1

    with open(TARGET_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for name, label in new_rows:
            w.writerow([name, label])

    print(f"added: {added}  |  skipped (already present): {skipped_dupe}  |  missing source image: {missing}")
    print(f"{TARGET_CSV} now has {len(existing_names)} labeled images total")


if __name__ == "__main__":
    main()
