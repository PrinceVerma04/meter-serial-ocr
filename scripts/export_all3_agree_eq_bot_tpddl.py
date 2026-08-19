#!/usr/bin/env python3
"""
Export the "all 3 models agree AND match bot" subset of the TPDDL July set as a
standalone labeled folder (images + ground_truth.csv), ready to fold into a
retrain (same format as `retraining dataset with label/`: filename, ground truth).

Input:  Some_analysis_part/merged_3models_vs_bot_tpddl.csv
Source images (the crops actually fed to the model):
        Some_analysis_part/1stjuly_4thjuly_2026_tpddl/

Output: Some_analysis_part/all3_agree_eq_bot_tpddl/
            images/            copies of the matching crops
            ground_truth.csv   filename, ground truth   (label = bot_serial,
                                which equals what all 3 models predicted)

Usage:
    python3 scripts/export_all3_agree_eq_bot_tpddl.py
"""
import csv
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "Some_analysis_part")
SRC_CSV = os.path.join(BASE, "merged_3models_vs_bot_tpddl.csv")
SRC_IMG_DIR = os.path.join(BASE, "1stjuly_4thjuly_2026_tpddl")

OUT_DIR = os.path.join(BASE, "all3_agree_eq_bot_tpddl")
OUT_IMG_DIR = os.path.join(OUT_DIR, "images")
OUT_CSV = os.path.join(OUT_DIR, "ground_truth.csv")


def main():
    os.makedirs(OUT_IMG_DIR, exist_ok=True)

    with open(SRC_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def is_match(r):
        return r["all_agree"] == "yes" and r["fp32_eq_bot"] == "1"

    matches = [r for r in rows if is_match(r)]
    print(f"matched rows: {len(matches)} / {len(rows)}")

    copied, missing = 0, 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "ground truth"])
        for r in matches:
            name = r["image_name"]
            label = r["bot_serial"]
            src = os.path.join(SRC_IMG_DIR, name)
            dst = os.path.join(OUT_IMG_DIR, name)
            if not os.path.exists(src):
                missing += 1
                continue
            shutil.copy(src, dst)
            copied += 1
            w.writerow([name, label])

    print(f"copied images: {copied}  (missing source files: {missing})")
    print(f"wrote {OUT_CSV}")
    print(f"wrote images to {OUT_IMG_DIR}")


if __name__ == "__main__":
    main()
