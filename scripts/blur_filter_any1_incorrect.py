#!/usr/bin/env python3
"""
Blur-filter the 'any 1 model incorrect' TPCODL set (variance of Laplacian).

Step 1 (analyze): compute sharpness for every image in the any1-incorrect set and
print the distribution so a threshold can be chosen.
Step 2 (apply, --threshold T): COPY images with score < T into
'Some_analysis_part/any1_incorrect_blurry/' (originals untouched) and write a
report CSV. Sharp images are left where they are.
"""
import os
import argparse
import shutil
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, "Some_analysis_part")
GT_CSV = os.path.join(ANALYSIS, "1staug_4thaug_2026_serialno_tpcodl.csv")
PRED_CSV = os.path.join(ANALYSIS, "three_model_consensus_tpcodl.csv")
IMG_DIR = os.path.join(ANALYSIS, "1staug_4thaug_2026_serialno_tpcodl")
BLURRY_DIR = os.path.join(ANALYSIS, "any1_incorrect_blurry")
REPORT = os.path.join(ANALYSIS, "any1_incorrect_blur_report.csv")

MODELS = ["fp32", "qat", "ptq"]


def any1_incorrect_names():
    gt = pd.read_csv(GT_CSV)
    gt["image_name"] = gt["readings.KWH.bigImg"].astype(str).str.split("/").str[-1]
    gt = gt.rename(columns={"readings.KWH.bot_annotation.serial_value": "gt"})
    gt["gt"] = gt["gt"].fillna("").astype(str).str.strip()
    pred = pd.read_csv(PRED_CSV)
    for m in MODELS:
        pred[m] = pred[m].fillna("").astype(str).str.strip()
    m = pd.merge(gt[["image_name", "gt"]], pred[["image_name"] + MODELS],
                 on="image_name", how="inner")
    wrong = (m["fp32"] != m["gt"]) | (m["qat"] != m["gt"]) | (m["ptq"] != m["gt"])
    return m[wrong]["image_name"].tolist()


def sharpness(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=None,
                    help="if set, COPY images scoring below this into any1_incorrect_blurry/")
    args = ap.parse_args()

    names = any1_incorrect_names()
    rows = []
    for n in names:
        s = sharpness(os.path.join(IMG_DIR, n))
        rows.append({"image_name": n, "blur_score": s})
    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["blur_score"])
    scores = valid["blur_score"]

    print(f"any1-incorrect images: {len(df)}  (readable: {len(valid)})")
    print("sharpness (variance of Laplacian) percentiles — lower = blurrier:")
    for p in [1, 5, 10, 20, 25, 50, 75, 90]:
        print(f"  p{p:>2} = {np.percentile(scores, p):8.1f}")
    for t in [50, 75, 100, 150, 200]:
        print(f"  below {t:>4}: {(scores < t).sum():>5} images")

    if args.threshold is not None:
        t = args.threshold
        df["removed_blurry"] = df["blur_score"].apply(
            lambda s: (s is not None) and (s < t))
        df.sort_values("blur_score", inplace=True, na_position="first")
        df.to_csv(REPORT, index=False)

        os.makedirs(BLURRY_DIR, exist_ok=True)
        removed = df[df["removed_blurry"]]
        for n in removed["image_name"]:
            src = os.path.join(IMG_DIR, n)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(BLURRY_DIR, n))
        print(f"\nthreshold = {t}")
        print(f"REMOVED (blurry, copied to any1_incorrect_blurry/): {len(removed)}")
        print(f"remaining sharp in any1-incorrect: {len(df) - len(removed)}")
        print(f"report: {REPORT}")


if __name__ == "__main__":
    main()
