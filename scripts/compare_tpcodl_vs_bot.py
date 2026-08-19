#!/usr/bin/env python3
"""
Compare the extracted OCR results of all 3 models (fp32, qat, ptq) against the
bot ground truth, model-wise correct/incorrect counts.

Ground truth : 1staug_4thaug_2026_serialno_tpcodl.csv
               column 'readings.KWH.bot_annotation.serial_value'
               (image_name = last "/"-segment of 'readings.KWH.bigImg')

Extracted    : three_model_consensus_tpcodl.csv
               columns 'fp32', 'qat', 'ptq'

Matched by image_name. Strict exact-match after strip().
Writes one comparison CSV per model under Some_analysis_part/comparison_results/.
"""
import os
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, "Some_analysis_part")

DEFAULT_GT = os.path.join(ANALYSIS, "1staug_4thaug_2026_serialno_tpcodl.csv")
DEFAULT_PRED = os.path.join(ANALYSIS, "three_model_consensus_tpcodl.csv")
DEFAULT_OUT = os.path.join(ANALYSIS, "comparison_results")

MODELS = ["fp32", "qat", "ptq"]


def load_gt(path):
    gt = pd.read_csv(path)
    gt["image_name"] = gt["readings.KWH.bigImg"].astype(str).str.split("/").str[-1]
    gt = gt.rename(columns={"readings.KWH.bot_annotation.serial_value": "actual_value"})
    gt["actual_value"] = gt["actual_value"].fillna("").astype(str).str.strip()
    return gt[["image_name", "actual_value"]]


def evaluate(gt, pred, model, out_dir):
    p = pred[["image_name", model]].copy()
    p[model] = p[model].fillna("").astype(str).str.strip()
    merged = pd.merge(gt, p, on="image_name", how="inner")
    merged = merged.rename(columns={model: "predicted_value"})
    merged["Result"] = (
        merged["actual_value"] == merged["predicted_value"]
    ).map({True: "Correct", False: "Wrong"})

    correct = int((merged["Result"] == "Correct").sum())
    total = len(merged)
    wrong = total - correct
    acc = 100 * correct / total if total else 0.0
    out_file = os.path.join(out_dir, f"{model}_vs_bot_comparison.csv")
    merged.to_csv(out_file, index=False)
    return {"model": model, "total": total, "correct": correct,
            "wrong": wrong, "accuracy": acc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", default=DEFAULT_GT)
    ap.add_argument("--pred", default=DEFAULT_PRED)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gt = load_gt(args.gt)
    pred = pd.read_csv(args.pred)

    print("=" * 64)
    print("ALL 3 MODELS vs BOT GROUND TRUTH")
    print("(GT = readings.KWH.bot_annotation.serial_value)")
    print("=" * 64)
    for model in MODELS:
        r = evaluate(gt, pred, model, args.out)
        print(f"  {r['model']:<6} total={r['total']:>6}  "
              f"correct={r['correct']:>6}  wrong={r['wrong']:>6}  "
              f"acc={r['accuracy']:6.2f}%")
    print("=" * 64)
    print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
