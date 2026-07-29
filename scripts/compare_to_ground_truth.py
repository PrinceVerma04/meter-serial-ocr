#!/usr/bin/env python3
"""
Score the 3 meter-OCR models (FP32 / QAT / PTQ) against the hand-checked ground truth.

Inputs
    datasets/final_test/final_results.xlsx        filename, final_results  (ground truth)
    datasets/final_test/three_model_results.csv   filename, fp32_result, qat_result, ptq_result

Outputs
    datasets/final_test/model_comparison.xlsx     one row per scored image
    datasets/final_test/accuracy_summary.xlsx     per-model totals

Rules (as specified):
  - join on filename
  - drop rows whose ground-truth final_results is blank/empty
  - EXACT string match; no normalization, no case/space folding, no brand-prefix
    stripping, no similarity scoring

Run:
    python3 scripts/compare_to_ground_truth.py
"""
import os, csv, argparse

import numpy as np
# numpy-alias shim (old openpyxl in the import chain on this box)
for _a in ("float", "int", "bool", "object", "str"):
    if not hasattr(np, _a):
        setattr(np, _a, getattr(__builtins__, _a, None) if not isinstance(__builtins__, dict)
                else __builtins__.get(_a))
import openpyxl
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

GT_XLSX = "datasets/final_test/final_results.xlsx"
PRED_CSV = "datasets/final_test/three_model_results.csv"
OUT_COMPARE = "datasets/final_test/model_comparison.xlsx"
OUT_SUMMARY = "datasets/final_test/accuracy_summary.xlsx"
MODELS = ["fp32", "qat", "ptq"]


def load_ground_truth(path):
    """{filename: final_results} exactly as stored; blanks kept as '' so they can be counted."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    gt = {}
    for name, value in wb.worksheets[0].iter_rows(min_row=2, values_only=True):
        if name is None:
            continue
        gt[str(name)] = "" if value is None else str(value)
    wb.close()
    return gt


def load_predictions(path):
    with open(path, newline="") as fh:
        return {r["filename"]: r for r in csv.DictReader(fh) if r.get("filename")}


def verify(gt, pred):
    """Report filename coverage across the two files. Returns True when they match 1:1."""
    g, p = set(gt), set(pred)
    missing = sorted(g - p)      # in ground truth, no prediction
    extra = sorted(p - g)        # predicted, not in ground truth

    print("=" * 62)
    print("FILENAME VERIFICATION")
    print("=" * 62)
    print(f"  ground-truth rows (final_results.xlsx) : {len(gt)}")
    print(f"  prediction rows   (three_model_results.csv): {len(pred)}")
    print(f"  matched on filename                    : {len(g & p)}")
    print(f"  in ground truth but NOT in predictions  : {len(missing)}")
    print(f"  in predictions but NOT in ground truth  : {len(extra)}")
    for label, rows in (("missing", missing), ("unmatched", extra)):
        for f in rows[:10]:
            print(f"      [{label}] {f}")
        if len(rows) > 10:
            print(f"      ... and {len(rows) - 10} more")
    ok = not missing and not extra
    print("  RESULT: " + ("every filename matches 1:1 in both files."
                          if ok else "MISMATCH — see the lists above."))
    return ok


def autosize(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"
    for c in ws[1]:
        c.font = Font(bold=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="write the outputs even if filename verification fails")
    args = ap.parse_args()

    gt = load_ground_truth(GT_XLSX)
    pred = load_predictions(PRED_CSV)

    if not verify(gt, pred) and not args.force:
        print("\nAborting: filenames do not match 1:1. Re-run with --force to score anyway.")
        return 1

    # rule 2: drop blank ground truth; scoring set = readable rows present in both files
    blank = [f for f, v in gt.items() if not v]
    scored = sorted(f for f, v in gt.items() if v and f in pred)
    print(f"\n  blank ground truth removed             : {len(blank)}")
    print(f"  images scored                          : {len(scored)}")

    rows, correct = [], {m: 0 for m in MODELS}
    for f in scored:
        truth = gt[f]
        vals = {m: pred[f][f"{m}_result"] for m in MODELS}
        hit = {m: vals[m] == truth for m in MODELS}          # rule 4: exact string match
        for m in MODELS:
            correct[m] += hit[m]
        rows.append([f, truth,
                     vals["fp32"], hit["fp32"],
                     vals["qat"], hit["qat"],
                     vals["ptq"], hit["ptq"],
                     sum(hit.values())])

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "comparison"
    ws.append(["Filename", "Original (final_results)",
               "FP32 Result", "FP32 Correct",
               "QAT Result", "QAT Correct",
               "PTQ Result", "PTQ Correct",
               "Models Correct (0-3)"])
    for r in rows:
        ws.append(r)
    ws.auto_filter.ref = ws.dimensions
    autosize(ws, [52, 24, 22, 12, 22, 12, 22, 12, 20])
    wb.save(OUT_COMPARE)
    print(f"\n  wrote {OUT_COMPARE}  ({len(rows)} rows)")

    total = len(scored)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "summary"
    ws.append(["Model", "Total Images Compared", "Correct Predictions",
               "Incorrect Predictions", "Exact-Match Accuracy (%)"])
    for m in MODELS:
        c = correct[m]
        ws.append([m.upper(), total, c, total - c,
                   round(100.0 * c / total, 2) if total else 0.0])
    autosize(ws, [10, 22, 20, 22, 26])
    wb.save(OUT_SUMMARY)
    print(f"  wrote {OUT_SUMMARY}")

    print("\n" + "=" * 62)
    print(f"ACCURACY (exact string match, {total} images)")
    print("=" * 62)
    for m in MODELS:
        c = correct[m]
        print(f"  {m.upper():<5} {c:>6} correct / {total - c:>6} incorrect   {100.0*c/total:6.2f}%")
    dist = {n: sum(1 for r in rows if r[8] == n) for n in range(4)}
    print("\n  models correct per image: " +
          "  ".join(f"{n}->{dist[n]}" for n in range(4)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
