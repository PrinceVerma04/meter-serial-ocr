#!/usr/bin/env python3
"""
Drop the blank-ground-truth images from both source files and emit one combined sheet.

An image is "blank" when final_results in the ground-truth workbook is empty — the
image was judged unreadable. Those rows are removed from the ground truth AND their
prediction rows are removed from the model CSV, so both stay row-for-row aligned.

Inputs (never modified)
    datasets/final_test/final_results.xlsx        filename, final_results
    datasets/final_test/three_model_results.csv   filename, fp32_result, qat_result, ptq_result

Outputs
    datasets/final_test/combined_results.xlsx            filename, final_results, fp32, qat, ptq
    datasets/final_test/final_results_no_blanks.xlsx     ground truth, blanks removed
    datasets/final_test/three_model_results_no_blanks.csv  predictions, same rows removed

Values are copied verbatim — no normalization, no prefix stripping.

Run:
    python3 scripts/build_combined_results.py
"""
import os, csv

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
OUT_COMBINED = "datasets/final_test/combined_results.xlsx"
OUT_GT_CLEAN = "datasets/final_test/final_results_no_blanks.xlsx"
OUT_PRED_CLEAN = "datasets/final_test/three_model_results_no_blanks.csv"
MODELS = ["fp32", "qat", "ptq"]


def sheet(wb, title, header, widths):
    ws = wb.active
    ws.title = title
    ws.append(header)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for c in ws[1]:
        c.font = Font(bold=True)
    return ws


def main():
    wb = openpyxl.load_workbook(GT_XLSX, read_only=True, data_only=True)
    gt_rows = [(str(a), "" if b is None else str(b))
               for a, b in wb.worksheets[0].iter_rows(min_row=2, values_only=True) if a is not None]
    wb.close()

    with open(PRED_CSV, newline="") as fh:
        pred = {r["filename"]: r for r in csv.DictReader(fh) if r.get("filename")}

    blanks = [f for f, v in gt_rows if not v]
    keep = [(f, v) for f, v in gt_rows if v]
    no_pred = [f for f, _ in keep if f not in pred]

    print(f"ground-truth rows        : {len(gt_rows)}")
    print(f"blank (unreadable) rows  : {len(blanks)}  -> removed from BOTH files")
    print(f"rows kept                : {len(keep)}")
    if no_pred:
        print(f"WARNING: {len(no_pred)} kept rows have no prediction row; skipped: {no_pred[:5]}")
        keep = [(f, v) for f, v in keep if f in pred]

    # 1) combined sheet
    out = openpyxl.Workbook()
    ws = sheet(out, "combined",
               ["filename", "final_results", "fp32_result", "qat_result", "ptq_result"],
               [52, 24, 22, 22, 22])
    for f, truth in keep:
        p = pred[f]
        ws.append([f, truth, p["fp32_result"], p["qat_result"], p["ptq_result"]])
    ws.auto_filter.ref = ws.dimensions
    ws.freeze_panes = "A2"
    out.save(OUT_COMBINED)
    print(f"\nwrote {OUT_COMBINED}  ({len(keep)} rows)")

    # 2) cleaned ground truth
    out = openpyxl.Workbook()
    ws = sheet(out, "Sheet1", ["filename", "final_results"], [52, 24])
    for f, truth in keep:
        ws.append([f, truth])
    ws.freeze_panes = "A2"
    out.save(OUT_GT_CLEAN)
    print(f"wrote {OUT_GT_CLEAN}  ({len(keep)} rows)")

    # 3) cleaned predictions, same rows and same order
    with open(OUT_PRED_CLEAN, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "fp32_result", "qat_result", "ptq_result"])
        for f, _ in keep:
            p = pred[f]
            w.writerow([f, p["fp32_result"], p["qat_result"], p["ptq_result"]])
    print(f"wrote {OUT_PRED_CLEAN}  ({len(keep)} rows)")
    print("\noriginals left untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
