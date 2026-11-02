"""Score old model / v3 FP32 / v3 PTQ against the department serial (aug_departgiven).

Inputs:
  datasets/test/clean_test_oct27/labels.csv  -> image_name, department_serial (= aug_departgiven, ground truth),
                                                old_system_read (= aug_extracted_serial, old model)
  results/2026-10-27_v3_clean_test/fp32_vs_ptq_results.csv -> fp32_result, ptq_result, statuses (compare_fp32_ptq.py)
Outputs (results/2026-10-27_v3_clean_test/):
  accuracy_vs_department.csv          per-image results + correct flags
  accuracy_vs_department_summary.txt  summary tables

A read counts as correct only if it equals aug_departgiven exactly
(after trimming leading/trailing spaces).
"""
import os

import pandas as pd

PROJECT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = os.path.join(PROJECT, "results", "2026-10-27_v3_clean_test")
LABELS = os.path.join(PROJECT, "datasets", "test", "clean_test_oct27", "labels.csv")
RESULTS = os.path.join(ROOT, "fp32_vs_ptq_results.csv")
OUT_CSV = os.path.join(ROOT, "accuracy_vs_department.csv")
OUT_TXT = os.path.join(ROOT, "accuracy_vs_department_summary.txt")

MODELS = [("Old model (aug_extracted_serial)", "aug_extracted_serial", None),
          ("v3 FP32 (fp32_result)", "fp32_result", "fp32_status"),
          ("v3 PTQ deployed (ptq_result)", "ptq_result", "ptq_status")]


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "-"


def main():
    gt = pd.read_csv(LABELS, dtype=str, keep_default_na=False).rename(
        columns={"department_serial": "aug_departgiven", "old_system_read": "aug_extracted_serial"})
    gt = gt[["image_name", "aug_departgiven", "aug_extracted_serial"]]
    res = pd.read_csv(RESULTS, dtype=str)
    df = gt.merge(res, on="image_name", how="inner")
    for c in ["aug_departgiven", "aug_extracted_serial", "fp32_result", "ptq_result"]:
        df[c] = df[c].fillna("").str.strip()
    total = len(df)

    lines = [f"Ground truth: aug_departgiven   images scored: {total:,}", "",
             "1) Overall accuracy (exact match with aug_departgiven)",
             f"   {'Model':<36}{'Correct':>9}{'Wrong':>9}{'Accuracy':>10}"]
    for name, col, _ in MODELS:
        df[col + "_correct"] = df[col] == df["aug_departgiven"]
        ok = int(df[col + "_correct"].sum())
        lines.append(f"   {name:<36}{ok:>9,}{total - ok:>9,}{pct(ok, total):>10}")

    lines += ["", "2) Accuracy by confidence status (accepted = conf >= 0.95)",
              f"   {'Model':<36}{'Status':<10}{'Images':>8}{'Correct':>9}{'Wrong':>8}{'Accuracy':>10}"]
    for name, col, st in MODELS[1:]:
        for s in ["accepted", "review", "blank"]:
            sub = df[df[st] == s]
            if len(sub):
                ok = int(sub[col + "_correct"].sum())
                lines.append(f"   {name:<36}{s:<10}{len(sub):>8,}{ok:>9,}{len(sub) - ok:>8,}{pct(ok, len(sub)):>10}")

    f, p = df["fp32_result_correct"], df["ptq_result_correct"]
    lines += ["", "3) FP32 vs PTQ head-to-head",
              f"   Both correct          {int((f & p).sum()):>7,}",
              f"   Only FP32 correct     {int((f & ~p).sum()):>7,}",
              f"   Only PTQ correct      {int((~f & p).sum()):>7,}",
              f"   Both wrong            {int((~f & ~p).sum()):>7,}"]

    same = df["same_result"] == "yes"
    ok_same = int((same & f).sum())
    lines += ["", "4) When FP32 and PTQ give the same result (same_result)",
              f"   same_result = yes : {int(same.sum()):,} images, correct {ok_same:,} ({pct(ok_same, same.sum())})",
              f"   same_result = no  : {int((~same).sum()):,} images, FP32 correct {int((~same & f).sum()):,}, "
              f"PTQ correct {int((~same & p).sum()):,}, both wrong {int((~same & ~f & ~p).sum()):,}"]

    both_acc = (df["fp32_status"] == "accepted") & (df["ptq_status"] == "accepted") & ~same
    lines += ["", f"5) Both accepted but different results: {int(both_acc.sum()):,} images",
              f"   FP32 correct {int((both_acc & f).sum()):,}, PTQ correct {int((both_acc & p).sum()):,}, "
              f"both wrong {int((both_acc & ~f & ~p).sum()):,}"]

    text = "\n".join(lines)
    print(text)
    with open(OUT_TXT, "w") as fh:
        fh.write(text + "\n")
    cols = ["image_name", "aug_departgiven", "aug_extracted_serial", "aug_extracted_serial_correct",
            "fp32_result", "fp32_confidence", "fp32_status", "fp32_result_correct",
            "ptq_result", "ptq_confidence", "ptq_status", "ptq_result_correct", "same_result"]
    df[cols].to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}\nSaved: {OUT_TXT}")


if __name__ == "__main__":
    main()
