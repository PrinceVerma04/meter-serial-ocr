#!/usr/bin/env python3
"""
How many reads each model would AUTO-ACCEPT on a folder of crops (confidence >= 0.95, the
production gate). Reads only the output of scripts/step1_run_all_models.py - no ground truth,
no model runs. Accepted is not the same as correct: an accepted read goes out without a human
check, right or wrong.

    .venv/bin/python scripts/accepted_summary.py \
        --reads  "results/2026-11-05_review_23543/model_reads.xlsx" \
        --images "new testing data 4 nov/5 nov testing data/review"

Writes accepted_summary.txt next to model_reads.xlsx and prints it. Images that are also in v4's
training data (datasets/v4_train/labels.csv, not the 4-Nov rows) are counted separately, because
v4 has seen them.
"""
import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = 0.95
TITLES = {"v3_fp32": "v3 FP32", "v3_ptq": "v3 PTQ (deployed)", "v4_fp32": "v4 FP32", "v4_ptq": "v4 PTQ (candidate)"}


def load_reads(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    er = list(wb["extracted_results"].iter_rows(values_only=True))
    cf = list(wb["confidence"].iter_rows(values_only=True))
    wb.close()
    keys = [k for k in er[0][1:]]
    if [k for k in cf[0][1:]] != keys:
        sys.exit("!! the two sheets of model_reads.xlsx list different models")
    conf = {r[0]: r[1:] for r in cf[1:] if r[0] is not None}
    rows = {}
    for r in er[1:]:
        if r[0] is None:
            continue
        rows[r[0]] = {k: (str(r[1 + i] or "").strip(), float(conf[r[0]][i] or 0.0)) for i, k in enumerate(keys)}
    return keys, rows


def status(text, score):
    return "blank" if not text else ("accepted" if score >= GATE else "review")


def norm(s):
    return s.upper().replace(" ", "")


def pct(a, n):
    return f"{100 * a / n:6.2f}%" if n else "    - "


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", required=True, help="model_reads.xlsx written by step1_run_all_models.py")
    ap.add_argument("--images", required=True, help="the image folder step 1 read")
    a = ap.parse_args()

    keys, rows = load_reads(a.reads)
    in_folder = {f for f in os.listdir(a.images) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))}
    missing = in_folder - set(rows)
    if missing:
        sys.exit(f"!! {len(missing)} images of the folder have no reads yet - finish step 1 first")
    names = sorted(in_folder)
    v4_seen = set()
    lab = os.path.join(ROOT, "datasets", "v4_train", "labels.csv")
    if os.path.exists(lab):
        v4_seen = {r["image_name"] for r in csv.DictReader(open(lab, encoding="utf-8"))
                   if r.get("label_source") != "nov4_reviewed"} & in_folder
    groups = [("all images" + ("" if v4_seen else " (none of them is in v4's training data)"), names)]
    if v4_seen:
        groups += [("not in v4 training data", [n for n in names if n not in v4_seen]),
                   ("in v4 training data (v4 has seen them)", [n for n in names if n in v4_seen])]

    L = [f"Auto-accept summary - {os.path.relpath(os.path.abspath(a.images), ROOT)}",
         f"made {time.strftime('%Y-%m-%d %H:%M')} from {os.path.relpath(os.path.abspath(a.reads), ROOT)}",
         f"accepted = model read with confidence >= {GATE} (goes out without a human check); "
         "review = read below the gate; blank = nothing read.",
         "Accepted is NOT the same as correct - no ground truth is used here.", ""]
    for g, ns in groups:
        n = len(ns)
        if not n:
            continue
        L += [f"{g}: {n} images",
              f"  {'model':<22}{'accepted':>18}{'review':>18}{'blank':>16}"]
        for k in keys:
            st = [status(*rows[x][k]) for x in ns]
            c = {s: st.count(s) for s in ("accepted", "review", "blank")}
            L.append(f"  {TITLES.get(k, k):<22}{c['accepted']:>8} ({pct(c['accepted'], n)}){c['review']:>8} ({pct(c['review'], n)})"
                     f"{c['blank']:>6} ({pct(c['blank'], n)})")
        L.append("")

    # agreement between models, on all images
    n = len(names)
    acc = {k: {x for x in names if status(*rows[x][k]) == "accepted"} for k in keys}
    every = set.intersection(*acc.values()) if acc else set()
    none = set(names) - set.union(*acc.values()) if acc else set()
    L += ["How the models overlap (all images):",
          f"  {'accepted by all ' + str(len(keys)) + ' models':<24}{len(every):>7} ({pct(len(every), n)})",
          f"  {'accepted by no model':<24}{len(none):>7} ({pct(len(none), n)})"]
    if every:
        same = sum(len({norm(rows[x][k][0]) for k in keys}) == 1 for x in every)
        L.append(f"    of the 'all {len(keys)}': same serial from every model {same}, different serials {len(every) - same}")
    if "v3_ptq" in acc and "v4_ptq" in acc:
        a3, a4 = acc["v3_ptq"], acc["v4_ptq"]
        both = a3 & a4
        same = sum(norm(rows[x]["v3_ptq"][0]) == norm(rows[x]["v4_ptq"][0]) for x in both)
        L += ["", "Deployed (v3 PTQ) vs candidate (v4 PTQ):",
              f"  accepted by both        {len(both):>7}   (same serial {same}, different serial {len(both) - same})",
              f"  only v3 PTQ accepts     {len(a3 - a4):>7}",
              f"  only v4 PTQ accepts     {len(a4 - a3):>7}",
              f"  neither accepts         {n - len(a3 | a4):>7}",
              f"  total                   {n:>7}"]
    text = "\n".join(L)
    out = os.path.join(os.path.dirname(os.path.abspath(a.reads)), "accepted_summary.txt")
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    os.replace(out + ".tmp", out)
    print(text)
    print(f"\nsaved {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
