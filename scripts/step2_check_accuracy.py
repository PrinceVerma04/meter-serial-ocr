#!/usr/bin/env python3
"""
STEP 2 of 2 - check every model's reads (from step 1) against the ground truth.

Inputs:
    results/2026-11-04_all_models_test/model_reads.xlsx         (made by step 1, no ground truth in it)
    new testing data 4 nov/clean_extracted_serials.xlsx          (ground truth: image_name,
                                                                  extracted_serial_number)
Output:
    results/2026-11-04_all_models_test/accuracy_report.xlsx
        sheet "summary"  : per-model accuracy table + comparisons (also printed here)
        sheet "per_image": image_name, ground_truth, then for each model its read and correct yes/no

A read is correct when it equals the ground truth exactly, after upper-casing and removing
spaces (so "ss 1236" = "SS1236"; leading zeros must match: "0552" != "552").
Images with no ground truth row are left out of accuracy and counted separately.

Run from the project root (after step 1 has finished):
    .venv/bin/python scripts/step2_check_accuracy.py
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READS = os.path.join(ROOT, "results", "2026-11-04_all_models_test", "model_reads.xlsx")
TRUTH = os.path.join(ROOT, "new testing data 4 nov", "clean_extracted_serials.xlsx")
ACCEPT = 0.95   # same auto-accept threshold as deploy/ptq_int8/meterocr.py

TITLES = {"v3_fp32": "v3 FP32", "v3_ptq": "v3 PTQ (deployed)", "v4_fp32": "v4 FP32",
          "v4_ptq": "v4 PTQ (candidate)"}


def norm(s):
    return (s or "").strip().upper().replace(" ", "")


def cell_text(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def read_sheet(ws):
    rows = ws.iter_rows(values_only=True)
    head = [cell_text(h) for h in next(rows)]
    return head, [r for r in rows if r and r[0] is not None]


def main():
    import openpyxl
    from openpyxl.styles import Font, PatternFill

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", default=READS, help="model_reads.xlsx from step 1")
    ap.add_argument("--truth", default=TRUTH, help="ground truth Excel")
    ap.add_argument("--out", default=None, help="report xlsx (default: accuracy_report.xlsx next to --reads)")
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.reads)), "accuracy_report.xlsx")
    if not os.path.exists(a.reads):
        sys.exit(f"{a.reads} not found - run step 1 first")

    # ---- model reads (step 1)
    wb = openpyxl.load_workbook(a.reads, read_only=True)
    head, rows = read_sheet(wb["extracted_results"])
    keys = head[1:]
    reads = {cell_text(r[0]): [cell_text(v) for v in r[1:]] for r in rows}
    chead, crows = read_sheet(wb["confidence"])
    conf = {cell_text(r[0]): [float(v or 0) for v in r[1:]] for r in crows}
    wb.close()
    if chead[1:] != keys:
        sys.exit("model_reads.xlsx: confidence sheet columns do not match extracted_results")

    # ---- ground truth (opened only here, in step 2)
    wb = openpyxl.load_workbook(a.truth, read_only=True)
    thead, trows = read_sheet(wb.active)
    wb.close()
    i_name = thead.index("image_name")
    i_gt = [i for i in range(len(thead)) if i != i_name][0]   # extracted_serial_number
    truth = {cell_text(r[i_name]): cell_text(r[i_gt]) for r in trows}

    names = [n for n in reads if truth.get(n, "") != ""]
    no_truth = [n for n in reads if truth.get(n, "") == ""]
    not_read = [n for n in truth if n not in reads]
    n = len(names)
    ok = {n_: [bool(t) and norm(t) == norm(truth[n_]) for t in reads[n_]] for n_ in names}

    # ---- summary
    L = [f"Model accuracy vs ground truth   {time.strftime('%Y-%m-%d %H:%M')}",
         f"model reads : {os.path.relpath(a.reads, ROOT)}  ({len(reads)} images)",
         f"ground truth: {os.path.relpath(a.truth, ROOT)}  (column '{thead[i_gt]}', {len(truth)} rows)",
         f"scored      : {n} images  (no ground truth: {len(no_truth)}, in ground truth but not read: {len(not_read)})",
         ""]
    table = [["model", "right", "wrong", "accuracy %", "accepted (>=0.95)", "accepted right",
              "accepted wrong", "accepted precision %", "review (<0.95)", "blank"]]
    for j, k in enumerate(keys):
        right = sum(ok[x][j] for x in names)
        blank = sum(reads[x][j] == "" for x in names)
        acc = [x for x in names if reads[x][j] and conf[x][j] >= ACCEPT]
        acc_r = sum(ok[x][j] for x in acc)
        review = n - len(acc) - blank
        table.append([TITLES.get(k, k), right, n - right, round(100 * right / max(n, 1), 2), len(acc), acc_r,
                      len(acc) - acc_r, round(100 * acc_r / max(len(acc), 1), 2), review, blank])
    w = [20, 7, 7, 11, 18, 15, 15, 21, 15, 6]
    for row in table:
        L.append("  " + "".join(f"{str(v):<{w[0]}}" if i == 0 else f"{str(v):>{w[i]}}" for i, v in enumerate(row)))
    L.append("")

    for a_, b_ in (("v3_fp32", "v3_ptq"), ("v4_fp32", "v4_ptq")):
        if a_ in keys and b_ in keys:
            ia, ib = keys.index(a_), keys.index(b_)
            same = sum(norm(reads[x][ia]) == norm(reads[x][ib]) for x in names)
            L.append(f"  FP32 vs PTQ same read, {TITLES[a_].split()[0]:<7}{same:>6} of {n}  ({100 * same / max(n, 1):.1f}%)")
    L.append("")
    cand = "v4_ptq"
    if "v3_ptq" in keys and cand in keys:
        i3, i4 = keys.index("v3_ptq"), keys.index(cand)
        cname = TITLES[cand].replace(" (candidate)", "")
        both = sum(ok[x][i3] and ok[x][i4] for x in names)
        only3 = sum(ok[x][i3] and not ok[x][i4] for x in names)
        only4 = sum(ok[x][i4] and not ok[x][i3] for x in names)
        none = n - both - only3 - only4
        L += [f"  v3 PTQ (deployed) vs {cname} (candidate), per image:",
              f"    both right         {both:>6}", f"    only v3 PTQ right  {only3:>6}",
              f"    only {cname} right {only4:>5}", f"    both wrong         {none:>6}",
              f"    total              {n:>6}   ({both} + {only3} + {only4} + {none})", ""]
    all_wrong = sum(not any(ok[x]) for x in names)
    L += [f"  wrong on every model: {all_wrong} images (check these: bad crop or bad ground truth)", "",
          "right + wrong = scored images; accepted + review + blank = scored images.",
          "accepted precision = accepted right / accepted (accepted wrong reads would go out unchecked).",
          "correct = same as ground truth after upper-case and removing spaces."]
    print("\n".join(L))

    # ---- report xlsx
    bold = Font(bold=True)
    red = PatternFill("solid", fgColor="F8D7DA")
    wbo = openpyxl.Workbook()
    ws = wbo.active
    ws.title = "summary"
    for line in L[:4]:
        ws.append([line])
    ws.append([])
    for row in table:
        ws.append(row)
    for c in ws[6]:
        c.font = bold
    ws.append([])
    for line in L[5 + len(table):]:
        ws.append([line.strip()])
    ws.column_dimensions["A"].width = 24
    for col in "BCDEFGHIJ":
        ws.column_dimensions[col].width = 16

    wp = wbo.create_sheet("per_image")
    ph = ["image_name", "ground_truth"]
    for k in keys:
        ph += [f"{k}_result", f"{k}_correct"]
    wp.append(ph)
    for c in wp[1]:
        c.font = bold
    for x in names:
        vals = [x, truth[x]]
        for j in range(len(keys)):
            vals += [reads[x][j], "yes" if ok[x][j] else "no"]
        wp.append(vals)
        for j in range(len(keys)):
            cell = wp.cell(row=wp.max_row, column=4 + 2 * j)
            if not ok[x][j]:
                cell.fill = red
    for x in no_truth:
        wp.append([x, "(no ground truth)"] + sum(([reads[x][j], ""] for j in range(len(keys))), []))
    for row in wp.iter_rows(min_row=2):
        for c in row[1:]:
            c.number_format = "@"
    wp.column_dimensions["A"].width = 52
    wp.freeze_panes = "C2"
    wp.auto_filter.ref = wp.dimensions
    tmp = out + ".tmp.xlsx"
    wbo.save(tmp)
    os.replace(tmp, out)
    print(f"\nsaved {os.path.relpath(out, ROOT)}  (sheets: summary, per_image)")


if __name__ == "__main__":
    main()
