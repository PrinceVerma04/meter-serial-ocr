#!/usr/bin/env python3
"""
Copy out the crops on which EVERY model read the same serial (confidence ignored; a blank read
never agrees), with that serial in an Excel file (two columns: image_name, extracted_serial). Rows added to the
xlsx by hand (with their image in images/) are kept when this is re-run.

Reads only saved files (no model runs):
  results/2026-11-05_review_21425/model_reads.xlsx                     (step 1 reads + confidence)
  new testing data 4 nov/5 nov testing data/review_ground_truth.xlsx   (image_name, correct_serial)
Writes (images are copied; on 2026-11-05 the originals were then deleted from review/ by hand):
  new testing data 4 nov/5 nov testing data/all_models_same/images/*.jpg
  new testing data 4 nov/5 nov testing data/all_models_same/all_models_same.xlsx

    .venv/bin/python scripts/export_all_same.py
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEW = os.path.join(ROOT, "new testing data 4 nov", "5 nov testing data")
GATE = 0.95


def norm(s):
    return str(s or "").upper().replace(" ", "").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", default=os.path.join(ROOT, "results", "2026-11-05_review_21425", "model_reads.xlsx"))
    ap.add_argument("--gt", default=os.path.join(REVIEW, "review_ground_truth.xlsx"))
    ap.add_argument("--images", default=os.path.join(REVIEW, "review"))
    ap.add_argument("--out", default=os.path.join(REVIEW, "all_models_same"))
    a = ap.parse_args()

    import openpyxl
    from openpyxl.styles import Font
    wb = openpyxl.load_workbook(a.reads, read_only=True)
    er = list(wb["extracted_results"].iter_rows(values_only=True))
    cf = {r[0]: r[1:] for r in list(wb["confidence"].iter_rows(values_only=True))[1:] if r[0]}
    wb.close()
    keys = list(er[0][1:])
    ws = openpyxl.load_workbook(a.gt, read_only=True).worksheets[0]
    hdr = [str(h or "").strip() for h in next(ws.iter_rows(max_row=1, values_only=True))]
    i_n, i_s = hdr.index("image_name"), hdr.index("correct_serial")
    gt = {r[i_n]: str(r[i_s] if r[i_s] is not None else "").strip() for r in ws.iter_rows(min_row=2, values_only=True) if r[i_n]}

    rows, total = [], 0
    for r in sorted((r for r in er[1:] if r[0]), key=lambda r: r[0]):
        total += 1
        reads = {norm(x) for x in r[1:1 + len(keys)]}
        if len(reads) != 1 or "" in reads:
            continue
        serial = reads.pop()
        conf = [float(c or 0.0) for c in cf[r[0]][:len(keys)]]
        rows.append([r[0], serial, gt.get(r[0], ""), "yes" if norm(gt.get(r[0])) == serial else "no",
                     *[round(c, 4) for c in conf], round(min(conf), 4), "yes" if min(conf) >= GATE else "no"])

    img_dir = os.path.join(a.out, "images")
    missing = [x[0] for x in rows if not os.path.exists(os.path.join(img_dir, x[0])) and not os.path.exists(os.path.join(a.images, x[0]))]
    if missing:
        sys.exit(f"!! {len(missing)} images not found in {a.images} or {img_dir}, e.g. {missing[:3]}")
    os.makedirs(img_dir, exist_ok=True)
    for x in rows:
        dst = os.path.join(img_dir, x[0])
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(a.images, x[0]), dst)

    right = sum(x[3] == "yes" for x in rows)
    acc = sum(x[-1] == "yes" for x in rows)
    # keep rows added to the xlsx by hand (image in images/ but not from the all-same rule)
    xlsx = os.path.join(a.out, "all_models_same.xlsx")
    kept = []
    if os.path.exists(xlsx):
        have = {x[0] for x in rows}
        old = openpyxl.load_workbook(xlsx, read_only=True)
        for r in list(old.worksheets[0].iter_rows(min_row=2, values_only=True)):
            if r and r[0] and r[0] not in have and os.path.exists(os.path.join(img_dir, r[0])):
                kept.append([r[0], r[1]])
        old.close()
    out_rows = sorted([x[:2] for x in rows] + kept, key=lambda x: x[0])
    out = openpyxl.Workbook()
    sh = out.active
    sh.title = "all_models_same"
    sh.append(["image_name", "extracted_serial"])
    for c in sh[1]:
        c.font = Font(bold=True)
    for x in out_rows:
        sh.append(x)
    sh.freeze_panes = "A2"
    sh.auto_filter.ref = sh.dimensions
    sh.column_dimensions["A"].width, sh.column_dimensions["B"].width = 38, 20
    out.save(xlsx + ".tmp")
    os.replace(xlsx + ".tmp", xlsx)

    n_files = len([f for f in os.listdir(img_dir) if not f.startswith(".")])
    print(f"{len(rows)} of {total} images: every model read the same serial")
    print(f"  matches correct_serial {right}, does not match {len(rows) - right}; accepted by all at {GATE}: {acc}")
    print(f"kept {len(kept)} rows added by hand -> xlsx has {len(out_rows)} rows")
    print(f"images  -> {os.path.relpath(img_dir, ROOT)}  ({n_files} files)")
    print(f"serials -> {os.path.relpath(xlsx, ROOT)}")


if __name__ == "__main__":
    main()
