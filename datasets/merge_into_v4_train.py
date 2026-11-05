#!/usr/bin/env python3
"""
Merge the reviewed new crops (datasets/v4_retrain_nov4/, built by scripts/build_v4_retrain_data.py)
into the training collection datasets/v4_train/ - the data v4.2 is trained on.

  - only runs when all 3 review rounds are complete (BUILD_INFO.json says round3: complete)
  - each new crop gets the split of its meter (capture session) if that meter is already in
    v4_train; crops of new meters are split 80/10/10 by meter (seed 0), so no meter is in two splits
  - new rows: label_source = nov4_reviewed
  - images already in v4_train whose reviewed serial differs (your dashboard decisions) get the
    reviewed serial; the old value is kept in column v4_serial (empty everywhere else)
  - images are copied into v4_train/images/, labels.csv is rewritten in one step; the original
    labels.csv is kept once as labels_v4.csv (= exactly what v4 was trained on)
  - afterwards the staging folder datasets/v4_retrain_nov4/ is removed (rebuildable)
Safe to re-run: if the crops are already merged it only reports.

Run from the project root:
    .venv/bin/python datasets/merge_into_v4_train.py --dry-run
    .venv/bin/python datasets/merge_into_v4_train.py
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE = os.path.join(ROOT, "datasets", "v4_retrain_nov4")
V4 = os.path.join(ROOT, "datasets", "v4_train")
LABELS = os.path.join(V4, "labels.csv")
LABELS_V4 = os.path.join(V4, "labels_v4.csv")
TRUTH = os.path.join(ROOT, "new testing data 4 nov", "clean_extracted_serials.xlsx")
SOURCE = "nov4_reviewed"
SEED = 0


def session(name):
    return name.split("_null_bigImg")[0]


def norm(s):
    return (s or "").strip().upper().replace(" ", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(LABELS, encoding="utf-8")))
    fields = list(rows[0].keys())
    if any(r["label_source"] == SOURCE for r in rows):
        n = sum(r["label_source"] == SOURCE for r in rows)
        print(f"already merged: {n} rows with label_source {SOURCE} in {os.path.relpath(LABELS, ROOT)}")
        return
    info_p = os.path.join(STAGE, "BUILD_INFO.json")
    if not os.path.exists(info_p):
        sys.exit(f"{os.path.relpath(STAGE, ROOT)}/ not built - run scripts/build_v4_retrain_data.py first")
    info = json.load(open(info_p, encoding="utf-8"))
    if info.get("round3") != "complete":
        sys.exit("review round 3 is not complete - finish scripts/review_round3_server.py, "
                 "re-run scripts/build_v4_retrain_data.py, then this")
    new = list(csv.DictReader(open(os.path.join(STAGE, "labels.csv"), encoding="utf-8")))

    # reviewed ground truth for images already in v4_train (overlap)
    import openpyxl
    wb = openpyxl.load_workbook(TRUTH, read_only=True)
    gt = {str(r[0]).strip(): str(r[1]).strip() for r in list(wb["roi_labels"].iter_rows(values_only=True))[1:] if r[0]}
    wb.close()
    by_name = {r["image_name"]: r for r in rows}
    relabel = [(n, by_name[n]["serial"], s) for n, s in gt.items() if n in by_name and norm(s) != norm(by_name[n]["serial"])]

    # splits: follow the meter, else 80/10/10 by meter
    meter_split = defaultdict(set)
    for r in rows:
        meter_split[session(r["image_name"])].add(r["split"])
    bad = {m: s for m, s in meter_split.items() if len(s) > 1}
    if bad:
        sys.exit(f"{len(bad)} meters already sit in more than one split in v4_train - stop")
    dup = [r["image_name"] for r in new if r["image_name"] in by_name]
    if dup:
        sys.exit(f"{len(dup)} new crops are already in v4_train, e.g. {dup[0]} - stop")
    new_meters = sorted({session(r["image_name"]) for r in new} - set(meter_split))
    random.Random(SEED).shuffle(new_meters)
    k1, k2 = int(len(new_meters) * 0.8), int(len(new_meters) * 0.9)
    assign = {m: "train" for m in new_meters[:k1]}
    assign.update({m: "val" for m in new_meters[k1:k2]})
    assign.update({m: "test" for m in new_meters[k2:]})
    for m, s in meter_split.items():
        assign[m] = next(iter(s))

    add = []
    for r in new:
        src = os.path.join(STAGE, "images", r["image_name"])
        if not os.path.isfile(src):
            sys.exit(f"missing image {src}")
        add.append({"image_name": r["image_name"], "serial": r["serial"],
                    "split": assign[session(r["image_name"])], "label_source": SOURCE})
    cnt = defaultdict(int)
    for r in add:
        cnt[r["split"]] += 1
    known = sum(session(r["image_name"]) in meter_split for r in add)
    print(f"v4_train: {len(rows)} rows -> {len(rows) + len(add)}  (+{len(add)} {SOURCE}: "
          f"train {cnt['train']}, val {cnt['val']}, test {cnt['test']})")
    print(f"  {known} new crops follow their meter's existing split; {len(add) - known} from "
          f"{len(new_meters)} new meters split 80/10/10")
    print(f"  relabel {len(relabel)} existing rows to the reviewed serial:")
    for n, old, s in relabel:
        print(f"     {n}: {old} -> {s}  ({by_name[n]['split']})")
    if a.dry_run:
        print("dry run - nothing changed")
        return

    if not os.path.exists(LABELS_V4):
        shutil.copy2(LABELS, LABELS_V4)
        print(f"kept original as {os.path.relpath(LABELS_V4, ROOT)}")
    for r in add:
        shutil.copy2(os.path.join(STAGE, "images", r["image_name"]), os.path.join(V4, "images", r["image_name"]))
    out_fields = fields + (["v4_serial"] if "v4_serial" not in fields else [])
    rel = {n: s for n, _, s in relabel}
    tmp = LABELS + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            r = dict(r, v4_serial=r.get("v4_serial", ""))
            if r["image_name"] in rel:
                r["v4_serial"], r["serial"] = r["serial"], rel[r["image_name"]]
            w.writerow(r)
        for r in add:
            w.writerow(dict(r, v4_serial=""))
    os.replace(tmp, LABELS)
    n_img = len(os.listdir(os.path.join(V4, "images")))
    print(f"saved {os.path.relpath(LABELS, ROOT)} ({len(rows) + len(add)} rows), images/ now {n_img} files")
    if n_img != len(rows) + len(add):
        sys.exit("!! image count does not match labels - staging folder kept, check before training")
    shutil.rmtree(STAGE)
    print(f"removed staging folder {os.path.relpath(STAGE, ROOT)}/ (rebuild with scripts/build_v4_retrain_data.py)")
    print("Next: python3 training/train_v4_2.py")


if __name__ == "__main__":
    main()
