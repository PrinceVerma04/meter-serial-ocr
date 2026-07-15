#!/usr/bin/env python3
"""
Build a PaddleX text-recognition dataset from results/ground_truth.csv for
fine-tuning PP-OCRv5_server_rec.

Split is GROUPED BY METER SESSION (the c... filename prefix) so the same meter
never appears in both train and test -> no data leakage -> honest eval.

Output:
  finetune/dataset/
    images/            copies of the labeled crops
    train.txt          "images/<file>\\t<label>"  (per line)
    val.txt
    test.txt           held out for the final before/after
    dict.txt           character set (A-Z, 0-9, space)
"""
import csv, os, re, random, shutil, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
random.seed(0)

GT = "results/ground_truth.csv"
IMG_DIRS = ["data/serial_roi_test", "data/serialno_dev", "dataset for test/ROI"]   # search order
OUT = "finetune/dataset"


def find_image(name):
    for d in IMG_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def session(name):
    return name.split("_null_bigImg")[0]


rows = [r for r in csv.DictReader(open(GT)) if r["verdict"] != "unreadable" and r["true_label"].strip()]
print(f"readable labels in ground truth: {len(rows)}")

# resolve images, drop any missing
items = []
missing = 0
for r in rows:
    p = find_image(r["image"])
    if p is None:
        missing += 1
        continue
    label = r["true_label"].strip().upper()
    items.append((r["image"], p, label, session(r["image"])))
print(f"resolved: {len(items)}  (missing images skipped: {missing})")

# group by session, then split sessions 70/15/15
by_sess = collections.defaultdict(list)
for it in items:
    by_sess[it[3]].append(it)
sessions = list(by_sess)
random.shuffle(sessions)
n = len(sessions)
n_tr, n_va = int(n * 0.70), int(n * 0.15)
split = {"train": sessions[:n_tr], "val": sessions[n_tr:n_tr + n_va], "test": sessions[n_tr + n_va:]}
print(f"sessions: {n}  -> train {len(split['train'])}, val {len(split['val'])}, test {len(split['test'])}")

# write dataset
os.makedirs(f"{OUT}/images", exist_ok=True)
chars = set()
counts = {}
for part in ("train", "val", "test"):
    lines = []
    for s in split[part]:
        for name, src, label, _ in by_sess[s]:
            dst_rel = f"images/{name}"
            shutil.copy(src, f"{OUT}/{dst_rel}")
            lines.append(f"{dst_rel}\t{label}")
            chars.update(label)
    with open(f"{OUT}/{part}.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    counts[part] = len(lines)

# character dictionary (one char per line; PaddleX rec dict format)
dict_chars = sorted(c for c in chars if c != " ")   # space handled via use_space_char
with open(f"{OUT}/dict.txt", "w") as fh:
    fh.write("\n".join(dict_chars) + "\n")

print(f"\nwrote {OUT}/")
print(f"  images: train {counts['train']}, val {counts['val']}, test {counts['test']}")
print(f"  dict.txt chars ({len(dict_chars)}): {''.join(dict_chars)}")
print(f"  space in labels: {' ' in chars}")
