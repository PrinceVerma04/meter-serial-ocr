#!/usr/bin/env python3
"""
Build a PaddleX text-recognition dataset from the new labeled folder:

    retraining dataset with label/
        ground_truth.csv     columns: filename, ground truth
        images/              the crops

Split is GROUPED BY METER SESSION (the c... filename prefix, i.e. everything
before "_null_bigImg") so the same capture session never lands in both train
and test -> no leakage -> honest eval. 70 / 15 / 15.

Output (default: finetune/dataset_retrain/):
    images/            copies of the labeled crops
    train.txt          "images/<file>\\t<LABEL>"  (one per line)
    val.txt
    test.txt
    dict.txt           character set (one char per line; space handled via use_space_char)

Run:
    python3 scripts/prepare_retrain_data.py
    python3 scripts/prepare_retrain_data.py --out finetune/dataset   # overwrite the old set
"""
import csv, os, random, shutil, collections, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
random.seed(0)

ap = argparse.ArgumentParser()
ap.add_argument("--src",  default="retraining dataset with label",
                help="folder holding ground_truth.csv + images/")
ap.add_argument("--csv",  default=None, help="override path to the label CSV")
ap.add_argument("--out",  default="finetune/dataset_retrain",
                help="output PaddleX rec dataset dir")
ap.add_argument("--train", type=float, default=0.70)
ap.add_argument("--val",   type=float, default=0.15)
args = ap.parse_args()

SRC     = args.src
CSV     = args.csv or os.path.join(SRC, "ground_truth.csv")
IMG_DIR = os.path.join(SRC, "images")
OUT     = args.out


def session(name):
    return name.split("_null_bigImg")[0]


# ---- read labels (columns: filename, ground truth) ----
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
# tolerate either header spelling
fkey = "filename" if "filename" in rows[0] else list(rows[0])[0]
lkey = "ground truth" if "ground truth" in rows[0] else list(rows[0])[1]
print(f"rows in CSV: {len(rows)}  (using columns '{fkey}' / '{lkey}')")

items, missing, blank = [], 0, 0
for r in rows:
    name  = (r.get(fkey) or "").strip()
    label = (r.get(lkey) or "").strip().upper()
    if not name or not label:
        blank += 1
        continue
    p = os.path.join(IMG_DIR, name)
    if not os.path.exists(p):
        missing += 1
        continue
    items.append((name, p, label, session(name)))
print(f"usable: {len(items)}  (blank labels: {blank}, missing images: {missing})")

# ---- group by session, split sessions 70/15/15 ----
by_sess = collections.defaultdict(list)
for it in items:
    by_sess[it[3]].append(it)
sessions = list(by_sess)
random.shuffle(sessions)
n = len(sessions)
n_tr = int(n * args.train)
n_va = int(n * args.val)
split = {"train": sessions[:n_tr],
         "val":   sessions[n_tr:n_tr + n_va],
         "test":  sessions[n_tr + n_va:]}
print(f"sessions: {n}  -> train {len(split['train'])}, "
      f"val {len(split['val'])}, test {len(split['test'])}")

# ---- write dataset ----
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(f"{OUT}/images", exist_ok=True)

chars, counts = set(), {}
for part in ("train", "val", "test"):
    lines = []
    for s in split[part]:
        for name, src, label, _ in by_sess[s]:
            shutil.copy(src, f"{OUT}/images/{name}")
            lines.append(f"images/{name}\t{label}")
            chars.update(label)
    with open(f"{OUT}/{part}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    counts[part] = len(lines)

dict_chars = sorted(c for c in chars if c != " ")
with open(f"{OUT}/dict.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(dict_chars) + "\n")

# sanity: no session leakage across splits
seen = {}
leak = 0
for part in ("train", "val", "test"):
    for s in split[part]:
        if s in seen and seen[s] != part:
            leak += 1
        seen[s] = part

print(f"\nwrote {OUT}/")
print(f"  images: train {counts['train']}, val {counts['val']}, test {counts['test']}")
print(f"  dict.txt chars ({len(dict_chars)}): {''.join(dict_chars)}")
print(f"  space in labels: {' ' in chars}")
print(f"  session leakage across splits: {leak}  (must be 0)")
