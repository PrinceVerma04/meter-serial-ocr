#!/usr/bin/env python3
"""
Score model predictions against the hand-labeled ground truth.

Reads results/labels.csv (image,true_label) and compares each model's reading
from results/test_results.csv (4-model comparison) to it. Reports, per model:
  - exact-match accuracy (ignoring case/spacing)
  - character error rate (CER = edit distance / label length)
on the labeled subset (blank labels = "unreadable", scored separately).

Usage: python3 scripts/score_accuracy.py
"""
import csv, os, re, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="results/ground_truth.csv")
ap.add_argument("--preds", default="results/test_results.csv",
                help="CSV with one column per model (from compare_all_models.py)")
args = ap.parse_args()

if not os.path.exists(args.labels):
    raise SystemExit(f"No labels file at {args.labels}. Label some images first "
                     f"(scripts/make_labeling_page.py -> download labels.csv into results/).")


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def cer(a, b):
    a, b = norm(a), norm(b)
    if not b:
        return 0.0 if not a else 1.0
    # Levenshtein
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / len(b)


labels = {r["image"]: r["true_label"] for r in csv.DictReader(open(args.labels))
          if r["true_label"].strip() != "" or True}
labels = {k: v for k, v in labels.items()}
preds = {r["image"]: r for r in csv.DictReader(open(args.preds))}
models = [c for c in next(iter(preds.values())) if c not in ("image", "agree") and not c.endswith("_score")]

# split labeled images into readable (has a true serial) vs unreadable (blank label)
readable = [im for im in labels if norm(labels[im]) and im in preds]
unread = [im for im in labels if not norm(labels[im]) and im in preds]

print(f"labeled: {len(labels)}  (readable: {len(readable)}, marked unreadable: {len(unread)})\n")
print(f"{'model':16}{'exact-match':>13}{'char-acc':>11}{'false-reads*':>14}")
print("-" * 54)
for m in models:
    exact = sum(1 for im in readable if norm(preds[im][m]) == norm(labels[im]))
    cers = [cer(preds[im][m], labels[im]) for im in readable]
    char_acc = 1 - sum(cers) / len(cers) if cers else 0
    # false-reads: model output text on an image the human marked unreadable
    false_reads = sum(1 for im in unread if norm(preds[im][m]))
    ex = f"{exact}/{len(readable)} ({exact/len(readable)*100:.0f}%)" if readable else "-"
    print(f"{m:16}{ex:>13}{char_acc*100:>10.1f}%{false_reads:>10}/{len(unread)}")
print("\n* false-reads = model produced text where the image was truly unreadable "
      "(lower is better; this is the hallucination measure).")
