#!/usr/bin/env python3
"""
Score the 3 meter-OCR models (FP32 / QAT / PTQ) on the human-accepted label set.

Universe: the 1,004 crops whose curation decision was "accept" — every label here
was confirmed by eye, so unlike the earlier 26k run the ground truth is trusted.

Inputs
    datasets/eval_1000/accepted_dataset/accepted_labels.csv
    datasets/eval_1000/accepted_dataset/three_model_results.csv

Output
    datasets/eval_1000/accepted_dataset/model_results.csv
        filename, label, <m>_result, <m>_correct for each model, then a SUMMARY block

Matching: EXACT on the model's own normalization (upper, alnum, single-spaced) —
the same clean() the models' outputs already went through, applied to the label too,
so a label typed "TPWODL 1107954" is not marked wrong over whitespace.

Run:
    python3 scripts/score_accepted_dataset.py
"""
import os, csv, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

BASE = "datasets/eval_1000/accepted_dataset"
LABELS = f"{BASE}/accepted_labels.csv"
PREDS = f"{BASE}/three_model_results.csv"
OUT = f"{BASE}/model_results.csv"
MODELS = ["fp32", "qat", "ptq"]


def clean(text):
    """Same normalization test_three_models.py applies to every model reading."""
    return re.sub(r"[^A-Z0-9]+", " ", (text or "").upper()).strip()


def main():
    labels = {r["filename"]: r["label"]
              for r in csv.DictReader(open(LABELS, newline="", encoding="utf-8-sig"))}
    preds = {r["filename"]: r
             for r in csv.DictReader(open(PREDS, newline="", encoding="utf-8-sig"))}

    rows, correct = [], {m: 0 for m in MODELS}
    missing = []
    for fn, label in labels.items():
        p = preds.get(fn)
        if p is None:
            missing.append(fn)
            continue
        gt = clean(label)
        row = [fn, label]
        for m in MODELS:
            read = p[f"{m}_result"]
            ok = clean(read) == gt and gt != ""
            correct[m] += ok
            row += [read, "TRUE" if ok else "FALSE"]
        rows.append(row)

    total = len(rows)
    header = ["filename", "label"]
    for m in MODELS:
        header += [f"{m}_result", f"{m}_correct"]

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
        w.writerow([])
        w.writerow(["SUMMARY", "total", "correct", "wrong", "accuracy_%"])
        for m in MODELS:
            w.writerow([m.upper(), total, correct[m], total - correct[m],
                        round(100 * correct[m] / total, 2) if total else 0])

    print(f"scored {total} images" + (f"  ({len(missing)} label rows had no prediction)" if missing else ""))
    for m in MODELS:
        print(f"  {m.upper():5} {correct[m]:5}/{total}  = {100 * correct[m] / total:.2f}%")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
