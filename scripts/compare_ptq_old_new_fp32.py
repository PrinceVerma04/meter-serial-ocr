#!/usr/bin/env python3
"""
Head-to-head test of three recognizer models on the 1,004 labeled eval images:
  - FP32 base      (size_reduction/models/meterocr_v3_fp32.onnx, 84 MB)
  - New PTQ v3 INT8 (deploy/ptq_int8/model/meterocr_ptq_int8.onnx, 21 MB)
  - Old PTQ v2 INT8 (deploy/ptq_py38/model/meterocr_ptq_int8.onnx, 21 MB)

Ground truth = human labels in datasets/eval_1000/accepted_dataset/accepted_labels.csv
(independent of every model -> a fair comparison, incl. for old PTQ).

Scoring: space-insensitive, case-insensitive, leading zeros KEPT
         norm = s.strip().upper().replace(" ", "")

Outputs a side-by-side accuracy table, an N->W defect count, and writes a
per-image CSV to datasets/eval_1000/accepted_dataset/ptq_compare_results.csv
"""
import os
import csv
import numpy as np
import cv2
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "datasets", "eval_1000", "accepted_dataset")
LABELS = os.path.join(BASE, "accepted_labels.csv")
IMAGES = os.path.join(BASE, "images")
DICT = os.path.join(ROOT, "deploy", "ptq_int8", "model", "ppocrv5_dict.txt")
OUT = os.path.join(BASE, "ptq_compare_results.csv")

MODELS = {
    "fp32":    os.path.join(ROOT, "size_reduction", "models", "meterocr_v3_fp32.onnx"),
    "new_ptq": os.path.join(ROOT, "deploy", "ptq_int8", "model", "meterocr_ptq_int8.onnx"),
    "old_ptq": os.path.join(ROOT, "deploy", "ptq_py38", "model", "meterocr_ptq_int8.onnx"),
}

IMG_H, IMG_W = 48, 320


def load_vocab(dict_path):
    chars = [l.rstrip("\n") for l in open(dict_path, encoding="utf-8")]
    return ["blank"] + chars + [" "]


VOCAB = load_vocab(DICT)


def preprocess(bgr):
    h, w = bgr.shape[:2]
    rw = min(IMG_W, int(np.ceil(IMG_H * w / h)))
    r = cv2.resize(bgr, (rw, IMG_H)).astype("float32").transpose(2, 0, 1) / 255.0
    r = (r - 0.5) / 0.5
    pad = np.zeros((1, 3, IMG_H, IMG_W), dtype="float32")
    pad[0, :, :, :rw] = r
    return pad


def decode(logits):
    probs = logits[0]
    idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    out, scores, prev = [], [], -1
    for t, i in enumerate(idx):
        if i != prev and i != 0:
            out.append(VOCAB[i] if i < len(VOCAB) else "")
            scores.append(conf[t])
        prev = i
    text = "".join(out)
    score = float(np.mean(scores)) if scores else 0.0
    return text, score


def norm(s):
    return (s or "").strip().upper().replace(" ", "")


def main():
    # load labels
    rows = []
    with open(LABELS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["filename"], r["label"]))
    print("labeled images:", len(rows))

    # load sessions
    sess = {}
    for name, path in MODELS.items():
        sess[name] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        print("loaded %-8s %s" % (name, os.path.basename(path)))

    inp = {n: s.get_inputs()[0].name for n, s in sess.items()}

    results = []
    missing = 0
    for fn, label in rows:
        p = os.path.join(IMAGES, fn)
        bgr = cv2.imread(p)
        if bgr is None:
            missing += 1
            continue
        x = preprocess(bgr)
        row = {"filename": fn, "gt": label}
        for name, s in sess.items():
            logits = s.run(None, {inp[name]: x})[0]
            text, score = decode(logits)
            row[name + "_pred"] = text
            row[name + "_conf"] = round(score, 4)
        results.append(row)

    n = len(results)
    print("scored:", n, "| unreadable files:", missing)

    # accuracy
    print("\n=== EXACT-MATCH ACCURACY (space/case-insensitive, leading zeros kept) ===")
    stats = {}
    for name in MODELS:
        correct = sum(1 for r in results if norm(r[name + "_pred"]) == norm(r["gt"]))
        blanks = sum(1 for r in results if not norm(r[name + "_pred"]))
        stats[name] = (correct, blanks)
        print("  %-8s  %5d/%d = %6.2f%%   (blank preds: %d)"
              % (name, correct, n, 100.0 * correct / n, blanks))

    # agreement with fp32 (quantization fidelity)
    print("\n=== AGREEMENT WITH FP32 BASE (quantization fidelity) ===")
    for name in ("new_ptq", "old_ptq"):
        agree = sum(1 for r in results if norm(r[name + "_pred"]) == norm(r["fp32_pred"]))
        print("  %-8s == fp32 : %5d/%d = %6.2f%%" % (name, agree, n, 100.0 * agree / n))

    # N->W defect: cases where fp32 has an 'N' that the INT8 model flipped to 'W'
    print("\n=== N->W FLIP (fp32 char 'N' -> INT8 'W' at same position) ===")
    def nw_flips(name):
        cnt = 0
        ex = []
        for r in results:
            a, b = norm(r["fp32_pred"]), norm(r[name + "_pred"])
            if len(a) == len(b) and a != b:
                if any(ca == "N" and cb == "W" for ca, cb in zip(a, b)) and \
                   all((ca == cb) or (ca == "N" and cb == "W") for ca, cb in zip(a, b)):
                    cnt += 1
                    if len(ex) < 5:
                        ex.append("%s -> %s" % (a, b))
        return cnt, ex
    for name in ("new_ptq", "old_ptq"):
        c, ex = nw_flips(name)
        print("  %-8s : %d cases   e.g. %s" % (name, c, "; ".join(ex)))

    # write per-image csv
    cols = ["filename", "gt", "fp32_pred", "fp32_conf",
            "new_ptq_pred", "new_ptq_conf", "old_ptq_pred", "old_ptq_conf",
            "fp32_ok", "new_ptq_ok", "old_ptq_ok"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            for name in MODELS:
                r[name + "_ok"] = int(norm(r[name + "_pred"]) == norm(r["gt"]))
            w.writerow({k: r.get(k, "") for k in cols})
    print("\nwrote per-image results ->", OUT)


if __name__ == "__main__":
    main()
