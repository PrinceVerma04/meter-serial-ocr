#!/usr/bin/env python3
"""
Run all 3 available recognizer models over a folder of crops and write ONE csv
with every model's prediction side by side.

Models (all ONNX, CPU, run in the SAME venv — no Paddle needed):
    fp32     size_reduction/models/meterocr_v3_fp32.onnx        (84 MB, reference)
    ptq_v3   deploy/ptq_int8/model/meterocr_ptq_int8.onnx        (21 MB, shipped/current)
    ptq_v2   size_reduction/models/server_int8.onnx              (21 MB, old)

Note: the QAT model is not on disk anymore (removed in a cleanup pass) so it is
not included here. Re-add it (Paddle-only, separate venv) if you restore it.

Usage:
    python3 scripts/run_3models_tpddl_julyset.py
    python3 scripts/run_3models_tpddl_julyset.py --images "Some_analysis_part/1stjuly_4thjuly_2026_tpddl" \
        --out "Some_analysis_part/three_model_results_tpddl_julyset.csv" --workers 14

Runs in the FOREGROUND (no background/daemonizing) — just call it directly.
"""
import os
import csv
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_IMAGES = os.path.join(ROOT, "Some_analysis_part", "1stjuly_4thjuly_2026_tpddl")
DEFAULT_OUT = os.path.join(ROOT, "Some_analysis_part", "three_model_results_tpddl_julyset.csv")

MODELS = {
    "fp32":   os.path.join(ROOT, "size_reduction", "models", "meterocr_v3_fp32.onnx"),
    "ptq_v3": os.path.join(ROOT, "deploy", "ptq_int8", "model", "meterocr_ptq_int8.onnx"),
    "ptq_v2": os.path.join(ROOT, "size_reduction", "models", "server_int8.onnx"),
}
DICT_PATH = os.path.join(ROOT, "deploy", "ptq_int8", "model", "ppocrv5_dict.txt")

IMG_H, IMG_W = 48, 320

VOCAB = ["blank"] + [l.rstrip("\n") for l in open(DICT_PATH, encoding="utf-8")] + [" "]


def preprocess(bgr):
    h, w = bgr.shape[:2]
    rw = min(IMG_W, int(np.ceil(IMG_H * w / h)))
    r = cv2.resize(bgr, (rw, IMG_H)).astype("float32").transpose(2, 0, 1) / 255.0
    r = (r - 0.5) / 0.5
    pad = np.zeros((1, 3, IMG_H, IMG_W), dtype="float32")
    pad[0, :, :, :rw] = r
    return pad


def decode(logits):
    p = logits[0]
    idx = p.argmax(1)
    prev = -1
    out = []
    for i in idx:
        if i != prev and i != 0:
            out.append(VOCAB[i] if i < len(VOCAB) else "")
        prev = i
    return "".join(out)


# ---- one ONNX session per worker process, re-used across images ----
_SESS = None
_INPUT_NAME = None


def _init(model_path):
    global _SESS, _INPUT_NAME
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    _SESS = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    _INPUT_NAME = _SESS.get_inputs()[0].name


def _work(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return os.path.basename(path), ""
    logits = _SESS.run(None, {_INPUT_NAME: preprocess(bgr)})[0]
    return os.path.basename(path), decode(logits)


def run_model(name, model_path, files, workers):
    print(f"== {name} ({os.path.basename(model_path)}) ==", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(model_path,)) as ex:
        done = 0
        for image_name, text in ex.map(_work, files, chunksize=16):
            results[image_name] = text
            done += 1
            if done % 1000 == 0:
                print(f"  [{name}] {done}/{len(files)}", flush=True)
    print(f"  [{name}] done {len(files)}", flush=True)
    return results


def norm(s):
    return (s or "").strip().upper().replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=DEFAULT_IMAGES, help="folder of .jpg crops")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output csv path")
    ap.add_argument("--workers", type=int, default=min(14, os.cpu_count() or 4))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    if not files:
        raise SystemExit(f"no .jpg files found under {args.images}")
    print(f"images: {len(files)}  |  workers: {args.workers}", flush=True)

    per_model = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            raise SystemExit(f"model missing: {path}")
        per_model[name] = run_model(name, path, files, args.workers)

    model_names = list(MODELS)
    agree_n = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name"] + model_names + ["all_agree"])
        for path in files:
            name = os.path.basename(path)
            preds = [per_model[m].get(name, "") for m in model_names]
            all_agree = len({norm(p) for p in preds}) == 1 and norm(preds[0]) != ""
            if all_agree:
                agree_n += 1
            w.writerow([name] + preds + ["yes" if all_agree else "no"])

    print(f"\nwrote {args.out}")
    print(f"all-{len(model_names)}-agree: {agree_n} / {len(files)} ({100.0 * agree_n / len(files):.1f}%)")


if __name__ == "__main__":
    main()
