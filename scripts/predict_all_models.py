#!/usr/bin/env python3
"""
Read one (or a few) meter crops with ALL THREE models and print what each one says.

    python3 predict_all_models.py crop.jpg
    python3 predict_all_models.py a.jpg b.jpg c.jpg
    python3 predict_all_models.py crop.jpg --json
    python3 predict_all_models.py *.jpg --csv out.csv

Models (loaded from the project tree):
    FP32 : finetune/inference_v2/            81 MB  Paddle   98.31% *
    QAT  : deploy/qat_int8/model/   22 MB  Paddle   98.31% *
    PTQ  : deploy/ptq_int8/model/   21 MB  ONNX     97.01% *
    * exact-match on the 1,004 human-verified crops

Each model reports serial, confidence, a gate status (accepted/review/blank) and
its own inference time. Models that fail to load are reported and skipped rather
than killing the run — so this still works on a box with only onnxruntime.

For a whole folder use test_three_models.py instead; it is built for bulk and resumable.
"""
import os
import sys
import re
import csv
import json
import time
import argparse

import numpy as np
# numpy-alias shim (old openpyxl in the import chain on this box)
for _a in ("float", "int", "bool", "object", "str"):
    if not hasattr(np, _a):
        setattr(np, _a, getattr(__builtins__, _a, None) if not isinstance(__builtins__, dict)
                else __builtins__.get(_a))
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

FP32_DIR = "finetune/inference_v2"
QAT_PDMODEL = "deploy/qat_int8/model/meterocr_qat_int8.pdmodel"
QAT_PDIPARAMS = "deploy/qat_int8/model/meterocr_qat_int8.pdiparams"
PTQ_ONNX = "deploy/ptq_int8/model/meterocr_ptq_int8.onnx"
DICT = "deploy/ptq_int8/model/ppocrv5_dict.txt"

ACCEPT_THRESHOLD = 0.95
IMG_H, IMG_W = 48, 320
MODELS = ["fp32", "qat", "ptq"]

CHARS = ["blank"] + [l.rstrip("\n") for l in open(DICT, encoding="utf-8")] + [" "]


# ---------- shared preprocessing + CTC decode (QAT & PTQ) ----------
def preprocess(bgr):
    """Must match training: resize to 3x48x320, /255, then (x-0.5)/0.5."""
    h, w = bgr.shape[:2]
    rw = min(IMG_W, max(1, int(np.ceil(IMG_H * w / h))))
    r = cv2.resize(bgr, (rw, IMG_H)).astype("float32").transpose(2, 0, 1) / 255.0
    r = (r - 0.5) / 0.5
    pad = np.zeros((1, 3, IMG_H, IMG_W), "float32")
    pad[0, :, :, :rw] = r
    return pad


def decode(logits):
    """CTC greedy decode -> (text, mean confidence over emitted chars)."""
    probs = logits[0]
    idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    out, scores, prev = [], [], -1
    for t, i in enumerate(idx):
        if i != prev and i != 0:            # collapse repeats, drop blank(0)
            out.append(CHARS[i] if i < len(CHARS) else "")
            scores.append(conf[t])
        prev = i
    return "".join(out), (float(np.mean(scores)) if scores else 0.0)


def clean(text):
    """Same normalization used when these models were benchmarked."""
    return re.sub(r"[^A-Z0-9]+", " ", (text or "").upper()).strip()


def gate(text, score, threshold):
    if not text:
        return "blank"
    return "accepted" if score >= threshold else "review"


# ---------- model wrappers ----------
class OnnxRec:                             # PTQ
    name = "ptq"
    def __init__(self, path, use_gpu=False):
        import onnxruntime as ort
        prov = (["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu
                else ["CPUExecutionProvider"])
        self.s = ort.InferenceSession(path, providers=prov)
        self.i = self.s.get_inputs()[0].name
    def read(self, bgr):
        return decode(self.s.run(None, {self.i: preprocess(bgr)})[0])


class PaddleRec:                           # QAT (legacy .pdmodel)
    name = "qat"
    def __init__(self, pdmodel, pdiparams, use_gpu=False):
        import paddle.inference as pi
        cfg = pi.Config(pdmodel, pdiparams)
        cfg.enable_use_gpu(500, 0) if use_gpu else cfg.disable_gpu()
        cfg.disable_glog_info()
        self.p = pi.create_predictor(cfg)
        self.i = self.p.get_input_names()[0]
        self.o = self.p.get_output_names()[0]
    def read(self, bgr):
        h = self.p.get_input_handle(self.i)
        h.copy_from_cpu(preprocess(bgr))
        self.p.run()
        return decode(self.p.get_output_handle(self.o).copy_to_cpu())


class TextRec:                             # FP32 (via PaddleOCR)
    name = "fp32"
    def __init__(self, model_dir, use_gpu=True):
        from paddleocr import TextRecognition
        kw = dict(device="gpu") if use_gpu else dict(enable_mkldnn=False)
        self.m = TextRecognition(model_name="PP-OCRv5_server_rec", model_dir=model_dir, **kw)
    def read(self, bgr):
        r = self.m.predict(bgr)
        d = r[0] if r else {}
        d = d if isinstance(d, dict) else getattr(d, "res", d)
        return d.get("rec_text", ""), float(d.get("rec_score", 0.0))


def load_models(args):
    """Build each model, reporting (not raising) on the ones that cannot load."""
    models, failed = {}, {}
    specs = [
        ("fp32", lambda: TextRec(FP32_DIR, use_gpu=not args.cpu)),
        ("qat",  lambda: PaddleRec(QAT_PDMODEL, QAT_PDIPARAMS, use_gpu=args.qat_gpu)),
        ("ptq",  lambda: OnnxRec(PTQ_ONNX, use_gpu=args.ptq_gpu)),
    ]
    for key, build in specs:
        if key in args.skip:
            continue
        try:
            t0 = time.time()
            models[key] = build()
            if not args.quiet:
                print(f"  loaded {key.upper():4} ({time.time()-t0:.1f}s)", file=sys.stderr)
        except Exception as e:
            failed[key] = f"{type(e).__name__}: {e}"
            print(f"  [!] {key.upper()} unavailable — {type(e).__name__}: {e}", file=sys.stderr)
    return models, failed


def run_one(models, path, threshold, raw):
    bgr = cv2.imread(path)
    row = {"image": path}
    if bgr is None:
        row["error"] = "unreadable file (not an image, or path is wrong)"
        for m in MODELS:
            row[m] = {"serial": "", "confidence": 0.0, "status": "blank", "ms": 0.0}
        return row

    row["size"] = f"{bgr.shape[1]}x{bgr.shape[0]}"
    for key in MODELS:
        if key not in models:
            row[key] = None
            continue
        t0 = time.time()
        text, score = models[key].read(bgr)
        ms = (time.time() - t0) * 1000
        if not raw:
            text = clean(text)
        row[key] = {"serial": text, "confidence": round(score, 4),
                    "status": gate(text, score, threshold), "ms": round(ms, 1)}
    return row


def print_human(row, models):
    print(f"\n{row['image']}" + (f"   [{row['size']}]" if "size" in row else ""))
    if "error" in row:
        print(f"  ERROR: {row['error']}")
        return
    print(f"  {'MODEL':6} {'SERIAL':22} {'CONF':>7}  {'STATUS':9} {'TIME':>9}")
    print(f"  {'-'*6} {'-'*22} {'-'*7}  {'-'*9} {'-'*9}")
    for key in MODELS:
        r = row.get(key)
        if r is None:
            print(f"  {key.upper():6} {'(not loaded)':22}")
            continue
        serial = r["serial"] or "(blank)"
        print(f"  {key.upper():6} {serial:22} {r['confidence']:>7.4f}  "
              f"{r['status']:9} {r['ms']:>7.1f}ms")

    reads = {k: row[k]["serial"] for k in MODELS if row.get(k)}
    if len(reads) > 1:
        uniq = set(reads.values())
        if len(uniq) == 1:
            print("  -> all models agree")
        else:
            print(f"  -> DISAGREE ({len(uniq)} distinct readings) — trust FP32/QAT over PTQ")


def main():
    ap = argparse.ArgumentParser(
        description="Read meter crops with the FP32, QAT and PTQ models at once.")
    ap.add_argument("images", nargs="+", help="image path(s)")
    ap.add_argument("--threshold", type=float, default=ACCEPT_THRESHOLD,
                    help="auto-accept confidence cutoff (default 0.95)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--csv", metavar="FILE", help="also write a flat CSV")
    ap.add_argument("--raw", action="store_true",
                    help="skip normalization and show each model's verbatim output")
    ap.add_argument("--skip", default="", help="comma-separated models to skip, e.g. --skip qat")
    ap.add_argument("--cpu", action="store_true", help="run FP32 on CPU (default: GPU if available)")
    ap.add_argument("--qat-gpu", action="store_true", help="try QAT on GPU (needs INT8 kernels)")
    ap.add_argument("--ptq-gpu", action="store_true", help="try PTQ on GPU (needs onnxruntime-gpu)")
    ap.add_argument("--quiet", action="store_true", help="suppress load messages")
    args = ap.parse_args()
    args.skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    missing = [p for p in args.images if not os.path.exists(p)]
    if missing:
        # Most common cause by far: an unquoted path with spaces, which the shell
        # split into several "files". Detect it by re-joining and testing.
        rejoined = " ".join(args.images)
        if len(args.images) > 1 and os.path.exists(rejoined):
            print(f"no such file: {', '.join(missing)}\n\n"
                  f"That path contains spaces and was not quoted, so the shell split it\n"
                  f"into {len(args.images)} arguments. Wrap it in quotes:\n\n"
                  f'    python3 {os.path.basename(__file__)} "{rejoined}"',
                  file=sys.stderr)
        else:
            print("no such file: " + "\n               ".join(missing), file=sys.stderr)
        return 2

    if not args.quiet:
        print("loading models ...", file=sys.stderr)
    models, failed = load_models(args)
    if not models:
        print("no models could be loaded — nothing to do.", file=sys.stderr)
        return 1

    rows = [run_one(models, p, args.threshold, args.raw) for p in args.images]

    if args.json:
        print(json.dumps(rows if len(rows) > 1 else rows[0], indent=2))
    else:
        for row in rows:
            print_human(row, models)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            head = ["image"]
            for m in MODELS:
                head += [f"{m}_serial", f"{m}_confidence", f"{m}_status", f"{m}_ms"]
            w.writerow(head)
            for row in rows:
                line = [row["image"]]
                for m in MODELS:
                    r = row.get(m) or {}
                    line += [r.get("serial", ""), r.get("confidence", ""),
                             r.get("status", ""), r.get("ms", "")]
                w.writerow(line)
        print(f"\nwrote {args.csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
