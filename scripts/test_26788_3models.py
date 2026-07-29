#!/usr/bin/env python3
"""
Run the three current models on the 26,788-image test set (no ground truth) and
write side-by-side predictions.  All three are ONNX / ONNX Runtime (CPU), so the
run is single-environment and reproducible.

Models:
    v3_fp32  : size_reduction/models/meterocr_v3_fp32.onnx                 (81 MB, reference)
    v3_int8  : size_reduction/models/meterocr_v3_int8_ptq.onnx             (22 MB, new PTQ)
    v2_int8  : deploy/ptq_int8_OLD_backup_2026-07-15/model/meterocr_ptq_int8.onnx (21 MB, old PTQ)

Output:
    26788 images for test/three_model_results.csv   filename,v3_fp32,v3_int8,v2_int8

Resumable: re-running continues from rows already written.
Usage:
    size_reduction/compat_env/venv/bin/python scripts/test_26788_3models.py [--limit N]
"""
import os, glob, csv, sys, time
import numpy as np
import cv2
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

IMGDIR = "26788 images for test/images"
OUTCSV = "26788 images for test/three_model_results.csv"
DICT = "deploy/ptq_int8/model/ppocrv5_dict.txt"
MODELS = {
    "v3_fp32": "size_reduction/models/meterocr_v3_fp32.onnx",
    "v3_int8": "size_reduction/models/meterocr_v3_int8_ptq.onnx",
    "v2_int8": "deploy/ptq_int8_OLD_backup_2026-07-15/model/meterocr_ptq_int8.onnx",
}
IMG_H, IMG_W = 48, 320
CHARS = ["blank"] + [l.rstrip("\n") for l in open(DICT, encoding="utf-8")] + [" "]

limit = None
if "--limit" in sys.argv:
    limit = int(sys.argv[sys.argv.index("--limit") + 1])


def preprocess(bgr):
    h, w = bgr.shape[:2]
    rw = min(IMG_W, max(1, int(np.ceil(IMG_H * w / h))))
    r = cv2.resize(bgr, (rw, IMG_H)).astype("float32").transpose(2, 0, 1) / 255.0
    r = (r - 0.5) / 0.5
    pad = np.zeros((1, 3, IMG_H, IMG_W), "float32")
    pad[0, :, :, :rw] = r
    return pad


def decode(logits):
    idx = logits[0].argmax(1)
    out, prev = [], -1
    for i in idx:
        if i != prev and i != 0:
            out.append(CHARS[i] if i < len(CHARS) else "")
        prev = i
    return "".join(out)


def main():
    so = ort.SessionOptions()
    so.intra_op_num_threads = os.cpu_count() or 4
    sess = {k: ort.InferenceSession(v, so, providers=["CPUExecutionProvider"])
            for k, v in MODELS.items()}
    inp = {k: s.get_inputs()[0].name for k, s in sess.items()}

    files = sorted(glob.glob(os.path.join(IMGDIR, "*.jpg")))
    if limit:
        files = files[:limit]

    done = set()
    if os.path.exists(OUTCSV):
        with open(OUTCSV) as f:
            done = {r[0] for r in csv.reader(f) if r and r[0] != "filename"}
    new_file = not os.path.exists(OUTCSV)

    t0 = time.time()
    with open(OUTCSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["filename", "v3_fp32", "v3_int8", "v2_int8"])
        n = 0
        for i, path in enumerate(files):
            name = os.path.basename(path)
            if name in done:
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            x = preprocess(img)
            row = [name]
            for k in ("v3_fp32", "v3_int8", "v2_int8"):
                out = sess[k].run(None, {inp[k]: x})
                row.append(decode(out[0]))
            w.writerow(row)
            n += 1
            if n % 200 == 0:
                dt = time.time() - t0
                rate = n / dt
                left = (len(files) - len(done) - n) / rate if rate else 0
                print(f"  {n} done  |  {rate:.1f} img/s  |  ~{left/60:.1f} min left", flush=True)
                f.flush()
    print(f"DONE: {n} new rows in {(time.time()-t0)/60:.1f} min -> {OUTCSV}")


if __name__ == "__main__":
    main()
