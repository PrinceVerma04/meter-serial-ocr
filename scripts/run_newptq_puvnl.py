
#!/usr/bin/env python3
"""
Run the NEW PTQ v3 INT8 model over the PUVNL dataset -> CSV (image_name, extracted_result).
Parallel across CPU cores. CPU-only ONNX Runtime.

    python3 scripts/run_newptq_puvnl.py                      # defaults below
    python3 scripts/run_newptq_puvnl.py --workers 14 --out puvnl_results.csv
"""
import os, csv, glob, argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2, onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, "25th_26th_july_2026_serialno_puvnl")
MODEL  = os.path.join(ROOT, "deploy", "ptq_int8", "model", "meterocr_ptq_int8.onnx")
DICT   = os.path.join(ROOT, "deploy", "ptq_int8", "model", "ppocrv5_dict.txt")
IMG_H, IMG_W = 48, 320

_VOCAB = ["blank"] + [l.rstrip("\n") for l in open(DICT, encoding="utf-8")] + [" "]
_SESS = None                     # one ONNX session per worker process
_INP = None

def _init():
    global _SESS, _INP
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    _SESS = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    _INP = _SESS.get_inputs()[0].name

def _pre(bgr):
    h, w = bgr.shape[:2]
    rw = min(IMG_W, int(np.ceil(IMG_H * w / h)))
    r = cv2.resize(bgr, (rw, IMG_H)).astype("float32").transpose(2, 0, 1) / 255.0
    r = (r - 0.5) / 0.5
    pad = np.zeros((1, 3, IMG_H, IMG_W), dtype="float32"); pad[0, :, :, :rw] = r
    return pad

def _decode(logits):
    p = logits[0]; idx = p.argmax(1); prev = -1; out = []
    for i in idx:
        if i != prev and i != 0:
            out.append(_VOCAB[i] if i < len(_VOCAB) else "")
        prev = i
    return "".join(out)

def _work(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return os.path.basename(path), ""
    logits = _SESS.run(None, {_INP: _pre(bgr)})[0]
    return os.path.basename(path), _decode(logits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=IMAGES)
    ap.add_argument("--out", default=os.path.join(ROOT, "puvnl_newptq_results.csv"))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.images, "*.jpg")) +
                   glob.glob(os.path.join(a.images, "*.jpeg")) +
                   glob.glob(os.path.join(a.images, "*.png")))
    print("images: %d  |  workers: %d" % (len(files), a.workers))

    with open(a.out, "w", newline="", encoding="utf-8") as f, \
         ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        w = csv.writer(f); w.writerow(["image_name", "extracted_result"])
        done = 0
        for name, text in ex.map(_work, files, chunksize=32):
            w.writerow([name, text]); done += 1
            if done % 2000 == 0:
                print("  %d/%d" % (done, len(files)), flush=True)
    print("wrote %s" % a.out)

if __name__ == "__main__":
    main()