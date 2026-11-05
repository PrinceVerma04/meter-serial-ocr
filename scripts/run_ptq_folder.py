
#!/usr/bin/env python3
"""
Run a PTQ INT8 model over a folder of serial crops -> CSV (image_name, extracted_result).
Parallel across CPU cores, CPU-only ONNX Runtime, same preprocessing/decoding as deploy/ptq_int8.

    python3 scripts/run_ptq_folder.py --images <crops_dir> --out results.csv            # v3 (deployed)
    python3 scripts/run_ptq_folder.py --images <crops_dir> --out results.csv --model v4
"""
import os, csv, glob, argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2, onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"v3": os.path.join(ROOT, "models", "v3", "meterocr_v3_int8_ptq.onnx"),   # = deployed model
          "v4": os.path.join(ROOT, "models", "v4", "meterocr_v4_int8_ptq.onnx"),
          "v4.2": os.path.join(ROOT, "models", "v4.2", "meterocr_v4.2_int8_ptq.onnx")}
MODEL  = MODELS["v3"]                       # replaced from --model in main() before workers start
DICT   = os.path.join(ROOT, "deploy", "ptq_int8", "model", "ppocrv5_dict.txt")
IMG_H, IMG_W = 48, 320

_VOCAB = ["blank"] + [l.rstrip("\n") for l in open(DICT, encoding="utf-8")] + [" "]
_SESS = None                     # one ONNX session per worker process
_INP = None

def _init(model_path):
    global _SESS, _INP
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    _SESS = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
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
    ap.add_argument("--images", required=True, help="folder of serial crops")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--model", choices=sorted(MODELS), default="v3", help="v3 = deployed (default), v4 = new")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()
    global MODEL
    MODEL = MODELS[a.model]

    files = sorted(glob.glob(os.path.join(a.images, "*.jpg")) +
                   glob.glob(os.path.join(a.images, "*.jpeg")) +
                   glob.glob(os.path.join(a.images, "*.png")))
    print("images: %d  |  workers: %d  |  model: %s" % (len(files), a.workers, a.model))

    with open(a.out, "w", newline="", encoding="utf-8") as f, \
         ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(MODEL,)) as ex:
        w = csv.writer(f); w.writerow(["image_name", "extracted_result"])
        done = 0
        for name, text in ex.map(_work, files, chunksize=32):
            w.writerow([name, text]); done += 1
            if done % 2000 == 0:
                print("  %d/%d" % (done, len(files)), flush=True)
    print("wrote %s" % a.out)

if __name__ == "__main__":
    main()