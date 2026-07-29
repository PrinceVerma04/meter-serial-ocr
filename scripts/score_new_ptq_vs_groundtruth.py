#!/usr/bin/env python3
"""
Score the NEW PTQ INT8 (v3) model against the consensus ground truth.

Ground truth = agreed_fp32_oldptq_qat/result.csv (filename,result) — the serial the
other three models (retrained FP32 + old PTQ v2 + QAT) all agreed on. Runs ONLY the
new PTQ model on those exact images and computes exact-match accuracy.

Output: agreed_fp32_oldptq_qat/new_ptq_vs_groundtruth.csv
        (filename, ground_truth, new_ptq_pred, match)

Run: size_reduction/compat_env/venv/bin/python scripts/score_new_ptq_vs_groundtruth.py
"""
import os, csv, time
import numpy as np
import cv2
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

D = "26788 images for test/agreed_fp32_oldptq_qat"
GT = os.path.join(D, "result.csv")
IMGDIR = os.path.join(D, "images")
OUT = os.path.join(D, "new_ptq_vs_groundtruth.csv")
MODEL = "size_reduction/models/meterocr_v3_int8_ptq.onnx"   # NEW PTQ (deployed)
DICT = "deploy/ptq_int8/model/ppocrv5_dict.txt"
IMG_H, IMG_W = 48, 320
CHARS = ["blank"] + [l.rstrip("\n") for l in open(DICT, encoding="utf-8")] + [" "]


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
    gt = list(csv.DictReader(open(GT)))
    so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 4
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    t0 = time.time(); correct = 0; done = 0; missing = 0
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "ground_truth", "new_ptq_pred", "match"])
        for r in gt:
            name, truth = r["filename"], r["result"]
            path = os.path.join(IMGDIR, name)
            img = cv2.imread(path)
            if img is None:
                missing += 1; continue
            pred = decode(sess.run(None, {iname: preprocess(img)})[0])
            m = int(pred == truth); correct += m; done += 1
            w.writerow([name, truth, pred, m])
            if done % 1000 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(gt)} | acc so far {100*correct/done:.2f}% | {rate:.0f} img/s", flush=True)
                f.flush()
    print("\n=== NEW PTQ vs ground truth ===")
    print(f"  images scored : {done}  (missing {missing})")
    print(f"  correct       : {correct}")
    print(f"  ACCURACY      : {100*correct/done:.2f}%")
    print(f"  time          : {(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
