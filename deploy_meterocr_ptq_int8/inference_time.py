#!/usr/bin/env python3
"""
inference_time.py  —  measure OCR inference time on CPU and GPU (ONNX model).

Install (one line — the -gpu build ships BOTH CPU and CUDA providers):
    pip install onnxruntime-gpu opencv-python numpy

Run:
    python inference_time.py path/to/crop.jpg
    python inference_time.py path/to/crop.jpg --runs 100
"""
import time, argparse
import numpy as np
import cv2
import onnxruntime as ort

MODEL = "model/meterocr_ptq_int8.onnx"
DICT  = "model/ppocrv5_dict.txt"
IMG_H, IMG_W = 48, 320

# CTC decode vocabulary: blank at index 0, space appended last
CHARS = ["blank"] + [l.rstrip("\n") for l in open(DICT, encoding="utf-8")] + [" "]


def preprocess(bgr):
    h, w = bgr.shape[:2]
    rw = min(IMG_W, int(np.ceil(IMG_H * w / h)))
    r = cv2.resize(bgr, (rw, IMG_H)).astype("float32").transpose(2, 0, 1) / 255.0
    r = (r - 0.5) / 0.5
    pad = np.zeros((1, 3, IMG_H, IMG_W), "float32")
    pad[0, :, :, :rw] = r
    return pad


def decode(logits):
    idx, conf = logits[0].argmax(1), logits[0].max(1)
    out, sc, prev = [], [], -1
    for t, i in enumerate(idx):
        if i != prev and i != 0:
            out.append(CHARS[i] if i < len(CHARS) else "")
            sc.append(conf[t])
        prev = i
    return "".join(out), (float(np.mean(sc)) if sc else 0.0)


def benchmark(provider, label, image_path, runs):
    if provider not in ort.get_available_providers():
        print(f"\n[{label}]  not available (install onnxruntime-gpu for CUDA). Skipped.")
        return
    sess = ort.InferenceSession(MODEL, providers=[provider])
    iname = sess.get_inputs()[0].name
    x = preprocess(cv2.imread(image_path))

    t = time.time(); logits = sess.run(None, {iname: x})[0]; first = (time.time()-t)*1000
    text, score = decode(logits)

    t = time.time()
    for _ in range(runs): sess.run(None, {iname: x})
    infer = (time.time()-t)/runs*1000

    t = time.time()
    for _ in range(runs): sess.run(None, {iname: preprocess(cv2.imread(image_path))})
    full = (time.time()-t)/runs*1000

    print(f"\n[{label}]  result: '{text}'  (confidence {score:.3f})")
    print(f"    first call        : {first:7.1f} ms")
    print(f"    inference only    : {infer:7.1f} ms/img   ({1000/infer:.0f} img/s)")
    print(f"    load + inference  : {full:7.1f} ms/img")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to a cropped serial image")
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    print("image     :", args.image)
    print("providers :", ort.get_available_providers())
    benchmark("CPUExecutionProvider", "CPU", args.image, args.runs)
    benchmark("CUDAExecutionProvider", "GPU (CUDA)", args.image, args.runs)


if __name__ == "__main__":
    main()
