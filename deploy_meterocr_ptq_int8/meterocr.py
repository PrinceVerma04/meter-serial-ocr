#!/usr/bin/env python3
"""
Meter serial-number OCR — PTQ INT8 model, self-contained inference.

Runs the 21 MB INT8 ONNX recognizer with ONNX Runtime. No PaddlePaddle needed.
Input: a cropped image of the meter serial area (BGR, as read by cv2.imread).
Output: (serial_text, confidence) plus a status from the confidence gate.

Usage:
    from meterocr import MeterOCR
    ocr = MeterOCR()                       # loads model/ next to this file
    text, score, status = ocr.predict("crop.jpg")
"""
import os
import numpy as np
import cv2
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model", "meterocr_ptq_int8.onnx")
DICT = os.path.join(HERE, "model", "ppocrv5_dict.txt")

# reads at/above this confidence are auto-accepted; below -> route to human review
ACCEPT_THRESHOLD = 0.95
IMG_H, IMG_W = 48, 320


class MeterOCR:
    def __init__(self, model_path=MODEL, dict_path=DICT, providers=None):
        self.sess = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        chars = [l.rstrip("\n") for l in open(dict_path, encoding="utf-8")]
        # CTC decode vocabulary: blank at index 0, space appended last
        self.character = ["blank"] + chars + [" "]

    # ---- preprocessing (must match training: resize to 3x48x320, normalize) ----
    def _preprocess(self, bgr):
        h, w = bgr.shape[:2]
        rw = min(IMG_W, int(np.ceil(IMG_H * w / h)))
        r = cv2.resize(bgr, (rw, IMG_H)).astype("float32").transpose(2, 0, 1) / 255.0
        r = (r - 0.5) / 0.5
        pad = np.zeros((1, 3, IMG_H, IMG_W), dtype="float32")
        pad[0, :, :, :rw] = r
        return pad

    # ---- CTC greedy decode ----
    def _decode(self, logits):
        probs = logits[0]                    # [T, num_classes]
        idx = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        out, scores, prev = [], [], -1
        for t, i in enumerate(idx):
            if i != prev and i != 0:         # collapse repeats, drop blank(0)
                out.append(self.character[i] if i < len(self.character) else "")
                scores.append(conf[t])
            prev = i
        text = "".join(out)
        score = float(np.mean(scores)) if scores else 0.0
        return text, score

    def read(self, bgr):
        """Return (raw_text, confidence) for a BGR image array."""
        logits = self.sess.run(None, {self.inp: self._preprocess(bgr)})[0]
        return self._decode(logits)

    def predict(self, image, threshold=ACCEPT_THRESHOLD):
        """image: path or BGR array. Returns (serial, score, status).
        status: 'accepted' (>=threshold) | 'review' (below) | 'blank' (no read)."""
        bgr = cv2.imread(image) if isinstance(image, str) else image
        if bgr is None:
            return "", 0.0, "blank"
        text, score = self.read(bgr)
        if not text:
            return "", 0.0, "blank"
        status = "accepted" if score >= threshold else "review"
        return text, round(score, 4), status


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python meterocr.py <image.jpg> [more.jpg ...]")
        raise SystemExit(1)
    ocr = MeterOCR()
    for p in sys.argv[1:]:
        t, s, st = ocr.predict(p)
        print(f"{p}\t{t}\t{s}\t{st}")
