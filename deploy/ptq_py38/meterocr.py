#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meter serial-number OCR — PTQ INT8 ONNX model. Self-contained inference core.

PYTHON 3.8 COMPATIBLE. No walrus in hot paths, no f-string '=' specifier, no
dict/list generic annotations (list[str]), no match statement — nothing that
needs 3.9+.

Runs the 21 MB INT8 recognizer on ONNX Runtime (CPU). No PaddlePaddle, no torch,
no TensorFlow. Input is a crop of the meter serial area.

    from meterocr import MeterOCR
    ocr = MeterOCR()                       # loads model/ next to this file
    ocr.predict("crop.jpg")                # -> ('WHL059419', 0.9913, 'accepted')

Thread-safety: ONNX Runtime sessions are thread-safe for concurrent run() calls,
so ONE MeterOCR instance can serve many request threads. Build it once at process
start — never per request (session creation costs ~1 s).
"""
from __future__ import absolute_import, division, print_function

import os
import re

import numpy as np
import cv2
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("METEROCR_MODEL", os.path.join(HERE, "model", "meterocr_ptq_int8.onnx"))
DICT = os.environ.get("METEROCR_DICT", os.path.join(HERE, "model", "ppocrv5_dict.txt"))

# reads at/above this confidence are auto-accepted; below -> route to human review
ACCEPT_THRESHOLD = float(os.environ.get("METEROCR_THRESHOLD", "0.95"))
IMG_H, IMG_W = 48, 320


def normalize(text):
    """Tidy a raw reading: uppercase, alphanumeric, single-spaced.

    This is the exact normalization used when the model was benchmarked at
    97.01% on the 1,004-image verified set. Keep it on so production output
    matches the measured numbers.
    """
    return re.sub(r"[^A-Z0-9]+", " ", (text or "").upper()).strip()


class MeterOCR(object):
    def __init__(self, model_path=MODEL, dict_path=DICT, providers=None, threads=None):
        if not os.path.exists(model_path):
            raise IOError("model not found: %s" % model_path)
        if not os.path.exists(dict_path):
            raise IOError("dictionary not found: %s" % dict_path)

        opts = ort.SessionOptions()
        # Pin thread counts so N workers don't each spawn a full thread pool
        # and thrash a small EC2 box.
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.sess = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

        with open(dict_path, "r", encoding="utf-8") as fh:
            chars = [l.rstrip("\n") for l in fh]
        # CTC vocabulary: blank at index 0, space appended last
        self.character = ["blank"] + chars + [" "]

    # ---- preprocessing (must match training: 3x48x320, /255, then (x-0.5)/0.5) ----
    def _preprocess(self, bgr):
        h, w = bgr.shape[:2]
        rw = min(IMG_W, max(1, int(np.ceil(IMG_H * w / float(h)))))
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

    def predict(self, image, threshold=None, clean=True):
        """image: path, bytes, or BGR array. Returns (serial, score, status).

        status: 'accepted' (score >= threshold) | 'review' (below) | 'blank' (no read).
        Only trust 'accepted' without human eyes — that is what the gate is for.
        """
        threshold = ACCEPT_THRESHOLD if threshold is None else threshold

        if isinstance(image, str):
            bgr = cv2.imread(image)
        elif isinstance(image, (bytes, bytearray)):
            bgr = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        else:
            bgr = image

        if bgr is None or getattr(bgr, "size", 0) == 0:
            return "", 0.0, "blank"

        text, score = self.read(bgr)
        if clean:
            text = normalize(text)
        if not text:
            return "", 0.0, "blank"
        return text, round(score, 4), "accepted" if score >= threshold else "review"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 meterocr.py <image.jpg> [more.jpg ...]")
        raise SystemExit(1)
    ocr = MeterOCR()
    for p in sys.argv[1:]:
        t, s, st = ocr.predict(p)
        print("%s\t%s\t%s\t%s" % (p, t, s, st))
