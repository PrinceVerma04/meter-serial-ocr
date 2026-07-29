#!/usr/bin/env python3
"""
Meter serial-number OCR — QAT INT8 model, self-contained inference (PaddlePaddle).

Runs the 22 MB INT8 QAT recognizer via Paddle Inference. Needs `paddlepaddle`
(any recent 3.x). More accurate than the ONNX PTQ build (~92.8% vs ~85%).

Input: a cropped image of the meter serial area (BGR, as read by cv2.imread).
Output: (serial_text, confidence) + a status from the confidence gate.

Usage:
    from meterocr import MeterOCR
    ocr = MeterOCR()                       # loads model/ next to this file
    text, score, status = ocr.predict("crop.jpg")
"""
import os
import numpy as np
import cv2

# numpy-alias shim (harmless; helps on boxes with an old openpyxl in the import chain)
for _a in ("float", "int", "bool", "object", "str"):
    if not hasattr(np, _a):
        setattr(np, _a, getattr(__builtins__, _a, None)
                if not isinstance(__builtins__, dict) else __builtins__.get(_a))

import paddle.inference as pi

HERE = os.path.dirname(os.path.abspath(__file__))
PDMODEL = os.path.join(HERE, "model", "meterocr_qat_int8.pdmodel")
PDIPARAMS = os.path.join(HERE, "model", "meterocr_qat_int8.pdiparams")
DICT = os.path.join(HERE, "model", "ppocrv5_dict.txt")

ACCEPT_THRESHOLD = 0.95          # >= -> auto-accept; below -> human review
IMG_H, IMG_W = 48, 320


class MeterOCR:
    def __init__(self, pdmodel=PDMODEL, pdiparams=PDIPARAMS, dict_path=DICT, use_gpu=False):
        cfg = pi.Config(pdmodel, pdiparams)
        if use_gpu:
            cfg.enable_use_gpu(1000, 0)
        else:
            cfg.disable_gpu()
        cfg.disable_glog_info()
        self.pred = pi.create_predictor(cfg)
        self.iname = self.pred.get_input_names()[0]
        self.oname = self.pred.get_output_names()[0]
        chars = [l.rstrip("\n") for l in open(dict_path, encoding="utf-8")]
        self.character = ["blank"] + chars + [" "]        # CTC blank at 0, space last

    def _preprocess(self, bgr):
        h, w = bgr.shape[:2]
        rw = min(IMG_W, int(np.ceil(IMG_H * w / h)))
        r = cv2.resize(bgr, (rw, IMG_H)).astype("float32").transpose(2, 0, 1) / 255.0
        r = (r - 0.5) / 0.5
        pad = np.zeros((1, 3, IMG_H, IMG_W), dtype="float32")
        pad[0, :, :, :rw] = r
        return pad

    def _decode(self, logits):
        probs = logits[0]
        idx = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        out, scores, prev = [], [], -1
        for t, i in enumerate(idx):
            if i != prev and i != 0:
                out.append(self.character[i] if i < len(self.character) else "")
                scores.append(conf[t])
            prev = i
        text = "".join(out)
        return text, (float(np.mean(scores)) if scores else 0.0)

    def read(self, bgr):
        """Return (raw_text, confidence) for a BGR image array."""
        ih = self.pred.get_input_handle(self.iname)
        ih.copy_from_cpu(self._preprocess(bgr))
        self.pred.run()
        logits = self.pred.get_output_handle(self.oname).copy_to_cpu()
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
