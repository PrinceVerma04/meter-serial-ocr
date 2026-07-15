#!/usr/bin/env python3
"""
ONNX inference wrapper for a PP-OCRv5 recognizer (CTC head).

Replicates PaddleOCR's rec preprocessing (resize to 3x48x320, normalize) and
CTC greedy decoding so a quantized ONNX model produces the same kind of output
as the Paddle TextRecognition engine — needed to test the reduced models.
"""
import numpy as np, cv2, onnxruntime as ort

DICT = ("/home/ganesh/.local/lib/python3.10/site-packages/paddlex/repo_manager/"
        "repos/PaddleOCR/ppocr/utils/dict/ppocrv5_dict.txt")


class OnnxRec:
    def __init__(self, onnx_path, dict_path=DICT, use_space_char=True,
                 img_h=48, img_w=320):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.H, self.W = img_h, img_w
        chars = [l.rstrip("\n") for l in open(dict_path, encoding="utf-8")]
        if use_space_char:
            chars = chars + [" "]
        self.character = ["blank"] + chars          # CTC blank at index 0

    def _pre(self, bgr):
        h, w = bgr.shape[:2]
        ratio = w / float(h)
        rw = min(self.W, int(np.ceil(self.H * ratio)))
        r = cv2.resize(bgr, (rw, self.H)).astype("float32")
        r = r.transpose(2, 0, 1) / 255.0
        r = (r - 0.5) / 0.5
        pad = np.zeros((3, self.H, self.W), dtype="float32")
        pad[:, :, :rw] = r
        return pad[None]                            # [1,3,48,320]

    def _decode(self, logits):
        # logits: [1, T, C]
        probs = logits[0]
        idx = probs.argmax(axis=1)
        mx = probs.max(axis=1)
        out, scores, prev = [], [], -1
        for t, i in enumerate(idx):
            if i != prev and i != 0:                # collapse repeats, drop blank(0)
                out.append(self.character[i] if i < len(self.character) else "")
                scores.append(mx[t])
            prev = i
        text = "".join(out)
        score = float(np.mean(scores)) if scores else 0.0
        return text, score

    def read(self, bgr):
        logits = self.sess.run(None, {self.inp: self._pre(bgr)})[0]
        return self._decode(logits)
