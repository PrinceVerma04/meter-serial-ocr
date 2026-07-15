#!/usr/bin/env python3
"""Paddle-Inference reader for a PP-OCRv5 rec model (FP32 or QAT-INT8).

Loads a legacy .pdmodel/.pdiparams inference model and reads a crop with the
same preprocessing + CTC decode as onnx_reader.py, so Paddle and ONNX models
produce comparable outputs.
"""
import os, numpy as np, cv2
import paddle.inference as pi

DICT = ("/home/ganesh/.local/lib/python3.10/site-packages/paddlex/repo_manager/"
        "repos/PaddleOCR/ppocr/utils/dict/ppocrv5_dict.txt")


class PaddleRec:
    def __init__(self, model_dir, dict_path=DICT, use_gpu=False, img_h=48, img_w=320):
        cfg = pi.Config(os.path.join(model_dir, "inference.pdmodel"),
                        os.path.join(model_dir, "inference.pdiparams"))
        if use_gpu:
            cfg.enable_use_gpu(1000, 0)
        else:
            cfg.disable_gpu()
        cfg.disable_glog_info()
        self.pred = pi.create_predictor(cfg)
        self.iname = self.pred.get_input_names()[0]
        self.oname = self.pred.get_output_names()[0]
        self.H, self.W = img_h, img_w
        chars = [l.rstrip("\n") for l in open(dict_path, encoding="utf-8")]
        self.character = ["blank"] + chars + [" "]

    def _pre(self, bgr):
        h, w = bgr.shape[:2]
        rw = min(self.W, int(np.ceil(self.H * w / h)))
        r = cv2.resize(bgr, (rw, self.H)).astype("float32").transpose(2, 0, 1) / 255.0
        r = (r - 0.5) / 0.5
        pad = np.zeros((1, 3, self.H, self.W), dtype="float32")
        pad[0, :, :, :rw] = r
        return pad

    def _decode(self, logits):
        probs = logits[0]
        idx = probs.argmax(1)
        mx = probs.max(1)
        out, scores, prev = [], [], -1
        for t, i in enumerate(idx):
            if i != prev and i != 0:
                out.append(self.character[i] if i < len(self.character) else "")
                scores.append(mx[t])
            prev = i
        return "".join(out), (float(np.mean(scores)) if scores else 0.0)

    def read(self, bgr):
        ih = self.pred.get_input_handle(self.iname)
        ih.copy_from_cpu(self._pre(bgr))
        self.pred.run()
        logits = self.pred.get_output_handle(self.oname).copy_to_cpu()
        return self._decode(logits)
