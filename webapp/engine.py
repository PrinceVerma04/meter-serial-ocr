#!/usr/bin/env python3
"""
Shared inference engine for the CPU / GPU benchmark web apps.

Loads all THREE meter-serial OCR models and runs each one on a chosen device
("cpu" or "gpu"), returning the extracted serial number, confidence and the
per-image inference time (ms).

The two front-ends `app_cpu.py` and `app_gpu.py` are thin: they just create an
`Engine("cpu")` or `Engine("gpu")` and serve the same web page. All the model
handling lives here so the CPU and GPU builds stay identical apart from the
device they pin.

Models (see MODELS/README.md):
  fp32  FP32 original      finetune/inference_v2/            (Paddle, 81 MB)
  qat   INT8 QAT           deploy/qat_int8/model/   (Paddle, 22 MB)
  ptq   INT8 PTQ           deploy/ptq_int8/model/   (ONNX,   21 MB)

Honest device reporting: INT8 models are CPU-targeted. On GPU the QAT Paddle
model raises (INT8 kernels need TensorRT) and the PTQ ONNX model falls back to
CPU. The engine catches this, keeps the app alive, and reports the *actual*
device each model ran on.
"""
import os
import time
import numpy as np

# numpy-alias shim (an old openpyxl on this box references removed np.float etc.)
for _a in ("float", "int", "bool", "object", "str"):
    if not hasattr(np, _a):
        setattr(np, _a, getattr(__builtins__, _a, None)
                if not isinstance(__builtins__, dict) else __builtins__.get(_a))
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_H, IMG_W = 48, 320
ACCEPT_THRESHOLD = 0.95
DICT_PATH = os.path.join(ROOT, "deploy/ptq_int8", "model", "ppocrv5_dict.txt")

# CTC vocabulary: blank at index 0, real chars, space appended last
_CHARS = ["blank"] + [l.rstrip("\n") for l in open(DICT_PATH, encoding="utf-8")] + [" "]

# ---- the three models, in display order ----
MODEL_SPECS = [
    {"key": "fp32", "label": "FP32 original", "size": "81 MB", "runtime": "Paddle",
     "kind": "paddle",
     "model": os.path.join(ROOT, "finetune/inference_v2/inference.json"),
     "params": os.path.join(ROOT, "finetune/inference_v2/inference.pdiparams")},
    {"key": "qat", "label": "INT8 QAT", "size": "22 MB", "runtime": "Paddle",
     "kind": "paddle",
     "model": os.path.join(ROOT, "deploy/qat_int8/model/meterocr_qat_int8.pdmodel"),
     "params": os.path.join(ROOT, "deploy/qat_int8/model/meterocr_qat_int8.pdiparams")},
    {"key": "ptq", "label": "INT8 PTQ", "size": "21 MB", "runtime": "ONNX",
     "kind": "onnx",
     "model": os.path.join(ROOT, "deploy/ptq_int8/model/meterocr_ptq_int8.onnx"),
     "params": None},
]


# ---------- shared image ops (identical preprocessing for every model) ----------
def preprocess(bgr):
    """Resize to 3x48x320, keep aspect ratio, pad right, normalize to [-1, 1]."""
    h, w = bgr.shape[:2]
    rw = min(IMG_W, max(1, int(np.ceil(IMG_H * w / h))))
    r = cv2.resize(bgr, (rw, IMG_H)).astype("float32").transpose(2, 0, 1) / 255.0
    r = (r - 0.5) / 0.5
    pad = np.zeros((1, 3, IMG_H, IMG_W), dtype="float32")
    pad[0, :, :, :rw] = r
    return pad


def ctc_decode(logits):
    """Greedy CTC decode -> (text, mean_confidence)."""
    probs = logits[0]                       # [T, num_classes]
    idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    out, scores, prev = [], [], -1
    for t, i in enumerate(idx):
        if i != prev and i != 0:            # collapse repeats, drop blank(0)
            out.append(_CHARS[i] if i < len(_CHARS) else "")
            scores.append(conf[t])
        prev = i
    text = "".join(out)
    return text, (float(np.mean(scores)) if scores else 0.0)


# ---------------------------- per-runtime loaders ----------------------------
class _PaddleModel:
    def __init__(self, spec, device):
        import paddle.inference as pi
        cfg = pi.Config(spec["model"], spec["params"])
        if device == "gpu":
            cfg.enable_use_gpu(1000, 0)
        else:
            cfg.disable_gpu()               # mkldnn/oneDNN stays OFF by default (buggy on this build)
        cfg.disable_glog_info()
        self.pred = pi.create_predictor(cfg)
        self.iname = self.pred.get_input_names()[0]
        self.oname = self.pred.get_output_names()[0]
        self.device_used = device

    def infer(self, x):
        h = self.pred.get_input_handle(self.iname)
        h.copy_from_cpu(x)
        self.pred.run()
        return self.pred.get_output_handle(self.oname).copy_to_cpu()


class _OnnxModel:
    def __init__(self, spec, device):
        import onnxruntime as ort
        want = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device == "gpu" else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(spec["model"], providers=want)
        self.inp = self.sess.get_inputs()[0].name
        # what ONNX Runtime actually registered (may have dropped CUDA)
        active = self.sess.get_providers()
        self.device_used = "gpu" if active and active[0] == "CUDAExecutionProvider" else "cpu"

    def infer(self, x):
        return self.sess.run(None, {self.inp: x})[0]


# --------------------------------- engine ---------------------------------
class Engine:
    """Loads all three models on one device and reads images through them."""

    def __init__(self, device):
        assert device in ("cpu", "gpu")
        self.device = device
        self.models = {}      # key -> loaded model (only the ones that work)
        self.status = []      # per-model availability info for the UI

        for spec in MODEL_SPECS:
            info = {"key": spec["key"], "label": spec["label"], "size": spec["size"],
                    "runtime": spec["runtime"], "requested": device,
                    "available": False, "device_used": None, "note": ""}
            try:
                m = (_PaddleModel(spec, device) if spec["kind"] == "paddle"
                     else _OnnxModel(spec, device))
                # warm up once — this is where INT8-on-GPU actually fails
                dummy = np.zeros((1, 3, IMG_H, IMG_W), dtype="float32")
                m.infer(dummy)
                self.models[spec["key"]] = m
                info["available"] = True
                info["device_used"] = m.device_used
                if device == "gpu" and m.device_used == "cpu":
                    info["note"] = "requested GPU, ONNX Runtime fell back to CPU (INT8 CUDA unsupported)"
            except Exception as e:
                info["note"] = self._explain(spec, device, e)
            self.status.append(info)

    @staticmethod
    def _explain(spec, device, e):
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"
        if device == "gpu" and spec["key"] == "qat":
            return "GPU unsupported for INT8 QAT Paddle model (needs TensorRT INT8). " + msg
        return msg

    def available_models(self):
        return [s for s in self.status if s["available"]]

    def read_image(self, bgr):
        """Run every available model on one BGR image.
        Returns list of {key,label,serial,score,status,ms,device}."""
        x = preprocess(bgr)
        results = []
        for spec in MODEL_SPECS:
            m = self.models.get(spec["key"])
            if m is None:
                results.append({"key": spec["key"], "label": spec["label"],
                                "serial": "", "score": 0.0, "status": "unavailable",
                                "ms": None, "device": None})
                continue
            t0 = time.perf_counter()
            logits = m.infer(x)
            ms = (time.perf_counter() - t0) * 1000.0
            text, score = ctc_decode(logits)
            status = "blank" if not text else ("accepted" if score >= ACCEPT_THRESHOLD else "review")
            results.append({"key": spec["key"], "label": spec["label"],
                            "serial": text, "score": round(score, 4),
                            "status": status, "ms": round(ms, 2),
                            "device": m.device_used})
        return results
