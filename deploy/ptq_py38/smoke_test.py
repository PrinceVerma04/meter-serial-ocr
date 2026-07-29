#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-install verification. Run this FIRST on the server, before wiring anything up.

    python3 smoke_test.py

Checks, in order:
  1. the interpreter really is Python 3.8+
  2. every import resolves, with versions that are valid on 3.8
  3. the model file is present and the right size
  4. the session loads
  5. sample.jpg decodes to the expected serial
  6. latency over 20 runs is sane

Exit code 0 = safe to start the service. Non-zero = do not proceed.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# sample.jpg is one of the 1,004 human-verified crops; WHL059419 is its confirmed
# ground-truth label AND what this model reads. If this fails, the model file or
# the preprocessing has changed — do not deploy.
EXPECTED_SERIAL = "WHL059419"
fails = []


def check(label, fn):
    try:
        detail = fn()
        print("  [ok]   %s%s" % (label, (" — %s" % detail) if detail else ""))
    except Exception as e:
        print("  [FAIL] %s — %s: %s" % (label, type(e).__name__, e))
        fails.append(label)


print("meterocr PTQ INT8 (Python 3.8 build) — smoke test\n")

print("1. interpreter")
def _py():
    v = sys.version_info
    if v[:2] < (3, 8):
        raise RuntimeError("Python %d.%d is too old — this package needs 3.8+"
                           % (v[0], v[1]))
    return "Python %d.%d.%d" % (v[0], v[1], v[2])
check("Python 3.8 or newer", _py)

print("\n2. imports")
def _imports():
    import numpy, cv2, onnxruntime
    return "numpy %s, cv2 %s, ort %s" % (numpy.__version__, cv2.__version__,
                                         onnxruntime.__version__)
check("numpy / opencv / onnxruntime import", _imports)

def _ort_floor():
    import onnxruntime
    parts = onnxruntime.__version__.split(".")
    ver = (int(parts[0]), int(parts[1]))
    if ver < (1, 13):
        raise RuntimeError(
            "onnxruntime %s is too old — this model needs >=1.13 for the "
            "com.microsoft:QLinearSoftmax operator. Install onnxruntime==1.16.3"
            % onnxruntime.__version__)
    return onnxruntime.__version__
check("onnxruntime >= 1.13 (QLinearSoftmax support)", _ort_floor)

def _tf_isolation():
    """If TensorFlow shares this interpreter, numpy cannot satisfy both."""
    import importlib.util
    if importlib.util.find_spec("tensorflow") is None:
        return "no TensorFlow here — correctly isolated"
    import tensorflow as tf
    import numpy
    raise RuntimeError(
        "TensorFlow %s is installed in THIS interpreter alongside numpy %s. "
        "TF 2.4.x needs numpy~=1.19.2 but this model needs numpy>=1.21.6 — they "
        "cannot coexist. Run the OCR in its own venv and call it over HTTP."
        % (tf.__version__, numpy.__version__))
check("TensorFlow not sharing this interpreter", _tf_isolation)

def _numpy_ok():
    import numpy
    major = int(numpy.__version__.split(".")[0])
    minor = int(numpy.__version__.split(".")[1])
    if sys.version_info[:2] == (3, 8) and (major, minor) > (1, 24):
        raise RuntimeError(
            "numpy %s cannot be correct on Python 3.8 (1.24.4 is the last "
            "supported release) — check your install" % numpy.__version__)
    return numpy.__version__
check("numpy version valid for this interpreter", _numpy_ok)

print("\n3. model files")
def _model_file():
    p = os.path.join("model", "meterocr_ptq_int8.onnx")
    mb = os.path.getsize(p) / 1e6
    if mb < 15:
        raise RuntimeError("%s is only %.1f MB — truncated upload?" % (p, mb))
    return "%.1f MB" % mb
check("model/meterocr_ptq_int8.onnx present", _model_file)

def _dict_file():
    with open(os.path.join("model", "ppocrv5_dict.txt"), encoding="utf-8") as fh:
        return "%d chars" % sum(1 for _ in fh)
check("model/ppocrv5_dict.txt present", _dict_file)

print("\n4. session load")
from meterocr import MeterOCR
holder = {}
def _load():
    t0 = time.time()
    holder["ocr"] = MeterOCR(threads=2)
    return "%.2f s" % (time.time() - t0)
check("InferenceSession created", _load)

print("\n5. inference on the bundled sample")
def _predict():
    serial, score, status = holder["ocr"].predict("sample.jpg")
    if serial != EXPECTED_SERIAL:
        raise RuntimeError(
            "got %r (conf %s), expected %r — the model or preprocessing does "
            "not match this build" % (serial, score, EXPECTED_SERIAL))
    return "%r conf=%s status=%s" % (serial, score, status)
check("sample.jpg reads %r" % EXPECTED_SERIAL, _predict)

print("\n6. latency (20 runs)")
def _latency():
    import cv2
    bgr = cv2.imread("sample.jpg")
    ocr = holder["ocr"]
    ocr.read(bgr)                                  # warm-up, excluded
    t0 = time.time()
    for _ in range(20):
        ocr.read(bgr)
    ms = (time.time() - t0) / 20 * 1000
    if ms > 2000:
        raise RuntimeError("%.0f ms/img is far too slow — undersized instance?" % ms)
    return "%.0f ms/image" % ms
check("steady-state latency", _latency)

print()
if fails:
    print("FAILED (%d): %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("All checks passed — safe to start the service.")
