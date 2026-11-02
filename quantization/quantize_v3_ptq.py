#!/usr/bin/env python3
"""
v3 PTQ INT8 — the reproducible quantization of the v3 retrain (run 2026-07-25).

RECORD OF HOW THE DEPLOYED v3 PTQ WAS MADE. Its outputs already exist and are kept:
           models/v3/meterocr_v3_fp32.onnx      (84 MB, CTC-only FP32)
           models/v3/meterocr_v3_int8_ptq.onnx  (21 MB, real INT8 = deploy/ptq_int8 model)
The script refuses to run while they exist. For a new version, copy the pattern of
quantization/quantize_v4_ptq.py (one command: export -> ONNX -> INT8 -> scoring).

--------------------------------------------------------------------------------
WHY yesterday's attempt failed (and the two fixes here)
--------------------------------------------------------------------------------
1. Wrong source + wrong Paddle. `finetune/inference_v3/` is exported by the system
   Paddle 3.3.1, whose format paddle2onnx CANNOT read (SIGABRT). Re-export the
   checkpoint with the compat_env Paddle 3.0.0 first (PIR v1 format). See STEP 0.

2. The project path contains a SPACE ("Prince number detection in meter"), which
   breaks paddle2onnx's internal constant-folding subprocess:
     [W] Constant folding pass failed ... No such file or directory: '.../Prince'
   => weights stay as ONNX *Constant nodes* (not initializers), so quantize_static
      has nothing to quantize and the INT8 file never shrinks (stayed 205 MB).
   FIX: fold constants in-Python via onnxruntime `quant_pre_process` (STEP 2).
   Folding also prunes the unused NRTR head -> 205 MB drops to 81 MB (CTC-only).

3. per_channel=True corrupts this folded graph (wrong quant axis -> long serials
   collapse, e.g. SS11135228 -> SS1; only ~47% agreement with FP32). Use
   per_channel=False (per-tensor) -> ~78% agreement, long serials preserved.
   Calibrate on a RANDOM sample (not sorted[:N], which is one capture batch).

--------------------------------------------------------------------------------
STEP 0 (run ONCE, by hand, with the compat_env venv) — export a convertible model:

  (2026-07-25 paths; the v3 training checkpoint itself no longer exists — its trainable
   weights were rebuilt on 2026-11-02 as models/v3/meterocr_v3_train_weights.pdparams)
  OLD="/home/coral/Desktop/Prince/Prince number detection in meter"
  REPO="$OLD/finetune/PaddleOCR_repo";  VENV="$OLD/size_reduction/compat_env/venv"
  PYTHONPATH="$REPO" $VENV/bin/python "$REPO/tools/export_model.py" \
    -c training/rec_v3.yml \
    -o Global.pretrained_model=models/v3/meterocr_v3_train_weights \
       Global.save_inference_dir=quantization/compat_env/inference_v3_convertible

Then run THIS script (STEPS 1-3) with the same venv:
  "$VENV/bin/python" quantization/quantize_v3_ptq.py
--------------------------------------------------------------------------------
"""
import os, glob, subprocess, random
import numpy as np
import cv2
import onnx
from onnxruntime.quantization import (CalibrationDataReader, quantize_static,
                                      QuantFormat, QuantType)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SRC = "quantization/compat_env/inference_v3_convertible"   # PIR-v1 export (STEP 0)
OUT = "models/v3"
RAW_FP32 = f"{OUT}/_server_v3_fp32_raw.onnx"                  # weights as Constants
FP32 = f"{OUT}/meterocr_v3_fp32.onnx"                         # folded, CTC-only
INT8 = f"{OUT}/meterocr_v3_int8_ptq.onnx"                     # deploy this
CALIB_DIR = "datasets/v3_train/images"
N_CALIB = 200
IMG_H, IMG_W = 48, 320
os.makedirs(OUT, exist_ok=True)
if os.path.exists(INT8) or os.path.exists(FP32):
    raise SystemExit(f"{INT8} already exists (the deployed v3 PTQ) - not overwriting it")

mb = lambda p: os.path.getsize(p) / 1e6 if os.path.exists(p) else 0


def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    rw = min(IMG_W, int(np.ceil(IMG_H * w / h)))
    r = cv2.resize(img, (rw, IMG_H)).astype("float32").transpose(2, 0, 1) / 255.0
    r = (r - 0.5) / 0.5
    pad = np.zeros((3, IMG_H, IMG_W), dtype="float32")
    pad[:, :, :rw] = r
    return pad[None]


class RecCalib(CalibrationDataReader):
    def __init__(self, input_name, files):
        self.input_name, self.files, self.i = input_name, files, 0

    def get_next(self):
        while self.i < len(self.files):
            x = preprocess(self.files[self.i]); self.i += 1
            if x is not None:
                return {self.input_name: x}
        return None


# STEP 1) Paddle (PIR v1) -> ONNX FP32.  NOTE: constant folding fails here due to
#         the space in the path; that's expected — STEP 2 folds in-Python instead.
if not os.path.isdir(SRC):
    raise SystemExit(f"missing {SRC} — run STEP 0 (export_model.py) first")
print("[1/3] paddle2onnx export ...", flush=True)
subprocess.run(["paddle2onnx", "--model_dir", SRC,
                "--model_filename", "inference.json",
                "--params_filename", "inference.pdiparams",
                "--save_file", RAW_FP32, "--opset_version", "13"], check=True)
print(f"      raw ONNX: {mb(RAW_FP32):.1f} MB (weights as Constant nodes)")

# STEP 2) Fold constants in-Python (path-safe). Turns Constants -> initializers and
#         prunes the unused NRTR head. 205 MB -> ~81 MB, CTC-only.
print("[2/3] constant folding (quant_pre_process) ...", flush=True)
quant_pre_process(RAW_FP32, FP32, skip_symbolic_shape=True)
os.remove(RAW_FP32)
print(f"      folded FP32 ONNX: {mb(FP32):.1f} MB")

# STEP 3) Static INT8 (QDQ, PER-TENSOR), calibrated on a random crop sample -> ~21 MB.
print("[3/3] static INT8 quantization ...", flush=True)
input_name = onnx.load(FP32).graph.input[0].name
files = glob.glob(os.path.join(CALIB_DIR, "*.jpg"))
random.seed(42); random.shuffle(files); files = files[:N_CALIB]   # diverse, not one batch
print(f"      calibrating on {len(files)} crops (input '{input_name}') ...", flush=True)
quantize_static(FP32, INT8, calibration_data_reader=RecCalib(input_name, files),
                quant_format=QuantFormat.QDQ, per_channel=False,   # per-tensor (per-channel breaks it)
                weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"])
print("\n=== DONE ===")
print(f"  v3 FP32 ONNX (CTC-only): {mb(FP32):.1f} MB   {FP32}")
print(f"  v3 INT8 ONNX (deploy)  : {mb(INT8):.1f} MB   {INT8}")
