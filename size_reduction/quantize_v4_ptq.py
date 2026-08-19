#!/usr/bin/env python3
"""
v4 PTQ INT8 — quantization of the v4 retrain (12,886-image dataset, trained
2026-08-19). Same recipe as size_reduction/quantize_v3_ptq.py — see that
file's header for why each fix is needed (Paddle format, path-space
constant-folding bug, per-tensor vs per-channel). Run with the compat_env
venv (Paddle 3.0.0, CPU) via scripts/quantize_v4_ptq.sh, which also does the
required STEP 0 re-export first.

Produces:  size_reduction/models/meterocr_v4_fp32.onnx      (folded, CTC-only)
           size_reduction/models/meterocr_v4_int8_ptq.onnx  (deploy this)
"""
import os, sys, glob, subprocess, random
import numpy as np
import cv2
import onnx
from onnxruntime.quantization import (CalibrationDataReader, quantize_static,
                                      QuantFormat, QuantType)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SRC = "size_reduction/compat_env/inference_v4_convertible"   # PIR-v1 export (STEP 0)
OUT = "size_reduction/models"
RAW_FP32 = f"{OUT}/_server_v4_fp32_raw.onnx"                  # weights as Constants
FP32 = f"{OUT}/meterocr_v4_fp32.onnx"                         # folded, CTC-only
INT8 = f"{OUT}/meterocr_v4_int8_ptq.onnx"                     # deploy this
CALIB_DIR = "finetune/dataset_retrain/images"
N_CALIB = 200
IMG_H, IMG_W = 48, 320
os.makedirs(OUT, exist_ok=True)

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


# STEP 1) Paddle (PIR v1) -> ONNX FP32. Constant folding fails here (space in
#         the project path breaks paddle2onnx's subprocess); STEP 2 folds instead.
if not os.path.isdir(SRC):
    raise SystemExit(f"missing {SRC} — run scripts/quantize_v4_ptq.sh (does the re-export first)")
print("[1/3] paddle2onnx export ...", flush=True)
# use the console-script next to this interpreter (venv/bin/), not a bare
# "paddle2onnx" lookup on $PATH -- the venv isn't activated in this subprocess
PADDLE2ONNX = os.path.join(os.path.dirname(sys.executable), "paddle2onnx")
subprocess.run([PADDLE2ONNX, "--model_dir", SRC,
                "--model_filename", "inference.json",
                "--params_filename", "inference.pdiparams",
                "--save_file", RAW_FP32, "--opset_version", "13"], check=True)
print(f"      raw ONNX: {mb(RAW_FP32):.1f} MB (weights as Constant nodes)")

# STEP 2) Fold constants in-Python (path-safe). Prunes the unused NRTR head too.
print("[2/3] constant folding (quant_pre_process) ...", flush=True)
quant_pre_process(RAW_FP32, FP32, skip_symbolic_shape=True)
os.remove(RAW_FP32)
print(f"      folded FP32 ONNX: {mb(FP32):.1f} MB")

# STEP 3) Static INT8 (QDQ, PER-TENSOR — per-channel corrupts long serials),
#         calibrated on a random crop sample from the v4 training set.
print("[3/3] static INT8 quantization ...", flush=True)
input_name = onnx.load(FP32).graph.input[0].name
files = glob.glob(os.path.join(CALIB_DIR, "*.jpg"))
random.seed(42); random.shuffle(files); files = files[:N_CALIB]
print(f"      calibrating on {len(files)} crops (input '{input_name}') ...", flush=True)
quantize_static(FP32, INT8, calibration_data_reader=RecCalib(input_name, files),
                quant_format=QuantFormat.QDQ, per_channel=False,
                weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"])
print("\n=== DONE ===")
print(f"  v4 FP32 ONNX (CTC-only): {mb(FP32):.1f} MB   {FP32}")
print(f"  v4 INT8 ONNX (deploy)  : {mb(INT8):.1f} MB   {INT8}")
