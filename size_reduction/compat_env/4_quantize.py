#!/usr/bin/env python3
"""
STEP 4 — STATIC INT8 quantization of the server ONNX.

Dynamic quantization does NOT shrink this conv-heavy model (proven). Static
quantization runs a small set of real crops through the model to calibrate
activation ranges, then quantizes Conv + MatMul weights -> real size reduction.

Run (with the venv python):
  size_reduction/compat_env/venv/bin/python size_reduction/compat_env/4_quantize.py
"""
import os, glob
import numpy as np
import cv2
import onnx
from onnxruntime.quantization import (CalibrationDataReader, quantize_static,
                                      QuantFormat, QuantType)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

FP32 = "size_reduction/models/server_fp32.onnx"
INT8 = "size_reduction/models/server_int8.onnx"
CALIB_DIR = "finetune/dataset/images"     # real training crops for calibration
N_CALIB = 200
IMG_H, IMG_W = 48, 320


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
    return pad[None]                      # [1,3,48,320]


class RecCalib(CalibrationDataReader):
    def __init__(self, input_name, files):
        self.input_name = input_name
        self.files = files
        self.i = 0

    def get_next(self):
        while self.i < len(self.files):
            x = preprocess(self.files[self.i])
            self.i += 1
            if x is not None:
                return {self.input_name: x}
        return None


def main():
    if not os.path.exists(FP32):
        raise SystemExit(f"missing {FP32} — run 3_export_v1.sh first")
    input_name = onnx.load(FP32).graph.input[0].name
    files = sorted(glob.glob(os.path.join(CALIB_DIR, "*.jpg")))[:N_CALIB]
    print(f"calibrating on {len(files)} crops (input '{input_name}') ...")

    quantize_static(
        FP32, INT8,
        calibration_data_reader=RecCalib(input_name, files),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )
    mb = lambda p: os.path.getsize(p) / 1e6
    print(f"\n=== DONE ===")
    print(f"  server FP32 ONNX: {mb(FP32):.1f} MB")
    print(f"  server INT8 ONNX: {mb(INT8):.1f} MB   <- deploy this (ONNX Runtime)")
    print(f"  saved to {INT8}")


if __name__ == "__main__":
    main()
