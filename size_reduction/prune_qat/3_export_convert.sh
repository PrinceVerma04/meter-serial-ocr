#!/usr/bin/env bash
# STEP 3 — convert the QAT model to ONNX and report the final size.
# The QAT export already yields an INT8-simulated inference model; converting to
# ONNX + (if needed) static repack gives the deployable file.
set -e
cd "/home/ganesh/Desktop/Prince number detection in meter"
ENV=size_reduction/compat_env/venv
DIR=size_reduction/prune_qat

# QAT inference model -> ONNX
"$ENV/bin/paddle2onnx" --model_dir "$DIR/qat_inference" \
  --model_filename inference.json --params_filename inference.pdiparams \
  --save_file "$DIR/prune_qat_fp.onnx" --opset_version 13 || {
    echo "paddle2onnx failed — check the QAT inference model format/PIR version"; exit 1; }

echo "=== SIZES ==="
echo "original (Paddle) : 81 MB"
echo "prior INT8 (ONNX) : $(du -h size_reduction/models/server_int8.onnx | cut -f1)"
echo "prune+QAT (ONNX)  : $(du -h "$DIR/prune_qat_fp.onnx" | cut -f1)"
echo
echo "Verify accuracy with size_reduction/onnx_reader.py (see compat_env/README.md step 5),"
echo "then compare against model_compare_results.csv ground truth."
