#!/usr/bin/env bash
# STEP 3 — export the retrained checkpoint to the convertible PIR v1 inference model
# (using the venv's Paddle 3.0.0), then convert to ONNX FP32.
#
# Run:  bash size_reduction/compat_env/3_export_v1.sh
set -e
cd "/home/ganesh/Desktop/Prince number detection in meter"
ENV=size_reduction/compat_env/venv
REPO=~/.local/lib/python3.10/site-packages/paddlex/repo_manager/repos/PaddleOCR
DIR=size_reduction/compat_env

echo "### export -> inference (PIR v1) ###"
PYTHONPATH=$REPO "$ENV/bin/python" $REPO/tools/export_model.py \
  -c "$DIR/server_train.yml" \
  -o Global.pretrained_model="$DIR/output/best_accuracy" \
     Global.save_inference_dir="$DIR/inference_server_v1"

echo "-- PIR version (must be 1) --"
head -c 120 "$DIR/inference_server_v1/inference.json" | grep -oE '"version":[0-9]+'

echo "### paddle2onnx -> ONNX FP32 ###"
mkdir -p size_reduction/models
"$ENV/bin/paddle2onnx" --model_dir "$DIR/inference_server_v1" \
  --model_filename inference.json --params_filename inference.pdiparams \
  --save_file size_reduction/models/server_fp32.onnx --opset_version 13
ls -la size_reduction/models/server_fp32.onnx | awk '{printf "server_fp32.onnx: %.1f MB\n",$5/1e6}'
