#!/usr/bin/env bash
# ============================================================================
# PTQ INT8 quantization for v4. Two steps, same recipe v3 needed
# (size_reduction/quantize_v3_ptq.py header explains why):
#
#   A) re-export the v4 checkpoint with Paddle 3.0.0 -> PIR v1 format, the
#      only format paddle2onnx can convert (the training venv's Paddle 3.3.1
#      export can't be read by paddle2onnx — SIGABRT). Export is a single
#      forward pass, so CPU-only is fine here, no need for the Blackwell GPU
#      wheel used for training.
#   B) fold constants + static INT8 quantize (size_reduction/quantize_v4_ptq.py)
#
# Run:  bash scripts/quantize_v4_ptq.sh
# Idempotent — safe to re-run.
# ============================================================================
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
ENV="$ROOT/size_reduction/compat_env/venv"
REPO="$ROOT/finetune/PaddleOCR_repo"
CFG="$ROOT/finetune/rec_retrain_v4.yml"
CKPT="$ROOT/finetune/output_retrain_v4/best_accuracy"
CONVERTIBLE="$ROOT/size_reduction/compat_env/inference_v4_convertible"

echo "=== 1/2: Paddle 3.0.0 CPU venv (export-only, no GPU needed) ==="
if [ ! -d "$ENV" ]; then
  python3.10 -m venv "$ENV"
fi
"$ENV/bin/python" -m pip install -q --upgrade pip
if ! "$ENV/bin/python" -c "import paddle" 2>/dev/null; then
  "$ENV/bin/pip" install -q paddlepaddle==3.0.0 paddle2onnx onnx onnxruntime
  "$ENV/bin/pip" install -q "numpy<2" pyyaml "opencv-python<4.10" tqdm shapely \
    scikit-image pyclipper lmdb imgaug rapidfuzz albumentations Pillow scipy \
    requests python-Levenshtein
fi
"$ENV/bin/python" -c "import paddle; print('paddle', paddle.__version__)"

if [ ! -f "$REPO/tools/export_model.py" ]; then
  echo "!! $REPO not found — run scripts/retrain_all.sh first (it clones this repo)"
  exit 1
fi

echo "=== 2/2: re-export v4 checkpoint -> PIR v1 convertible format ==="
PYTHONPATH="$REPO" "$ENV/bin/python" "$REPO/tools/export_model.py" -c "$CFG" \
  -o Global.checkpoints="$CKPT" \
     Global.save_inference_dir="$CONVERTIBLE" \
     Global.use_gpu=False

echo "=== running quantization (fold constants + static INT8) ==="
"$ENV/bin/python" "$ROOT/size_reduction/quantize_v4_ptq.py"

echo ""
echo "Next: log the result in TRAINING_LOG.md, then score"
echo "  size_reduction/models/meterocr_v4_int8_ptq.onnx"
echo "against the current shipped model (deploy/ptq_int8/model/meterocr_ptq_int8.onnx,"
echo "which is v3) before replacing it in deploy/."
