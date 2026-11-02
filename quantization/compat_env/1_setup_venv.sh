#!/usr/bin/env bash
# STEP 1 — build the Paddle-3.0.0 venv that exports the convertible PIR v1 format.
# (Already partly done in your session; this is idempotent and safe to re-run.)
#
# Run:  bash quantization/compat_env/1_setup_venv.sh
# The venv lives where quantization/quantize_v4_ptq.py expects it (COMPAT_PY there):
set -e
ENV="/home/coral/Desktop/Prince/Prince number detection in meter/size_reduction/compat_env/venv"

[ -d "$ENV" ] || python3 -m venv "$ENV"
"$ENV/bin/python" -m pip install -q --upgrade pip

echo "installing Paddle 3.0.0 (CPU) + converter + repo deps ..."
"$ENV/bin/pip" install -q "paddlepaddle==3.0.0" paddle2onnx onnx onnxruntime
# repo training/export deps (numpy pinned <2 for Paddle 3.0.0)
"$ENV/bin/pip" install -q "numpy<2" pyyaml "opencv-python<4.10" tqdm shapely scikit-image \
  pyclipper lmdb imgaug rapidfuzz albumentations Pillow scipy

"$ENV/bin/python" - <<'PY'
import paddle
print("paddle", paddle.__version__, "| GPU:", paddle.is_compiled_with_cuda())
PY
echo "OK. (For GPU speed, see 2_retrain.sh header to install paddlepaddle-gpu==3.0.0 instead.)"
