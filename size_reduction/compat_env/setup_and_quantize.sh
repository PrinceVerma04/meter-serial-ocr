#!/usr/bin/env bash
#
# Build a Paddle-compatible venv to QUANTIZE the server model.
#
# Root cause: Paddle 3.3.1 exports PIR format v4; paddle2onnx 2.1.0 only supports
# PIR v1 (what the stock/convertible models use). Fix = an older Paddle (3.0.0)
# that exports v1, then convert -> INT8.
#
# Because our trained checkpoint was deleted, we RE-TRAIN the server in this env
# (pretrained + dataset are already present), then export v1 -> ONNX -> INT8.
#
# Run yourself, step by step (NOT in background):
#     bash size_reduction/compat_env/setup_and_quantize.sh
#
set -e
cd "/home/ganesh/Desktop/Prince number detection in meter"
ENV=size_reduction/compat_env/venv

echo "### 1. create venv ###"
python3 -m venv "$ENV"
source "$ENV/bin/activate"
python -m pip install -q --upgrade pip

echo "### 2. install older Paddle (v1 PIR exporter) + tools ###"
# CPU build of Paddle 3.0.0 (GPU 3.0.0 needs the paddle index; CPU is enough for export+convert)
pip install -q "paddlepaddle==3.0.0"
pip install -q "paddleocr==3.0.0" paddle2onnx onnx onnxruntime
python -c "import paddle,paddleocr; print('paddle',paddle.__version__,'paddleocr',paddleocr.__version__)"

echo "### 3. retrain server on our dataset (CPU — SLOW, or set up GPU paddle 3.0.0) ###"
REPO=$(python -c "import paddlex,os;print(os.path.join(os.path.dirname(paddlex.__file__),'repo_manager','repos','PaddleOCR'))")
# reuse the existing config + dataset + pretrained; output to this env
python - <<'PY'
import yaml,os
c=yaml.safe_load(open("finetune/rec_finetune.yml"))
c["Global"]["save_model_dir"]="size_reduction/compat_env/output"
c["Global"]["use_gpu"]=False           # CPU venv; flip to True if you install paddle-gpu 3.0.0
c["Global"]["epoch_num"]=20
c["Global"]["save_epoch_step"]=1000
yaml.safe_dump(c,open("size_reduction/compat_env/server_v1.yml","w"),sort_keys=False)
print("wrote size_reduction/compat_env/server_v1.yml")
PY
PYTHONPATH=$REPO python $REPO/tools/train.py -c size_reduction/compat_env/server_v1.yml

echo "### 4. export -> inference (PIR v1, convertible) ###"
PYTHONPATH=$REPO python $REPO/tools/export_model.py \
  -c size_reduction/compat_env/server_v1.yml \
  -o Global.pretrained_model=size_reduction/compat_env/output/best_accuracy \
     Global.save_inference_dir=size_reduction/compat_env/inference_server_v1

echo "### 5. paddle2onnx -> ONNX FP32 ###"
paddle2onnx --model_dir size_reduction/compat_env/inference_server_v1 \
  --model_filename inference.json --params_filename inference.pdiparams \
  --save_file size_reduction/models/server_fp32.onnx --opset_version 13

echo "### 6. ONNX FP32 -> INT8 (~20 MB) ###"
python - <<'PY'
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
quantize_dynamic("size_reduction/models/server_fp32.onnx",
                 "size_reduction/models/server_int8.onnx", weight_type=QuantType.QInt8)
for n in ("server_fp32.onnx","server_int8.onnx"):
    print(f"{n}: {os.path.getsize('size_reduction/models/'+n)/1e6:.1f} MB")
PY
deactivate
echo "### DONE — quantized server model: size_reduction/models/server_int8.onnx ###"
