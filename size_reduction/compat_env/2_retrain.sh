#!/usr/bin/env bash
# STEP 2 — retrain the SERVER model inside the Paddle-3.0.0 venv, so its weights
# can later be exported to the convertible v1 format.
#
# CPU is SLOW (hours). For GPU (~7 min) install the GPU build into the venv first:
#   size_reduction/compat_env/venv/bin/pip uninstall -y paddlepaddle
#   size_reduction/compat_env/venv/bin/pip install paddlepaddle-gpu==3.0.0 \
#       -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
#   then set use_gpu=True below.
#
# Run:  bash size_reduction/compat_env/2_retrain.sh
set -e
cd "/home/ganesh/Desktop/Prince number detection in meter"
ENV=size_reduction/compat_env/venv
REPO=~/.local/lib/python3.10/site-packages/paddlex/repo_manager/repos/PaddleOCR
DIR=size_reduction/compat_env

# build a training config (reuses finetune/dataset = your 6,709 labels + the pretrained)
"$ENV/bin/python" - <<'PY'
import yaml, os
c = yaml.safe_load(open("finetune/rec_finetune.yml"))
root = os.getcwd()
c["Global"]["save_model_dir"]  = f"{root}/size_reduction/compat_env/output"
c["Global"]["pretrained_model"] = f"{root}/size_reduction/legacy_server/pretrained/PP-OCRv5_server_rec_pretrained"
c["Global"]["save_epoch_step"] = 1000
c["Global"]["use_gpu"] = True            # GPU Paddle 3.0.0 installed in venv
yaml.safe_dump(c, open("size_reduction/compat_env/server_train.yml","w"), sort_keys=False)
print("wrote size_reduction/compat_env/server_train.yml")
PY

echo "training server (this reproduces the ~91% val model) ..."
PYTHONPATH=$REPO "$ENV/bin/python" $REPO/tools/train.py -c "$DIR/server_train.yml"
echo "DONE. checkpoint: $DIR/output/best_accuracy.pdparams"
