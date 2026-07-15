#!/usr/bin/env bash
# STEP 2 — QAT (quantization-aware training) of the PRUNED model.
# Fine-tunes with fake-quant so INT8 accuracy is recovered, and exports a
# quantized inference model. Run in the compat venv. ~15 min GPU.
#
# To QAT the ORIGINAL (un-pruned) model instead — for a more accurate 21 MB
# model rather than a smaller one — set PRETRAINED to the compat_env checkpoint.
set -e
cd "/home/ganesh/Desktop/Prince number detection in meter"
ENV=size_reduction/compat_env/venv
REPO=~/.local/lib/python3.10/site-packages/paddlex/repo_manager/repos/PaddleOCR
DIR=size_reduction/prune_qat

# start QAT from the PRUNED checkpoint (step 1). For QAT-only (21 MB, more accurate),
# point at: size_reduction/compat_env/output/best_accuracy
PRETRAINED="$DIR/prune_output/best_accuracy"

PYTHONPATH=$REPO "$ENV/bin/python" $REPO/deploy/slim/quantization/quant.py \
  -c size_reduction/compat_env/server_train.yml \
  -o Global.pretrained_model="$PRETRAINED" \
     Global.save_model_dir="$DIR/qat_output" \
     Global.save_inference_dir="$DIR/qat_inference" \
     Global.epoch_num=10 \
     Global.save_epoch_step=1000

echo "QAT quantized inference model: $DIR/qat_inference/"
du -sh "$DIR/qat_inference" 2>/dev/null
