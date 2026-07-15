#!/usr/bin/env bash
# STEP 1 — PRUNE the retrained server model (FPGM filter pruning + retrain).
# Uses PaddleOCR's slim tool. Outputs a slimmed model to prune_qat/prune_output.
# Originals untouched. Run in the compat venv. ~20-30 min GPU.
set -e
cd "/home/ganesh/Desktop/Prince number detection in meter"
ENV=size_reduction/compat_env/venv
REPO=~/.local/lib/python3.10/site-packages/paddlex/repo_manager/repos/PaddleOCR
DIR=size_reduction/prune_qat

# sensitivity analysis + pruned retraining, starting from your 6,709-img checkpoint
PYTHONPATH=$REPO "$ENV/bin/python" $REPO/deploy/slim/prune/sensitivity_anal.py \
  -c size_reduction/compat_env/server_train.yml \
  -o Global.pretrained_model=size_reduction/compat_env/output/best_accuracy \
     Global.save_model_dir="$DIR/prune_output" \
     Global.epoch_num=15 \
     Global.save_epoch_step=1000

# export the pruned model to an inference model (PIR v1, convertible)
PYTHONPATH=$REPO "$ENV/bin/python" $REPO/deploy/slim/prune/export_prune_model.py \
  -c size_reduction/compat_env/server_train.yml \
  -o Global.pretrained_model="$DIR/prune_output/best_accuracy" \
     Global.save_inference_dir="$DIR/prune_inference" 2>/dev/null || \
  echo "NOTE: if export_prune_model args differ, use tools/export_model.py on $DIR/prune_output/best_accuracy"
echo "pruned model checkpoint: $DIR/prune_output/  (feed this into 2_qat.sh)"
