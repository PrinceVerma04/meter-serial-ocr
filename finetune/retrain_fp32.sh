#!/usr/bin/env bash
# ============================================================================
# Retrain the FP32 PP-OCRv5 server recognizer on the new dataset.
#
#   dataset : finetune/dataset_retrain/   (built by scripts/prepare_retrain_data.py)
#   config  : finetune/rec_retrain.yml    (LR 2e-5, 20 epochs — the proven recipe)
#   output  : finetune/output_retrain/    (checkpoints) -> finetune/inference_v3/ (exported)
#
# This trains the FP32 model. QAT and PTQ are DERIVED from it afterwards
# (see the two blocks at the bottom of this file / the chat notes).
#
# Prereqs this script sets up if missing:
#   - the PaddleOCR training repo (tools/train.py is NOT in the pip package)
#   - the PP-OCRv5_server_rec pretrained weights
# Needs: the system python env (paddle 3.3.1 + paddleocr 3.6.0) + a GPU.
# ============================================================================
set -e
cd "$(dirname "$0")/.."                       # -> project root
ROOT="$(pwd)"

# The PaddleOCR training repo that the ORIGINAL training used is already on this
# box under paddlex's repo_manager — no clone / pip install needed (that path was
# failing to build anyway). tools/train.py + the ppocrv5 dict both live here.
REPO="$HOME/.local/lib/python3.10/site-packages/paddlex/repo_manager/repos/PaddleOCR"
CFG="$ROOT/finetune/rec_retrain.yml"
PRETRAIN_DIR="$ROOT/finetune/pretrained"
PRETRAIN="$PRETRAIN_DIR/PP-OCRv5_server_rec_pretrained"

# --- 1. sanity-check the training repo ---
if [ ! -f "$REPO/tools/train.py" ]; then
  echo "!! PaddleOCR training repo not found at $REPO"; exit 1
fi

# --- 2. get pretrained weights (deleted in the 2026-07-16 cleanup) ---
if [ ! -f "$PRETRAIN.pdparams" ]; then
  echo ">> downloading PP-OCRv5_server_rec pretrained weights"
  mkdir -p "$PRETRAIN_DIR"
  wget -O "$PRETRAIN.pdparams" \
    https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams \
    || { echo "!! download failed — check network / URL"; exit 1; }
fi

# --- 3. train (best epoch is auto-saved as best_accuracy.*) ---
echo ">> training on finetune/dataset_retrain (LR 2e-5, 20 epochs)"
PYTHONPATH="$REPO" python3 "$REPO/tools/train.py" -c "$CFG" \
  2>&1 | tee "$ROOT/finetune/retrain.log"

# --- 4. export the best checkpoint to a static inference model ---
echo ">> exporting best checkpoint -> finetune/inference_v3/"
PYTHONPATH="$REPO" python3 "$REPO/tools/export_model.py" -c "$CFG" \
  -o Global.checkpoints="$ROOT/finetune/output_retrain/best_accuracy" \
     Global.save_inference_dir="$ROOT/finetune/inference_v3"

echo ">> DONE. New FP32 model in finetune/inference_v3/"
echo "   Next: score it, then derive QAT + PTQ (see notes)."
