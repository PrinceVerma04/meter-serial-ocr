#!/usr/bin/env bash
# ============================================================================
# Restart training from scratch — rebuilds the Paddle training environment
# (deleted in earlier disk cleanups), builds the dataset from the merged
# "retraining dataset with label/" (12,886 images as of 2026-08-19 — see
# TRAINING_LOG.md), trains the FP32 recognizer, and exports it.
#
# This produces v4 (FP32). PTQ / QAT quantization are separate steps run
# afterwards (printed at the end) — they derive from this FP32 export.
#
# Run from the project root:
#   bash scripts/retrain_all.sh
#
# Safe to re-run: every step is idempotent (skips work already done).
# ============================================================================
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
ENV="$ROOT/finetune/train_env"
REPO="$ROOT/finetune/PaddleOCR_repo"
PRETRAIN_DIR="$ROOT/finetune/pretrained"
PRETRAIN="$PRETRAIN_DIR/PP-OCRv5_server_rec_pretrained"
DATASET="$ROOT/finetune/dataset_retrain"
CFG="$ROOT/finetune/rec_retrain_v4.yml"
OUT_CKPT="$ROOT/finetune/output_retrain_v4"
OUT_INFER="$ROOT/finetune/inference_v4"

echo "=== 1/6: training venv (python3.10 + Paddle GPU) ==="
if [ ! -d "$ENV" ]; then
  python3.10 -m venv "$ENV"
fi
"$ENV/bin/python" -m pip install -q --upgrade pip
# Needs a build with sm_120 (Blackwell / RTX 5080) kernels — the old pinned
# 3.0.0 cu118 wheel predates this GPU and fails with "no kernel image is
# available for execution on the device". Latest stable from PyPI carries
# current-generation CUDA kernels; this also matches what v3 was actually
# trained with (Paddle 3.3.1, per finetune/retrain_fp32.sh's header) rather
# than the 3.0.0 compat build (that one is only for the later PTQ export step).
if ! "$ENV/bin/python" -c "import paddle" 2>/dev/null; then
  echo ">> installing paddlepaddle-gpu from the official cu130 index (plain PyPI's"
  echo "   paddlepaddle-gpu is a stale 2.6.2 build with no cudnn/Blackwell support)"
  "$ENV/bin/pip" install -q paddlepaddle-gpu \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ \
    || "$ENV/bin/pip" install -q paddlepaddle
fi
"$ENV/bin/pip" install -q "numpy<2" pyyaml "opencv-python<4.10" tqdm shapely \
  scikit-image pyclipper lmdb imgaug rapidfuzz albumentations Pillow scipy python-Levenshtein
"$ENV/bin/python" -c "import paddle; print('paddle', paddle.__version__, '| GPU:', paddle.device.is_compiled_with_cuda())"

echo "=== 2/6: PaddleOCR training repo (tools/train.py, tools/export_model.py) ==="
if [ ! -f "$REPO/tools/train.py" ]; then
  git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git "$REPO"
fi
echo ">> installing the repo's own requirements (covers whatever our hand-picked list misses)"
"$ENV/bin/pip" install -q -r "$REPO/requirements.txt"
"$ENV/bin/pip" install -q requests

echo "=== 3/6: pretrained PP-OCRv5_server_rec weights ==="
if [ ! -f "$PRETRAIN.pdparams" ]; then
  mkdir -p "$PRETRAIN_DIR"
  wget -q -O "$PRETRAIN.pdparams" \
    https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams
fi

echo "=== 4/6: build finetune/dataset_retrain from 'retraining dataset with label/' ==="
python3 "$ROOT/scripts/prepare_retrain_data.py"   # rebuilds train/val/test.txt + dict.txt

echo "=== 5/6: write finetune/rec_retrain_v4.yml (paths resolved for this machine) ==="
"$ENV/bin/python" - <<PY
import yaml
c = yaml.safe_load(open("$ROOT/finetune/rec_retrain.yml"))
c["Global"]["save_model_dir"]      = "$OUT_CKPT"
c["Global"]["pretrained_model"]    = "$PRETRAIN"
c["Global"]["character_dict_path"] = "$REPO/ppocr/utils/dict/ppocrv5_dict.txt"
c["Global"]["use_gpu"]             = True
c["Train"]["dataset"]["data_dir"]        = "$DATASET"
c["Train"]["dataset"]["label_file_list"] = ["$DATASET/train.txt"]
c["Eval"]["dataset"]["data_dir"]         = "$DATASET"
c["Eval"]["dataset"]["label_file_list"]  = ["$DATASET/val.txt"]
yaml.safe_dump(c, open("$CFG", "w"), sort_keys=False)
print("wrote $CFG")
PY

echo "=== 6/6: train (LR 2e-5, 20 epochs) — this is the long step ==="
PYTHONPATH="$REPO" "$ENV/bin/python" "$REPO/tools/train.py" -c "$CFG" \
  2>&1 | tee "$ROOT/finetune/retrain_v4.log"

echo ">> exporting best checkpoint -> $OUT_INFER"
PYTHONPATH="$REPO" "$ENV/bin/python" "$REPO/tools/export_model.py" -c "$CFG" \
  -o Global.checkpoints="$OUT_CKPT/best_accuracy" \
     Global.save_inference_dir="$OUT_INFER"

echo ""
echo "DONE. v4 FP32 model exported to $OUT_INFER"
echo ""
echo "Next steps:"
echo "  1. Log this run in TRAINING_LOG.md (date, image count from step 4's"
echo "     console output, val/test accuracy from $ROOT/finetune/retrain_v4.log)"
echo "  2. Score v4 vs v3 before replacing production (reuse"
echo "     scripts/run_3models_tpddl_julyset.py style comparison)"
echo "  3. Derive PTQ INT8:  size_reduction/quantize_v3_ptq.py (point it at"
echo "     $OUT_INFER instead of inference_v3)"
echo "  4. Derive QAT INT8:  size_reduction/prune_qat/{1_prune,2_qat,3_export_convert}.sh"
