#!/usr/bin/env bash
# ============================================================================
# QAT (quantization-aware training) for v4 — QAT-ONLY, no pruning, matching
# what actually shipped as the v2/v3 22MB QAT model (per the deleted
# deploy/qat_int8/README.md: "~92.8% vs PTQ's ~85%"). Pruning is a separate,
# more experimental path (size_reduction/prune_qat/1_prune.sh) — skipped here.
#
# WHY THIS IS SLOWER THAN PTQ WAS: QAT fine-tunes the model with fake-quant
# ops inserted, so it's real training, not just an export. The only Paddle
# build proven to export a paddle2onnx-convertible format on this box is
# 3.0.0 (see quantize_v4_ptq.py's header) — and 3.0.0's GPU wheel has no
# Blackwell/sm_120 kernels for this RTX 5080 (same error retrain_all.sh hit
# originally). So this whole step runs on CPU, in the same compat_env venv
# used for the PTQ re-export. Expect HOURS, not the ~15 min GPU quoted in
# size_reduction/prune_qat/README.md (which predates this GPU).
#
# If you have GPU time pressure: it may be worth trying this same script
# with ENV pointed at finetune/train_env (GPU, Paddle 3.3.1) instead — faster,
# but then STEP 2 (paddle2onnx conversion) may hit the same PIR-format
# incompatibility PTQ needed to route around, with no guaranteed re-export
# fix for a QAT (fake-quant) checkpoint. Untested; CPU/compat_env is the
# known-safe path.
#
# Run:  bash scripts/qat_v4.sh
# ============================================================================
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
ENV="$ROOT/size_reduction/compat_env/venv"          # CPU, Paddle 3.0.0 (built by quantize_v4_ptq.sh)
REPO="$ROOT/finetune/PaddleOCR_repo"
CFG="$ROOT/finetune/rec_retrain_v4.yml"
PRETRAINED="$ROOT/finetune/output_retrain_v4/best_accuracy"   # v4's FP32 checkpoint, UN-pruned
DIR="$ROOT/size_reduction/prune_qat"
QAT_OUT="$DIR/qat_output_v4"
QAT_INFER="$DIR/qat_inference_v4"

if [ ! -d "$ENV" ]; then
  echo "!! $ENV not found — run scripts/quantize_v4_ptq.sh first (it builds this venv)"
  exit 1
fi
if [ ! -f "$PRETRAINED.pdparams" ]; then
  echo "!! $PRETRAINED.pdparams not found — run scripts/retrain_all.sh first"
  exit 1
fi

echo "=== 1/2: installing paddleslim into the compat venv ==="
"$ENV/bin/pip" install -q --no-cache-dir paddleslim==2.6.0

echo "=== 2/2: QAT fine-tune (CPU, ~hours) -> $QAT_INFER ==="
PYTHONPATH="$REPO" "$ENV/bin/python" "$REPO/deploy/slim/quantization/quant.py" \
  -c "$CFG" \
  -o Global.pretrained_model="$PRETRAINED" \
     Global.checkpoints=null \
     Global.save_model_dir="$QAT_OUT" \
     Global.save_inference_dir="$QAT_INFER" \
     Global.epoch_num=10 \
     Global.save_epoch_step=1000 \
     Global.use_gpu=False \
  2>&1 | tee "$ROOT/finetune/qat_v4.log"

echo ""
echo "QAT quantized inference model: $QAT_INFER"
du -sh "$QAT_INFER" 2>/dev/null

echo ""
echo "=== converting to ONNX ==="
"$ENV/bin/paddle2onnx" --model_dir "$QAT_INFER" \
  --model_filename inference.json --params_filename inference.pdiparams \
  --save_file "$DIR/meterocr_v4_qat_int8.onnx" --opset_version 13 || {
    echo "paddle2onnx failed on the QAT export — this is the known-risky step"
    echo "(PaddleOCR's slim tools + PIR format compatibility is uncertain, per"
    echo "size_reduction/prune_qat/README.md). The QAT Paddle checkpoint at"
    echo "$QAT_INFER is still usable directly (Paddle Inference runtime,"
    echo "no ONNX needed) even if this conversion fails."
    exit 1
  }

echo ""
echo "=== DONE ==="
echo "  v4 QAT (Paddle): $QAT_INFER"
echo "  v4 QAT (ONNX)  : $DIR/meterocr_v4_qat_int8.onnx"
echo ""
echo "Next: verify accuracy before trusting it (INT8/QAT can still regress on"
echo "specific chars) — score against v3's QAT (was ~92.8%) and v4's own PTQ"
echo "(21.4 MB) using the same held-out set, then log the result in TRAINING_LOG.md."
