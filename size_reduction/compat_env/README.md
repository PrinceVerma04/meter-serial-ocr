# Quantize the SERVER model — full runbook (Paddle 3.0.0 compat env)

**Why this exists:** Paddle 3.3.1 exports PIR **v4**, which `paddle2onnx 2.1.0` can't
convert. Paddle **3.0.0** exports PIR **v1**, which converts. So we build a 3.0.0
venv, retrain there, export v1, convert to ONNX, and **static**-quantize to INT8.
(Dynamic quant does NOT shrink this conv-heavy model — static is required.)

Your 81 MB model is untouched and backed up at `finetune/inference_v2_backup_2026-07-13/`.

## Run these in order (all foreground, in your terminal)

```bash
cd "/home/ganesh/Desktop/Prince number detection in meter"

# 1. build the Paddle-3.0.0 venv (~5 min; idempotent)
bash size_reduction/compat_env/1_setup_venv.sh

# 2. retrain the server model in the venv
#    CPU = hours. For GPU (~7 min): see the header of 2_retrain.sh to install
#    paddlepaddle-gpu==3.0.0 and set use_gpu=True, THEN run:
bash size_reduction/compat_env/2_retrain.sh

# 3. export -> PIR v1 -> ONNX FP32
bash size_reduction/compat_env/3_export_v1.sh

# 4. STATIC INT8 quantization (calibrates on real crops)
size_reduction/compat_env/venv/bin/python size_reduction/compat_env/4_quantize.py
```

Result: `size_reduction/models/server_int8.onnx`.

## 5. Verify accuracy of the quantized model (IMPORTANT)
INT8 can lose accuracy — always check before trusting it. `size_reduction/onnx_reader.py`
runs an ONNX rec model with the correct pre/post-processing:

```bash
size_reduction/compat_env/venv/bin/pip install onnxruntime opencv-python numpy
size_reduction/compat_env/venv/bin/python - <<'PY'
import glob, csv, sys
sys.path.insert(0, "size_reduction")
from onnx_reader import OnnxRec
import cv2
m = OnnxRec("size_reduction/models/server_int8.onnx")
for f in sorted(glob.glob("new dataset for test/tpcodl_tpsodl_roi_images/*.jpg"))[:20]:
    t, s = m.read(cv2.imread(f))
    print(f"{f.split('/')[-1][:22]:24} -> {t:18} {s:.3f}")
PY
```
Then compare against the 81 MB model's readings (from `model_compare_results.csv`)
to measure the real accuracy delta.

## What size to expect
- `server_fp32.onnx`  ~85–215 MB (depends on whether the NRTR head is included)
- `server_int8.onnx`  **~4× smaller than the FP32 ONNX** after static quant

If it lands larger than the 81 MB Paddle model or loses too much accuracy, the
Paddle FP32 model (`finetune/inference_v2/`) stays the better deployable option.

## Files
```
compat_env/
  1_setup_venv.sh       venv + Paddle 3.0.0 + deps
  2_retrain.sh          retrain server on your 6,709 labels
  3_export_v1.sh        export PIR v1 -> ONNX FP32
  4_quantize.py         static INT8 (calibrated) -> ONNX INT8
  server_train.yml      (generated) training config
  venv/                 the isolated environment
```
