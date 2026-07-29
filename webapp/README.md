# Meter-OCR inference-time benchmark (web UI)

Upload a **single image** or a **whole folder**; the page shows, for each of the
3 models, the **extracted serial number** and the **per-image inference time (ms)**,
plus average ms per model and a CSV export.

There are **separate apps for CPU and GPU** — the only difference is the device
each pins the models to.

## Run

```bash
# CPU — all 3 models on CPU
python3 webapp/app_cpu.py          # http://127.0.0.1:5001

# GPU — models on GPU (see note below)
python3 webapp/app_gpu.py          # http://127.0.0.1:5002
```

Options: `--port N`, `--host 0.0.0.0` (reach from another machine). Each app
loads the models once at startup (takes a few seconds), then serves the page.
Open the printed URL in a browser.

## The 3 models

| Model | File | Size | Runtime |
|---|---|---|---|
| **FP32 original** | `finetune/inference_v2/` | 81 MB | Paddle |
| **INT8 QAT** | `deploy_meterocr_qat_int8/model/` | 22 MB | Paddle |
| **INT8 PTQ** | `deploy_meterocr_ptq_int8/model/` | 21 MB | ONNX |

## CPU vs GPU — what actually runs where

Measured on this box (GTX 1080 Ti), single image, steady state:

| Model | CPU | GPU |
|---|---|---|
| FP32 original | ~300–460 ms | **~11 ms** ✅ big win |
| INT8 QAT | ~630–840 ms | ❌ **unavailable** (Paddle INT8 needs TensorRT) |
| INT8 PTQ (ONNX) | ~170–215 ms | ⚠️ **falls back to CPU** (INT8 CUDA unsupported) |

**INT8 quantization is a CPU-deployment optimization.** Only the FP32 model
meaningfully benefits from the GPU. The GPU app reports this honestly: each model
shows the device it *actually* ran on (green = on requested device, amber =
fell back, red dot = unavailable).

## Files
- `engine.py`  — shared: preprocessing, CTC decode, the 3 model loaders (CPU/GPU), timing.
- `server.py`  — Flask page + `/infer` route (shared by both apps).
- `app_cpu.py` — CPU launcher (`DEVICE = "cpu"`).
- `app_gpu.py` — GPU launcher (`DEVICE = "gpu"`).

Preprocessing is identical for all 3 models (resize→3×48×320, normalize), so the
timings are a fair comparison. Timing measures the model forward pass per image.
