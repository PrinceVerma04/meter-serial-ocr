# Quantization — FP32 (84 MB) → PTQ INT8 (21 MB)

Post-training static quantization (PTQ) with ONNX Runtime. The INT8 model is what the
server runs (`deploy/ptq_int8/`): 4x smaller and ~14x faster on CPU (18 ms vs 253 ms per crop).

| File | What |
|---|---|
| `quantize_v4_ptq.py` | **one command** for v4: export → ONNX FP32 → INT8 → score v3/v4 FP32/PTQ the deploy way. Run 2026-11-02. |
| `quantize_v3_ptq.py` | record of how the deployed v3 PTQ was made (2026-07-25); refuses to overwrite it |
| `compat_env/` | the Paddle 3.0.0 env needed for the export step (see its README) |
| `requirements.txt` | paddle2onnx / onnx / onnxruntime |

Outputs live in `models/<version>/` (`meterocr_vX_fp32.onnx`, `meterocr_vX_int8_ptq.onnx`).

## How (same recipe for v3 and v4)
1. **Export** the training weights (`models/vX/meterocr_vX_train_weights.pdparams`) with Paddle 3.0.0 → PIR v1.
2. **paddle2onnx** → ONNX FP32. Its own constant folding fails because the project path has a
   space, so constants are folded in Python with `quant_pre_process`; this also drops the
   unused NRTR training head (205 MB → 84 MB).
3. **quantize_static**: QDQ, **per-tensor** (`per_channel=False`; per-channel truncates long
   serials), Conv/MatMul/Gemm, calibrated on **200 random training crops** (seed 42).

## Measured (deploy pipeline, 0.95 gate) — `results/2026-11-02_v4/`
| | FP32 → PTQ agreement | test_old right, FP32 → PTQ |
|---|---|---|
| v3 | 93.3% | 932 → 893 (−39) |
| v4 | 98.6% | 930 → 922 (−8) |

```bash
python3 quantization/quantize_v4_ptq.py      # re-prints the summary; delete an output to redo a step
```
