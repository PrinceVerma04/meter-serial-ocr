# Models index — meter-serial OCR

Clearly-named links to every model. The links point to the real files (which stay
where the scripts expect them), so nothing breaks — this folder is just an
easy-to-read map.

## Naming scheme
`meterocr_<version>_<precision>_<method>_<size>[.runtime]`
- **version** — `v1` = first fine-tune (1,144 imgs) · `v2` = retrained (6,709 imgs) · `v3` = latest retrain (6,733 imgs, 2026-07-24)
- **precision** — `fp32` (32-bit) · `int8` (8-bit, quantized)
- **method** — `original` (no quant) · `ptq` (post-training quant) · `old`
- **size** — on-disk size · **runtime** — `.onnx` (ONNX Runtime) or `paddle`

## The models

| Name (in MODELS/) | Real path | Shape (in→out) | Size | Precision | Accuracy* | Runtime |
|---|---|---|---|---|---|---|
| `meterocr_v3_fp32_original_81mb` | `finetune/inference_v3/` | [N,3,48,W] → [N,T,18385] | 81 MB | FP32 | latest | Paddle |
| `meterocr_v3_int8_ptq_22mb.onnx` | `size_reduction/models/meterocr_v3_int8_ptq.onnx` | [N,3,48,W] → [N,T,18385] | 22 MB | INT8 (PTQ) | latest | ONNX |
| `meterocr_v2_fp32_original_81mb` | `finetune/inference_v2/` | [N,3,48,W] → [N,T,18385] | 81 MB | FP32 | **94.7%** | Paddle |
| `meterocr_v2_int8_ptq_21mb.onnx` | `size_reduction/models/server_int8.onnx` | [N,3,48,W] → [N,T,18385] | 21 MB | INT8 (PTQ) | 85.3% | ONNX |
| `meterocr_v1_fp32_old_81mb` | `finetune/inference/` | [N,3,48,W] → [N,T,18385] | 81 MB | FP32 | ~87% (val) | Paddle |

**v3 (latest, 2026-07-25):** re-quantized correctly from `finetune/inference_v3/`.
The v3 INT8 is the true 22 MB PTQ model now shipped in `deploy/ptq_int8/`.
Reproduce with `size_reduction/quantize_v3_ptq.py` (see its header for the two
fixes: export via Paddle 3.0.0, and fold constants in-Python because the space in
the project path breaks paddle2onnx's folding).

*Exact-match on 414 readable images of a 500-img TCPODL/TPSODL sample (human-judged
2026-07-15; 86 all-model-fail crops excluded). In-domain TPWODL data runs ~95%+.
Shape legend: N=batch, 3=RGB, 48=height, W=width(~320), T=time-steps, 18385=char classes.

**Deployment:** the PTQ INT8 model is packaged, ready-to-ship and Paddle-free, in
`deploy/ptq_int8/`.

## Which to use
- **Best accuracy:** `meterocr_v3_fp32_original_81mb` (81 MB, latest retrain)
- **Small + ONNX runtime (deploy):** `meterocr_v3_int8_ptq_22mb.onnx` (22 MB) — shipped in `deploy/ptq_int8/`

## Quantization methodology
- **PTQ** (→ ptq model): Paddle→ONNX, fold constants, then onnxruntime **static INT8**
  calibrated on ~150 real crops (per-tensor). Fast, no training, −3–4 pts accuracy.
  Required an isolated **Paddle 3.0.0** env (the installed 3.3.1 exports a format
  the ONNX converter can't read). Full pipeline in `size_reduction/`.
