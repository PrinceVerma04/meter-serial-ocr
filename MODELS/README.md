# Models index — meter-serial OCR

Clearly-named links to every model. The links point to the real files (which stay
where the scripts expect them), so nothing breaks — this folder is just an
easy-to-read map.

## Naming scheme
`meterocr_<version>_<precision>_<method>_<size>[.runtime]`
- **version** — `v1` = first fine-tune (1,144 imgs) · `v2` = retrained (6,709 imgs)
- **precision** — `fp32` (32-bit) · `int8` (8-bit, quantized)
- **method** — `original` (no quant) · `ptq` (post-training quant) · `qat` (quant-aware training) · `old`
- **size** — on-disk size · **runtime** — `.onnx` (ONNX Runtime) or `paddle`

## The models

| Name (in MODELS/) | Real path | Shape (in→out) | Size | Precision | Accuracy* | Runtime |
|---|---|---|---|---|---|---|
| `meterocr_v2_fp32_original_81mb` | `finetune/inference_v2/` | [N,3,48,W] → [N,T,18385] | 81 MB | FP32 | **94.7%** | Paddle |
| `meterocr_v2_int8_qat_22mb_paddle` ⭐ | `size_reduction/prune_qat/qat_inference/` | [N,3,48,320] → [N,T,18385] | 22 MB | INT8 (QAT) | **92.8%** | Paddle |
| `meterocr_v2_int8_ptq_21mb.onnx` | `size_reduction/models/server_int8.onnx` | [N,3,48,W] → [N,T,18385] | 21 MB | INT8 (PTQ) | 85.3% | ONNX |
| `meterocr_v1_fp32_old_81mb` | `finetune/inference/` | [N,3,48,W] → [N,T,18385] | 81 MB | FP32 | ~87% (val) | Paddle |

*Exact-match on 414 readable images of a 500-img TCPODL/TPSODL sample (human-judged
2026-07-15; 86 all-model-fail crops excluded). In-domain TPWODL data runs ~95%+.
Shape legend: N=batch, 3=RGB, 48=height, W=width(~320), T=time-steps, 18385=char classes.

**Deployment:** the PTQ INT8 model is packaged, ready-to-ship and Paddle-free, in
`deploy_meterocr_ptq_int8/`.

## Which to use
- **Best accuracy:** `meterocr_v2_fp32_original_81mb` (81 MB)
- **Best small model (Paddle):** `meterocr_v2_int8_qat_22mb_paddle` (22 MB, ~0.91) ⭐
- **Small + ONNX runtime:** `meterocr_v2_int8_ptq_21mb.onnx` (21 MB, ~0.80)

## Quantization methodology
- **PTQ** (→ ptq model): Paddle→ONNX, fold constants, then onnxruntime **static INT8**
  calibrated on ~150 real crops (per-tensor). Fast, no training, −3–4 pts accuracy.
- **QAT** (→ qat model): PaddleSlim inserts fake-quant on Conv2D+Linear, **fine-tune
  10 epochs** on the 6,709-label dataset, export INT8. Recovers nearly all accuracy.
- Both required an isolated **Paddle 3.0.0** env (the installed 3.3.1 exports a format
  the ONNX converter can't read). Full pipeline in `size_reduction/`.
