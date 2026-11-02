# Models — by version

All models are the **PP-OCRv5 server recognizer** (SVTR_HGNet backbone, CTC head) fine-tuned
on meter serial crops. Input `[N, 3, 48, W≤320]` (BGR crop, height 48, padded), output
`[N, T, 18385]` (CTC over the PP-OCRv5 character dictionary). Reads with confidence ≥ 0.95 are
auto-accepted, the rest go to human review.

```
models/
├── base/  PP-OCRv5_server_rec_pretrained.pdparams     stock PaddleOCR weights (start of v1–v3)
├── v3/                                                 deployed in production
│   ├── meterocr_v3_fp32_paddle/        FP32, Paddle inference (inference.json/.pdiparams/.yml)
│   ├── meterocr_v3_fp32.onnx           FP32, ONNX (CTC only) — quantization reference
│   ├── meterocr_v3_int8_ptq.onnx       INT8 PTQ, ONNX — the deployed model (= deploy/ptq_int8/model/)
│   └── meterocr_v3_train_weights.pdparams   v3 as training weights (rebuilt 2026-11-02, start of v4)
├── v4/                                                 trained 2026-11-02, not deployed yet
│   ├── meterocr_v4_fp32_paddle/
│   ├── meterocr_v4_fp32.onnx
│   ├── meterocr_v4_int8_ptq.onnx       INT8 PTQ, ONNX — deploy candidate
│   └── meterocr_v4_train_weights.pdparams   best epoch (16) — start point for a v5 / re-quantization
└── v4_aug/                                             August experiment (19 Aug), never tested — NOT the v4 above
    ├── meterocr_v4aug_fp32_paddle/
    ├── meterocr_v4aug_fp32.onnx
    ├── meterocr_v4aug_int8_ptq.onnx
    └── meterocr_v4aug_train_weights.pdparams   best epoch (10)
```

| Version | Date | Started from | Training data | FP32 | PTQ INT8 | Status |
|---|---|---|---|---|---|---|
| v1 | Jul 2026 | stock | 1,144 TPWODL | deleted | — | replaced |
| v2 | Jul 2026 | stock | 6,709 | deleted | deleted (QAT too) | replaced |
| **v3** | 2026-07-24 / PTQ 07-25 | stock | 6,733 `datasets/v3_train` | 81 MB | 21 MB | **deployed** |
| **v4** | 2026-11-02 | v3 | 23,707 `datasets/v4_train` | 81 MB | 21 MB | candidate |
| v4-aug | 2026-08-19 | stock | 12,886 `datasets/v4_aug_train` | 81 MB | 21 MB | experiment, never scored (val 94.08%) |

## Accuracy (exact match, deploy pipeline, 0.95 gate) — `results/2026-11-02_v4/`
| Test set | v3 FP32 | v3 PTQ (deployed) | v4 FP32 | v4 PTQ |
|---|---|---|---|---|
| test_old — 1,024 old test, 74% letter serials | 932 (91.02%) | 893 (87.21%) | 930 (90.82%) | 922 (90.04%) |
| test_new — 1,680 new held-out | 1,569 (93.39%) | 1,529 (91.01%) | 1,663 (98.99%) | 1,660 (98.81%) |
| test_new_hard — 171 human-reviewed (v3 had misread them) | 60 (35.09%) | 20 (11.70%) | 157 (91.81%) | 155 (90.64%) |
| precision of auto-accepted reads, test_old | 93.92% | 93.48% | 93.06% | 92.38% |
| CPU time per crop (1 thread) | 253 ms | 18 ms | 253 ms | 18 ms |

Read with care: the 1,509 easy `test_new` images were chosen because v3 read them right
(v3 1,509/1,509, v4 PTQ 1,505) and the 171 hard ones because v3 got them wrong; `test_old`
is the neutral comparison (+29 right for v4 PTQ, but 74 vs 58 confidently-wrong reads).

## Which file to use
- **Production / CPU / no Paddle:** `vX/meterocr_vX_int8_ptq.onnx` with `deploy/ptq_int8/meterocr.py`
  (`MeterOCR(model_path=...)`); `scripts/run_ptq_folder.py --model v3|v4` for a folder.
- **Best accuracy / Paddle:** `vX/meterocr_vX_fp32_paddle/`.
- **Train further:** start from `vX/meterocr_vX_train_weights.pdparams` (see `training/train_v4.py`).
- **Re-quantize:** `quantization/quantize_v4_ptq.py`.
