# Meter Serial OCR — PTQ INT8 (deployment package)

Self-contained OCR that reads the alphanumeric serial number from a cropped
electricity-meter nameplate image. Runs on **ONNX Runtime only — no PaddlePaddle**.

## Contents
```
deploy/ptq_int8/
├── model/
│   ├── meterocr_ptq_int8.onnx   # 22 MB INT8 recognizer (v3, latest retrain)
│   └── ppocrv5_dict.txt         # character dictionary (for decoding)
├── meterocr.py                  # inference module (preprocess + ONNX + CTC decode)
├── requirements.txt
└── README.md
```

## Model
- **Version:** v3 (latest retrain, 2026-07-24; requantized 2026-07-25).
- **Architecture:** PP-OCRv5 server recognizer, fine-tuned on meter serials, INT8 (PTQ)
- **Size:** 22 MB · **Input:** `[N,3,48,W]` BGR crop · **Output:** serial string + confidence
- **Accuracy:** ~85% on unseen out-of-distribution utilities; ~95% on in-domain data
  (measured exact-match on readable crops)

## Install
```bash
pip install -r requirements.txt
```

## Use
```python
from meterocr import MeterOCR

ocr = MeterOCR()                              # loads model/ automatically
serial, score, status = ocr.predict("crop.jpg")
print(serial, score, status)
# e.g.  TPSODL TS0169689  0.99  accepted
```

Or from the command line:
```bash
python meterocr.py crop1.jpg crop2.jpg
```

## Output / confidence gate
`predict()` returns `(serial, score, status)`:
- **`accepted`** — score ≥ 0.95, safe to trust automatically
- **`review`**   — below 0.95, send to a human check
- **`blank`**    — nothing readable

Tune the threshold: `ocr.predict(img, threshold=0.90)`.

## Notes
- Feed the **cropped serial region** (this is a recognizer, not a detector).
- GPU: `MeterOCR(providers=["CUDAExecutionProvider","CPUExecutionProvider"])`
  (requires `onnxruntime-gpu`).
- Preprocessing (resize to 48×320 + normalize) is built in — just pass the raw crop.
