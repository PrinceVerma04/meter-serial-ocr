# Meter Serial OCR — QAT INT8 (deployment package)

Self-contained OCR that reads the alphanumeric serial number from a cropped
electricity-meter nameplate image. This is the **quantization-aware-trained (QAT)
INT8** model — the most accurate small model (~92.8% vs the ONNX PTQ build's ~85%).

**Runtime:** PaddlePaddle (any recent 3.x). If you can't install PaddlePaddle, use the
ONNX package instead: `deploy_meterocr_ptq_int8/`.

## Contents
```
deploy_meterocr_qat_int8/
├── model/
│   ├── meterocr_qat_int8.pdmodel     # graph
│   ├── meterocr_qat_int8.pdiparams   # 22 MB INT8 weights
│   └── ppocrv5_dict.txt              # character dictionary
├── meterocr.py                       # inference (Paddle Inference + CTC decode)
├── requirements.txt
└── README.md
```

## Model
- **Architecture:** PP-OCRv5 server recognizer, fine-tuned on meter serials, INT8 (QAT)
- **Size:** 22 MB · **Input:** `[N,3,48,W]` BGR crop · **Output:** serial + confidence
- **Accuracy:** ~92.8% exact-match on unseen out-of-distribution utilities (readable crops);
  ~95%+ on in-domain data

## Install
```bash
pip install -r requirements.txt          # CPU
# GPU: pip install paddlepaddle-gpu==<ver matching your CUDA>  (then use_gpu=True)
```

## Use
```python
from meterocr import MeterOCR

ocr = MeterOCR()                              # CPU; GPU: MeterOCR(use_gpu=True)
serial, score, status = ocr.predict("crop.jpg")
print(serial, score, status)
# e.g.  TPSODL TS0169689  0.99  accepted
```

Command line:
```bash
python meterocr.py crop1.jpg crop2.jpg
```

## Output / confidence gate
`predict()` returns `(serial, score, status)`:
- **`accepted`** — score ≥ 0.95, safe to trust automatically
- **`review`**   — below 0.95, send to a human check
- **`blank`**    — nothing readable

Tune: `ocr.predict(img, threshold=0.90)`.

## Notes
- Feed the **cropped serial region** (recognizer, not a detector).
- Preprocessing (resize 48×320 + normalize) is built in — pass the raw crop.
- QAT vs PTQ: QAT is ~7.5 pts more accurate but needs PaddlePaddle; PTQ (ONNX) is
  lighter to deploy but less accurate. Pick per your environment.
