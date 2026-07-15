# Size Reduction — Path A (INT8 ~20 MB) & Path B (Mobile ~4 MB)

Shrink the retrained recognizer and test the reduced models side-by-side against
the full-size model on `new dataset for test/`.

Starting model: **`finetune/inference_v2/`** (retrained PP-OCRv5 **server** rec, 81 MB FP32,
79.7% on the new-dataset sample).

## The two paths

| Path | Method | Result size | Accuracy | Runnable |
|---|---|---|---|---|
| **A** | Quantize the server model to INT8 (via ONNX) | **~20 MB** | ~same (−1–3%) | `path_a_server_int8/quantize.py` |
| **B** | Fine-tune the **mobile** rec on our 6,709 labels, then INT8 | **~2–4 MB** | lower (smaller net) | `path_b_mobile_4mb/` |

### Why 20 MB is the floor for Path A
81 MB FP32 → INT8 is a 4× reduction (32-bit → 8-bit weights) = ~20 MB. You cannot
reach 4 MB by quantizing this network — 4 MB needs *fewer parameters* = a smaller
architecture (that is Path B). Below INT8 (INT4, pruning) either wrecks accuracy or
needs another retrain. Gzip helps on-disk only; the model still runs at full size.

## Folder layout
```
size_reduction/
  README.md
  requirements.txt              # onnx / onnxruntime / paddle2onnx / paddleslim
  path_a_server_int8/
    quantize.py                 # inference_v2 -> ONNX FP32 -> ONNX INT8 (dynamic)
  path_b_mobile_4mb/
    prepare_mobile_config.py    # writes the mobile fine-tune yml
    train_mobile.sh             # fine-tune en_PP-OCRv5_mobile_rec on our dataset
    quantize_mobile.py          # exported mobile -> ONNX INT8 (~4 MB)
  onnx_reader.py                # ONNX inference wrapper (pre/post + CTC decode)
  test_compare.py               # run any models on new dataset, side-by-side CSV
  models/                       # produced models land here
```

## Order to run (all foreground)
```bash
cd "/home/ganesh/Desktop/Prince number detection in meter"
pip install -r size_reduction/requirements.txt

# Path A
python3 size_reduction/path_a_server_int8/quantize.py

# Path B
python3 size_reduction/path_b_mobile_4mb/prepare_mobile_config.py
bash    size_reduction/path_b_mobile_4mb/train_mobile.sh        # ~20-30 min
python3 size_reduction/path_b_mobile_4mb/quantize_mobile.py

# Side-by-side accuracy test on the new dataset
python3 size_reduction/test_compare.py --n 500 --seed 42
```
