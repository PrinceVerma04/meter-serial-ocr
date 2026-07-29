#!/usr/bin/env python3
"""Benchmark inference time (CPU + GPU) for the three models:
  1. Stock det+rec (full pipeline, stock server_det + server_rec)
  2. Recognition-only (fine-tuned recognizer, inference_v3)
  3. Fine-tuned det+rec (full pipeline, stock det + fine-tuned rec)

Reports single-image (median of repeats) and avg-over-10 (distinct images),
timing only the predict() call. Runs each on GPU then CPU.
"""
import os, sys, time, glob, statistics
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMGDIR = os.path.join(ROOT, "26788 images for test", "agreed_fp32_oldptq_qat", "images")
FT_REC = os.path.join(ROOT, "finetune", "inference_v3")

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from paddleocr import PaddleOCR, TextRecognition  # noqa: E402

imgs = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(IMGDIR, "*.jpg")))[:10]]


def build(kind, device):
    dev = dict(device="gpu") if device == "gpu" else dict(device="cpu", enable_mkldnn=False)
    if kind == "stock_full":
        return PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                         use_textline_orientation=False,
                         text_detection_model_name="PP-OCRv5_server_det",
                         text_recognition_model_name="PP-OCRv5_server_rec", **dev)
    if kind == "recog_only":
        return TextRecognition(model_dir=FT_REC, **dev)
    if kind == "ft_full":
        return PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                         use_textline_orientation=False,
                         text_detection_model_name="PP-OCRv5_server_det",
                         text_recognition_model_dir=FT_REC, **dev)


def bench(kind, label, device):
    try:
        eng = build(kind, device)
    except Exception as e:
        print(f"[{label} / {device.upper()}] build failed: {e}")
        return
    for _ in range(3):  # warmup
        eng.predict(imgs[0])
    single = []
    for _ in range(12):
        t = time.perf_counter(); eng.predict(imgs[0]); single.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    for im in imgs:
        eng.predict(im)
    avg10 = (time.perf_counter() - t) / len(imgs) * 1000
    print(f"[{label:26} / {device.upper()}]  single {statistics.median(single):7.1f} ms   avg10 {avg10:7.1f} ms/img")
    del eng


models = [("stock_full", "Stock det+rec"),
          ("recog_only", "Recognition-only (FT)"),
          ("ft_full", "Fine-tuned det+rec")]

for device in ["gpu", "cpu"]:
    print(f"\n===== {device.upper()} =====")
    for kind, label in models:
        bench(kind, label, device)
