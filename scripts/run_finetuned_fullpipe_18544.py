#!/usr/bin/env python3
"""Run stock server_det + FINE-TUNED server_rec (inference_v3, trained on the
6,733-image set) as a FULL det+rec pipeline on the 18,544 set, scored vs GT.

This is the fine-tuned counterpart to run_stock_fullpipe_18544.py (which used the
stock recognizer). Detector is stock (can't be fine-tuned -- data has no boxes).

Writes: 26788 images for test/agreed_fp32_oldptq_qat/finetuned_fullpipe_scored.csv
"""
import csv, os, sys, time, argparse, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "26788 images for test")
CDIR = os.path.join(BASE, "agreed_fp32_oldptq_qat")
SRC = os.path.join(BASE, "images")
GT = os.path.join(CDIR, "result.csv")
OUT = os.path.join(CDIR, "finetuned_fullpipe_scored.csv")
FINETUNED_REC = os.path.join(ROOT, "finetune", "inference_v3")

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cv2                       # noqa: E402
import paddle                    # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402
import extract_serials as E      # reuse best_serial()  # noqa: E402


def norm(s): return (s or "").strip().upper().replace(" ", "")


def lev(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = [(r["filename"], r["result"]) for r in csv.DictReader(open(GT))]
    if args.limit:
        items = items[:args.limit]
    done = set()
    if os.path.exists(OUT) and not args.limit:
        done = {r["filename"] for r in csv.DictReader(open(OUT))}
    todo = [(f, g) for f, g in items if f not in done]
    print(f"total={len(items)} done={len(done)} todo={len(todo)}", flush=True)

    on_gpu = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    dev = dict(device="gpu") if on_gpu else dict(enable_mkldnn=False)
    engine = PaddleOCR(
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_dir=FINETUNED_REC,      # fine-tuned recognizer
        **dev)
    print(f"stock det + FINE-TUNED rec on {'GPU' if on_gpu else 'CPU'}", flush=True)

    def read(bgr):
        lines = []
        for r in engine.predict(bgr):
            d = r if isinstance(r, dict) else getattr(r, "res", r)
            for t, s in zip(d.get("rec_texts", []), d.get("rec_scores", [])):
                lines.append((t, float(s)))
        return E.best_serial(lines)

    write_header = not os.path.exists(OUT) or args.limit
    t0 = time.time()
    with open(OUT, "w" if args.limit else "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["filename", "ground_truth", "finetuned_pred", "score", "edit_distance", "match"])
        for i, (fn, gt) in enumerate(todo, 1):
            bgr = cv2.imread(os.path.join(SRC, fn))
            serial, score = read(bgr) if bgr is not None else ("", 0.0)
            d = lev(norm(gt), norm(serial))
            w.writerow([fn, gt, serial, round(float(score), 4), d, 1 if d == 0 else 0])
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  {rate:.2f} img/s  ETA {(len(todo)-i)/rate/60:.1f} min", flush=True)
                f.flush()
    dt = time.time() - t0
    print(f"\ndone {len(todo)} imgs in {dt:.1f}s ({len(todo)/dt:.2f} img/s)")


if __name__ == "__main__":
    main()
