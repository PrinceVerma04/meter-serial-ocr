#!/usr/bin/env python3
"""
Run the baseline and the deployed model over a folder of serial crops and
record what each one reads, side by side.

  baseline : v3 FP32, original Paddle model  models/v3/meterocr_v3_fp32_paddle/
  deployed : v3 PTQ INT8, ONNX               deploy/ptq_int8/ (its own MeterOCR code)

Both use the same preprocessing (resize to 48x320, normalize), greedy CTC
decode and 0.95 confidence gate, so the only difference is the model.

Output CSV, one row per image:
    image_name,
    fp32_result, fp32_confidence, fp32_status, fp32_ms,
    ptq_result,  ptq_confidence,  ptq_status,  ptq_ms,
    same_result        (yes/no — compared upper-case, spaces ignored)
status: accepted (confidence >= 0.95) | review (below) | blank (nothing read)
plus <out>_summary.txt with per-model counts, agreement and speed.

Runs on CPU in parallel worker processes (each loads both models once). It is
resumable: re-run the same command and images already in the CSV are skipped.

Run from the project root:
    python3 scripts/compare_fp32_ptq.py
    python3 scripts/compare_fp32_ptq.py --workers 8 --limit 50     # quick trial
"""
import os, sys, csv, time, glob, argparse, signal, warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore", message="No ccache found")   # harmless paddle import warning

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deploy", "ptq_int8"))

FP32_DIR = os.path.join(ROOT, "models", "v3", "meterocr_v3_fp32_paddle")
IMAGES = os.path.join(ROOT, "datasets", "test", "clean_test_oct27", "images")
OUT = os.path.join(ROOT, "results", "2026-10-27_v3_clean_test", "fp32_vs_ptq_results.csv")
FIELDS = ["image_name",
          "fp32_result", "fp32_confidence", "fp32_status", "fp32_ms",
          "ptq_result", "ptq_confidence", "ptq_status", "ptq_ms",
          "same_result"]

norm = lambda s: s.strip().upper().replace(" ", "")
_fp32 = _ptq = None


def init_worker():
    """Load both models once per worker, single-threaded so workers don't fight."""
    global _fp32, _ptq
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl+C is handled once, in the main process
    import paddle
    paddle.disable_signal_handler()                # no C++ crash dump when the pool shuts down
    import paddle.inference as pi
    import onnxruntime as ort
    from meterocr import MeterOCR, MODEL

    # deployed: the package's own MeterOCR, with a 1-thread session
    _ptq = MeterOCR()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    _ptq.sess = ort.InferenceSession(MODEL, sess_options=so, providers=["CPUExecutionProvider"])

    # baseline: original Paddle inference model, CPU (oneDNN off — it crashes on this box)
    config = pi.Config(os.path.join(FP32_DIR, "inference.json"),
                       os.path.join(FP32_DIR, "inference.pdiparams"))
    config.disable_gpu()
    config.disable_mkldnn()
    config.set_cpu_math_library_num_threads(1)
    config.disable_glog_info()
    predictor = pi.create_predictor(config)
    _fp32 = (predictor,
             predictor.get_input_handle(predictor.get_input_names()[0]),
             predictor.get_output_handle(predictor.get_output_names()[0]))


def read_fp32(bgr):
    predictor, inp, out = _fp32
    inp.copy_from_cpu(_ptq._preprocess(bgr))      # same preprocessing as deployed
    predictor.run()
    return _ptq._decode(out.copy_to_cpu())        # same CTC decode as deployed


def gate(text, score):
    if not text:
        return "", 0.0, "blank"
    return text, round(score, 4), "accepted" if score >= 0.95 else "review"


def process(path):
    row = {"image_name": os.path.basename(path)}
    bgr = cv2.imread(path)
    if bgr is None:
        row.update(fp32_status="unreadable_file", ptq_status="unreadable_file", same_result="no")
        return row
    for name, fn in (("fp32", read_fp32), ("ptq", _ptq.read)):
        t = time.perf_counter()
        text, score = fn(bgr)
        ms = (time.perf_counter() - t) * 1000
        text, score, status = gate(text, score)
        row.update({f"{name}_result": text, f"{name}_confidence": score,
                    f"{name}_status": status, f"{name}_ms": round(ms, 1)})
    row["same_result"] = "yes" if norm(row["fp32_result"]) == norm(row["ptq_result"]) else "no"
    return row


def summarize(out_csv):
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    n = len(rows)
    pct = lambda k: f"{k:6d}  ({100 * k / n:5.1f}%)" if n else f"{k:6d}"
    lines = [f"Images: {n}", ""]
    for m, title in (("fp32", "Baseline  v3 FP32 (Paddle)"), ("ptq", "Deployed  v3 PTQ INT8 (ONNX)")):
        st = [r[f"{m}_status"] for r in rows]
        ms = [float(r[f"{m}_ms"]) for r in rows if r[f"{m}_ms"]]
        lines += [title,
                  f"  accepted (conf >= 0.95) : {pct(st.count('accepted'))}",
                  f"  review   (conf <  0.95) : {pct(st.count('review'))}",
                  f"  blank    (no read)      : {pct(st.count('blank'))}",
                  f"  avg time per image      : {np.mean(ms):.1f} ms" if ms else "", ""]
    same = sum(r["same_result"] == "yes" for r in rows)
    both_acc = [r for r in rows if r["fp32_status"] == "accepted" and r["ptq_status"] == "accepted"]
    lines += ["Agreement (same serial from both models)",
              f"  all images              : {pct(same)}",
              f"  both accepted           : {len(both_acc)}  -> same serial in "
              f"{sum(r['same_result'] == 'yes' for r in both_acc)}",
              "",
              "No ground truth was used: these numbers show what each model reads and how",
              "confident it is, not accuracy. Times are measured inside parallel workers."]
    text = "\n".join(lines)
    path = os.path.splitext(out_csv)[0] + "_summary.txt"
    open(path, "w", encoding="utf-8").write(text + "\n")
    print("\n" + text + f"\n\nsummary -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=IMAGES, help="folder of serial crops")
    ap.add_argument("--out", default=OUT, help="output CSV")
    ap.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) - 2)),
                    help="parallel processes (each holds both models, ~1 GB RAM)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (trial run)")
    args = ap.parse_args()

    images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG")
                    for p in glob.glob(os.path.join(args.images, ext)))
    if args.limit:
        images = images[:args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {r["image_name"] for r in csv.DictReader(open(args.out, encoding="utf-8"))}
    todo = [p for p in images if os.path.basename(p) not in done]
    print(f"images: {len(images)}  already done: {len(done & {os.path.basename(p) for p in images})}  "
          f"to run: {len(todo)}  workers: {args.workers}")

    if todo:
        new_file = not os.path.exists(args.out)
        print(f"loading both models in {args.workers} workers (takes ~30 s)...", flush=True)
        start = time.time()
        pool = Pool(args.workers, initializer=init_worker)
        try:
            with open(args.out, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if new_file:
                    w.writeheader()
                for i, row in enumerate(pool.imap_unordered(process, todo, chunksize=4), 1):
                    w.writerow(row)
                    if i == 1:
                        print(f"  models loaded, first result after {time.time() - start:.0f} s", flush=True)
                    if i % 50 == 0 or i == len(todo):
                        fh.flush()
                        rate = i / (time.time() - start)
                        print(f"  {i}/{len(todo)}  {rate:.1f} img/s  ~{(len(todo) - i) / rate / 60:.1f} min left",
                              flush=True)
            pool.close()
        except KeyboardInterrupt:
            pool.terminate()
            print("\nstopped — results so far are saved. Run the same command again to continue.")
            return
        finally:
            pool.join()
    summarize(args.out)


if __name__ == "__main__":
    main()
