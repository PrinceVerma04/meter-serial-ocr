#!/usr/bin/env python3
"""
STEP 1 of 2 - run every model on the test images and save what each one reads.
This step never opens the ground truth file; it only reads the images.

Models (all ONNX, CPU): v3 FP32, v3 PTQ (deployed), v4 FP32, v4 PTQ (whichever exist).
All go through the deploy code (deploy/ptq_int8/meterocr.py): same preprocessing, greedy CTC
decode and confidence as production (and as the dashboard).

Input : new testing data 4 nov/clean/            (images only)
Output: results/2026-11-04_all_models_test/model_reads.xlsx
          sheet "extracted_results": image_name + extracted result of each model
          sheet "confidence"       : image_name + confidence of each model (used by step 2
                                     for the 0.95 auto-accept numbers)

Resumable: progress is kept in model_reads_progress.csv while running; Ctrl+C keeps what is
done and re-running the same command continues. The progress file is removed once every
image in the folder is done.

Run from the project root:
    .venv/bin/python scripts/step1_run_all_models.py --limit 50    # trial on 50 images
    .venv/bin/python scripts/step1_run_all_models.py               # all images (continues)
"""
import argparse
import csv
import os
import signal
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deploy", "ptq_int8"))

IMAGES = os.path.join(ROOT, "new testing data 4 nov", "clean")
OUT_DIR = os.path.join(ROOT, "results", "2026-11-04_all_models_test")

# key, file  (same list as dashboard/app.py)
MODELS = [
    ("v3_fp32", "models/v3/meterocr_v3_fp32.onnx"),
    ("v3_ptq", "models/v3/meterocr_v3_int8_ptq.onnx"),
    ("v4_fp32", "models/v4/meterocr_v4_fp32.onnx"),
    ("v4_ptq", "models/v4/meterocr_v4_int8_ptq.onnx"),
]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ---------------------------------------------------------------- worker side
_READERS = {}


def _init_worker(model_list):
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # main process handles Ctrl+C
    import onnxruntime as ort
    from meterocr import DICT, MeterOCR

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    chars = [l.rstrip("\n") for l in open(DICT, encoding="utf-8")]
    for key, path in model_list:
        r = MeterOCR.__new__(MeterOCR)              # deploy class, 1-thread session
        r.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        r.inp = r.sess.get_inputs()[0].name
        r.character = ["blank"] + chars + [" "]
        _READERS[key] = r


def _read_one(item):
    import cv2
    name, path = item
    row = {"image_name": name}
    bgr = cv2.imread(path)
    for key, r in _READERS.items():
        text, score = r.read(bgr) if bgr is not None else ("", 0.0)
        row[f"{key}_result"] = text
        row[f"{key}_confidence"] = round(score, 4) if text else 0.0
    return row


# ---------------------------------------------------------------- excel
def write_xlsx(progress_csv, out_xlsx, keys, order):
    import openpyxl
    from openpyxl.styles import Font
    rows = {r["image_name"]: r for r in csv.DictReader(open(progress_csv, encoding="utf-8"))}
    names = [n for n in order if n in rows]
    wb = openpyxl.Workbook()
    for title, suffix in (("extracted_results", "result"), ("confidence", "confidence")):
        ws = wb.active if title == "extracted_results" else wb.create_sheet(title)
        ws.title = title
        ws.append(["image_name"] + keys)
        for c in ws[1]:
            c.font = Font(bold=True)
        for n in names:
            vals = [rows[n][f"{k}_{suffix}"] for k in keys]
            if suffix == "confidence":
                vals = [float(v) for v in vals]
            ws.append([n] + vals)
        ws.column_dimensions["A"].width = 52
        for i in range(len(keys)):
            ws.column_dimensions[openpyxl.utils.get_column_letter(i + 2)].width = 14
            for c in ws.iter_rows(min_row=2, min_col=i + 2, max_col=i + 2):
                c[0].number_format = "@" if suffix == "result" else "0.0000"
        ws.freeze_panes = "B2"
    tmp = out_xlsx + ".tmp.xlsx"
    wb.save(tmp)
    os.replace(tmp, out_xlsx)
    return len(names)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=IMAGES, help="folder of crops")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 4) - 4)),
                    help="parallel processes, each loads all models (~1 GB RAM each)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (trial run)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    progress = os.path.join(a.out_dir, "model_reads_progress.csv")
    out_xlsx = os.path.join(a.out_dir, "model_reads.xlsx")

    names = sorted(f for f in os.listdir(a.images) if f.lower().endswith(IMG_EXT))
    models = []
    for key, rel in MODELS:
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p):
            models.append((key, p))
        else:
            print(f"missing {key}: {rel}  (skipped)")
    keys = [k for k, _ in models]
    print(f"STEP 1  images {len(names)} in {os.path.relpath(a.images, ROOT)}   models {len(keys)}: {', '.join(keys)}")

    fields = ["image_name"]
    for k in keys:
        fields += [f"{k}_result", f"{k}_confidence"]

    if not os.path.exists(progress) and os.path.exists(out_xlsx):
        # earlier run finished: keep its reads, only read images added to the folder since
        import openpyxl
        wb = openpyxl.load_workbook(out_xlsx, read_only=True)
        er = list(wb["extracted_results"].iter_rows(values_only=True))
        cf = {r[0]: r[1:] for r in list(wb["confidence"].iter_rows(values_only=True))[1:]}
        wb.close()
        if list(er[0][1:]) != keys:
            sys.exit(f"{out_xlsx} has models {list(er[0][1:])}, expected {keys} - delete it to run again")
        old = {r[0]: r for r in er[1:] if r[0] is not None}
        in_folder = set(names)
        if set(old) == in_folder:
            sys.exit(f"already complete: {os.path.relpath(out_xlsx, ROOT)}  (delete it to run again)")
        with open(progress, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(fields)
            for n, r in old.items():
                if n in in_folder:
                    row = [n]
                    for j in range(len(keys)):
                        row += [r[1 + j] or "", cf[n][j] or 0.0]
                    w.writerow(row)
        print(f"kept reads of {len(old.keys() & in_folder)} images from {os.path.relpath(out_xlsx, ROOT)}; "
              f"{len(old.keys() - in_folder)} no longer in the folder are dropped")
    done = set()
    if os.path.exists(progress):
        with open(progress, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != fields:
                sys.exit(f"{progress} was made with a different model list - delete it and re-run")
            done = {r["image_name"] for r in rd}
    todo_names = names[:a.limit] if a.limit else names
    todo = [(n, os.path.join(a.images, n)) for n in todo_names if n not in done]
    print(f"already done {len(done)}, to run now {len(todo)}, workers {a.workers}")

    if todo:
        new = not os.path.exists(progress)
        print("loading models in each worker ...", flush=True)
        start = time.time()
        pool = Pool(a.workers, initializer=_init_worker, initargs=(models,))
        try:
            with open(progress, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                if new:
                    w.writeheader()
                for i, row in enumerate(pool.imap_unordered(_read_one, todo, chunksize=4), 1):
                    w.writerow(row)
                    if i % 100 == 0 or i == len(todo):
                        fh.flush()
                        os.fsync(fh.fileno())
                        rate = i / (time.time() - start)
                        print(f"  {i}/{len(todo)}  {rate:.1f} img/s  ~{(len(todo) - i) / rate / 60:.1f} min left",
                              flush=True)
            pool.close()
        except KeyboardInterrupt:
            pool.terminate()
            sys.exit("\nstopped - results so far are saved; run the same command to continue")
        finally:
            pool.join()

    n = write_xlsx(progress, out_xlsx, keys, names)
    print(f"\nsaved {os.path.relpath(out_xlsx, ROOT)}  ({n} of {len(names)} images)")
    if n == len(names):
        os.remove(progress)
        print("all images done. Next: .venv/bin/python scripts/step2_check_accuracy.py")
    else:
        print(f"{len(names) - n} images still to do - run again without --limit to finish")


if __name__ == "__main__":
    main()
