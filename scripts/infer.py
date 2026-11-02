#!/usr/bin/env python3
"""
Run serial-number inference on a SINGLE image or a WHOLE FOLDER of images.

Uses the same pipeline as scripts/extract_serials.py (fine-tuned PP-OCRv5
recognition-only model + confidence gate), so results match the main pipeline.

Examples
--------
  # one image  -> prints the reading to the terminal
  python3 scripts/infer.py datasets/test/tpddl/images/c09yz8olRxqAmao10dnaa-_null_bigImg_1781491789.jpg

  # a folder   -> reads every image, writes a CSV, prints a summary
  python3 scripts/infer.py datasets/test/tpddl/images
  python3 scripts/infer.py datasets/test/tpddl/images --out results/tpddl_infer.csv

  # options
  --threshold 0.95   confidence gate for accepted/review (default 0.95)
  --base-model       use the stock recognizer instead of the fine-tuned one
  --recursive        also search sub-folders when a folder is given
"""
import os, csv, glob, argparse

# reuse the exact engine + logic from the main pipeline (no duplication)
import extract_serials as es

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def gather_images(path, recursive=False):
    """Return a list of image files from a file path or a folder path."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        files = []
        for root, _dirs, names in (os.walk(path) if recursive else [(path, [], os.listdir(path))]):
            for n in names:
                if n.lower().endswith(IMG_EXTS):
                    files.append(os.path.join(root, n))
        return sorted(files)
    return []


def infer_one(path):
    """Run the pipeline on one image -> (display_text, serial_only, score)."""
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        return "", "", 0.0
    lines_raw = es.run_ocr(bgr)
    lines_pre = es.run_ocr(es.preprocess(bgr))
    text, tscore = es.choose_text(es.display_text(lines_raw), es.display_text(lines_pre))
    serial, _ = es.choose_serial(es.best_serial(lines_raw), es.best_serial(lines_pre))
    return text, serial, tscore


def gate(text, score, threshold):
    if not text:
        return "blank", ""
    if score >= threshold:
        return "accepted", text
    return "review", "NEEDS_REVIEW"


def main():
    ap = argparse.ArgumentParser(description="Infer meter serials on one image or a folder.")
    ap.add_argument("input", help="path to a single image OR a folder of images")
    ap.add_argument("--out", default=None,
                    help="CSV output path (default results/<foldername>_infer.csv for a folder)")
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="confidence gate: >= is auto-accepted, below -> NEEDS_REVIEW")
    ap.add_argument("--base-model", action="store_true",
                    help="use the stock recognizer instead of the fine-tuned one")
    ap.add_argument("--recursive", action="store_true",
                    help="search sub-folders too when a folder is given")
    args = ap.parse_args()

    files = gather_images(args.input, args.recursive)
    if not files:
        raise SystemExit(f"No image found at: {args.input}")

    es.build_engine("recog", use_finetuned=not args.base_model)

    # ---- single image: just print ----
    if len(files) == 1 and os.path.isfile(args.input):
        text, serial, score = infer_one(files[0])
        status, final = gate(text, score, args.threshold)
        print(f"\nImage        : {files[0]}")
        print(f"Reading      : {text or '(nothing read)'}")
        print(f"Serial only  : {serial or '(none)'}")
        print(f"Confidence   : {score:.3f}")
        print(f"Status       : {status}   ->  final: {final or '(blank)'}")
        return

    # ---- folder: write a CSV ----
    out = args.out
    if out is None:
        base = os.path.basename(os.path.normpath(args.input)) or "images"
        out = os.path.join("results", f"{base}_infer.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"Processing {len(files)} images ...", flush=True)
    rows = []
    for i, f in enumerate(files, 1):
        text, serial, score = infer_one(f)
        status, final = gate(text, score, args.threshold)
        rows.append((os.path.basename(f), os.path.abspath(f), text, serial,
                     round(score, 3), final, status))
        if i % 25 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "image_path", "display_text", "serial_only",
                    "score", "final_serial", "status"])
        w.writerows(rows)

    acc = sum(1 for r in rows if r[6] == "accepted")
    rev = sum(1 for r in rows if r[6] == "review")
    bl = sum(1 for r in rows if r[6] == "blank")
    print(f"\nDone. {len(rows)} images @ threshold {args.threshold}: "
          f"{acc} accepted, {rev} review, {bl} blank.\nWrote {out}")


if __name__ == "__main__":
    main()
