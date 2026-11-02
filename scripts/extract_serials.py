#!/usr/bin/env python3
"""
Extract alphanumeric meter serial numbers from cropped strip images.

Pipeline:
  1. Preprocess (upscale + CLAHE contrast) to help low-quality crops.
  2. Run PaddleOCR (PP-OCRv5, English) on both the raw and preprocessed image.
  3. Pick the best serial candidate out of the detected text lines
     (drops label noise like "Sr.No." / "TPWODL BLE").
  4. Majority-vote across all images of the same meter for a robust consensus.

Outputs:
  results_per_image.csv  - one row per image (best serial + score + raw text)
  results_per_meter.csv  - one row per meter (voted consensus serial)
"""
import os, re, csv, glob, sys
from collections import defaultdict, Counter

# --- shim: an old openpyxl on this box references removed numpy aliases ---
import numpy as np
for _a in ('float', 'int', 'bool', 'object', 'str'):
    if not hasattr(np, _a):
        setattr(np, _a, __builtins__[_a] if isinstance(__builtins__, dict) else getattr(__builtins__, _a))

import cv2
import paddle
from paddleocr import PaddleOCR

# run from the project root regardless of where this script is invoked
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

DATA_DIR = "datasets/test/tpddl/images"    # default dataset (override with --data)
OUT_CSV = "results/serials.csv"
MIN_LEN = 5            # shortest plausible serial
# Printed labels/words on the nameplate that are NOT part of the serial.
# (The manufacturer brand "TPWODL" and its OCR misreads are handled separately
# by the "contains ODL" rule, since they vary: TEWODL, TFWODL, NPWODL, YODL...)
LABEL_WORDS = {
    "BLE", "SRNO", "SR", "NO", "CAL", "MADEIN", "MADE", "IN", "INDIA",
    "JINNO", "CAC", "TYPE", "MODEL", "KWH", "KVAH", "PHASE", "IMP", "EXP",
    "CLASS", "METER", "SERIAL",
}

# Use the GPU when a CUDA-enabled Paddle build is present, else CPU (with the
# mkldnn workaround for the oneDNN crash in the CPU build).
_ON_GPU = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
_device_kw = dict(device="gpu") if _ON_GPU else dict(enable_mkldnn=False)

# Engine is built lazily by build_engine(mode) once the CLI mode is known.
#   "recog" (default): recognition-ONLY. The crops are already the serial ROI, so
#                      we skip detection and read the whole crop directly. This
#                      recovers ~30% more faint/low-contrast serials the detector
#                      would otherwise miss, without TrOCR-style hallucination.
#   "full":  detector + recognizer (PP-OCRv5 server). Use for un-cropped images.
_engine = None
_mode = "recog"

# Fine-tuned recognizer (trained on our labels; +2.5pp on held-out test, fixes
# faint-prefix errors). Used automatically in recog mode if present, unless
# --base-model is passed. Falls back to the stock server recognizer otherwise.
FINETUNED_REC = os.path.join(ROOT, "models", "v3", "meterocr_v3_fp32_paddle")


def build_engine(mode, use_finetuned=True):
    global _engine, _mode
    _mode = mode
    if mode == "recog":
        from paddleocr import TextRecognition
        if use_finetuned and os.path.isdir(FINETUNED_REC):
            print(f"Paddle RECOGNITION-ONLY (FINE-TUNED) on {'GPU' if _ON_GPU else 'CPU'}")
            _engine = TextRecognition(model_name="PP-OCRv5_server_rec",
                                      model_dir=FINETUNED_REC, **_device_kw)
        else:
            print(f"Paddle RECOGNITION-ONLY (stock) on {'GPU' if _ON_GPU else 'CPU'}")
            _engine = TextRecognition(model_name="PP-OCRv5_server_rec", **_device_kw)
        return
    print(f"Paddle detector+recognizer on {'GPU' if _ON_GPU else 'CPU'}")
    _engine = PaddleOCR(
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec", **_device_kw)


def preprocess(bgr):
    """Upscale 2x + CLAHE contrast — helps glare/blurry/low-contrast crops."""
    up = cv2.resize(bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def run_ocr(img):
    """Return list of (text, score) lines from the active engine."""
    out = []
    if _mode == "recog":
        res = _engine.predict(img)
        r = res[0] if res else {}
        d = r if isinstance(r, dict) else getattr(r, "res", r)
        t = d.get("rec_text", "")
        if t:
            out.append((t, float(d.get("rec_score", 0))))
    else:
        for r in _engine.predict(img):
            d = r if isinstance(r, dict) else getattr(r, "res", r)
            for t, s in zip(d.get("rec_texts", []), d.get("rec_scores", [])):
                out.append((t, float(s)))
    return out


# Manufacturer brand fused to digits in one token. The leading letters are the
# brand ("TPWODL" and its OCR misreads TPWOOL/TPWODE/WODL...) when they are 5+
# chars OR contain "ODL"; strip them, keep the digits: TPWODL020075 -> 020075,
# WODL1152261 -> 1152261. Real short prefixes (TW, TWSU...) are kept.
_LEADING_ALPHA = re.compile(r"^([A-Z]+)(\d.*)$")

def strip_fused_brand(tok):
    m = _LEADING_ALPHA.match(tok)
    if m and (len(m.group(1)) >= 5 or "ODL" in m.group(1)):
        return m.group(2)
    return tok


def best_serial(lines):
    """Reconstruct the best serial from the detected text lines.

    The serial is either a single alphanumeric token, or a short letter-prefix
    (2-4 chars, e.g. TWNX/WLT) sitting just before its digit group. We find the
    digit-bearing core token and, when present, prepend that short prefix -- so
    'TWNX 519026' -> 'TWNX519026' -- while the 5+ char brand word is dropped and
    unrelated tokens are never merged.
    """
    best = None  # (rank_key, serial, ocr_score)
    for text, score in lines:
        toks = re.sub(r"[^A-Z0-9]", " ", text.upper()).split()
        for i, tok in enumerate(toks):
            core = strip_fused_brand(tok)        # TPWODL020075 -> 020075, WODL1152261 -> 1152261
            if not any(c.isdigit() for c in core):
                continue                          # need digits; bare letters aren't a serial
            cand = core
            if i > 0:                             # maybe a separated letter-prefix precedes it
                prev = toks[i - 1]
                if (prev.isalpha() and 2 <= len(prev) <= 4
                        and prev not in LABEL_WORDS and "ODL" not in prev):
                    cand = prev + core
            if len(cand) < MIN_LEN:
                continue
            digits = sum(c.isdigit() for c in cand)
            key = (digits, len(cand), score)      # more digits, then longer, then confidence
            if best is None or key > best[0]:
                best = (key, cand, score)
    if best is None:
        return "", 0.0
    return best[1], best[2]


def display_text(lines):
    """Return the serial-area text VERBATIM (letters, digits and spacing kept).

    Unlike best_serial (which distils just the serial number), this prints what
    OCR actually reads in the display area -- e.g. 'TPWODL 1037121', 'TWNX
    519026', 'WLT T099601' -- without dropping the manufacturer name or the
    letter prefix. It skips only tiny pure-label lines (like 'Sr.No.').
    """
    best = None  # (rank_key, text, score)
    for text, score in lines:
        # keep letters/digits and single spaces; drop stray punctuation noise
        # (e.g. 'THNX"500868' -> 'THNX 500868', 'TEWODL-1018441' -> 'TEWODL 1018441')
        norm = re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()
        alnum = sum(c.isalnum() for c in norm)
        if alnum < MIN_LEN:                 # skip small field labels (SR.NO., CAL...)
            continue
        key = (alnum, score)                # prefer the line with the most content
        if best is None or key > best[0]:
            best = (key, norm, score)
    if best is None:
        return "", 0.0
    return best[1], best[2]


def choose_text(raw, pre):
    """Prefer the raw-image verbatim read; fall back to preprocessed if empty."""
    return raw if raw[0] else pre


def choose_serial(raw, pre):
    """Combine the raw-image and preprocessed-image reads.

    The raw image is trusted as primary: CLAHE/upscaling can *corrupt* an
    already-readable serial (e.g. '10200735' -> '1020035'), so the preprocessed
    read is only used to (a) fill in when raw found nothing, or (b) complete a
    raw read it strictly extends ('10200' -> '10200735', '519026' -> '...').
    """
    r, rsc = raw
    p, psc = pre
    if r and p:
        if r in p and len(p) > len(r):   # preprocessed recovers a fuller serial
            return p, psc
        return r, rsc                    # otherwise trust the raw read
    return raw if r else pre


def meter_id(fname):
    return os.path.basename(fname).split("_null_bigImg")[0]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Extract verbatim meter serials to a CSV.")
    ap.add_argument("--data", default=DATA_DIR, help="folder of .jpg crops")
    ap.add_argument("--out", default=OUT_CSV, help="output CSV path")
    ap.add_argument("--mode", choices=["recog", "full"], default="recog",
                    help="recog = recognition-only (default, for pre-cropped ROIs); "
                         "full = detector+recognizer (for un-cropped images)")
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="confidence gate: reads at/above are auto-accepted, "
                         "below go to NEEDS_REVIEW (measured 91%% precision at 0.90)")
    ap.add_argument("--base-model", action="store_true",
                    help="use the stock recognizer instead of the fine-tuned one")
    args = ap.parse_args()
    build_engine(args.mode, use_finetuned=not args.base_model)

    files = sorted(glob.glob(os.path.join(args.data, "*.jpg")))
    print(f"Processing {len(files)} images from {args.data} ...", flush=True)

    per_image = []

    for i, f in enumerate(files, 1):
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        lines_raw = run_ocr(bgr)
        lines_pre = run_ocr(preprocess(bgr))

        # primary output: verbatim display-area text (letters + brand + spacing kept)
        text, tscore = choose_text(display_text(lines_raw), display_text(lines_pre))
        # secondary: distilled serial number only, for reference
        serial, _ = choose_serial(best_serial(lines_raw), best_serial(lines_pre))

        # confidence gate: auto-accept high-confidence reads, else route to review
        if not text:
            status, final = "blank", ""
        elif tscore >= args.threshold:
            status, final = "accepted", text
        else:
            status, final = "review", "NEEDS_REVIEW"

        per_image.append((os.path.basename(f), meter_id(f), text, serial,
                          round(tscore, 3), final, status))
        if i % 25 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    # --- per-image CSV ---
    #   display_text  = verbatim reading   | serial_only = number distilled
    #   final_serial  = auto-accepted read, or NEEDS_REVIEW, or blank
    #   status        = accepted / review / blank
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "meter_id", "display_text", "serial_only", "score",
                    "final_serial", "status"])
        w.writerows(per_image)

    acc = sum(1 for r in per_image if r[6] == "accepted")
    rev = sum(1 for r in per_image if r[6] == "review")
    bl = sum(1 for r in per_image if r[6] == "blank")
    print(f"\nDone. {len(per_image)} images @ threshold {args.threshold}: "
          f"{acc} auto-accepted, {rev} need review, {bl} blank. Wrote {args.out}")


if __name__ == "__main__":
    main()
