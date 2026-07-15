#!/usr/bin/env python3
"""
Build a clean ground-truth CSV from the human-reviewed ROI results, for
comparing OCR models and finalising which model to use.

Input : temporary result only/roi_results.csv   (the review-page export)
Output: 3 columns  ->  image_path, correct_label, correct_serial
  * image_path     - absolute path to the crop
  * correct_label  - the full human-verified reading (keeps letter prefix, e.g.
                     LW512323, TWSP51061127); blank for 'not_visible' crops
  * correct_serial - the distilled serial number only, derived from the label
                     with the SAME brand/prefix logic as extract_serials.py
                     (TPWODL1037121 -> 1037121, TWECE -> '' (no digits))

Only REVIEWED images are written (verdict in correct/wrong/not_visible); the
still-'unchecked' rows have no ground truth and are skipped.
"""
import csv, os, re, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# --- serial-distilling logic, copied verbatim from scripts/extract_serials.py ---
MIN_LEN = 5
LABEL_WORDS = {
    "BLE", "SRNO", "SR", "NO", "CAL", "MADEIN", "MADE", "IN", "INDIA",
    "JINNO", "CAC", "TYPE", "MODEL", "KWH", "KVAH", "PHASE", "IMP", "EXP",
    "CLASS", "METER", "SERIAL",
}
_LEADING_ALPHA = re.compile(r"^([A-Z]+)(\d.*)$")

def strip_fused_brand(tok):
    m = _LEADING_ALPHA.match(tok)
    if m and (len(m.group(1)) >= 5 or "ODL" in m.group(1)):
        return m.group(2)
    return tok

def best_serial(text):
    """Distil the serial number from a (clean, human-verified) label string."""
    best = None  # (rank_key, serial)
    toks = re.sub(r"[^A-Z0-9]", " ", text.upper()).split()
    for i, tok in enumerate(toks):
        core = strip_fused_brand(tok)
        if not any(c.isdigit() for c in core):
            continue                       # need digits; bare letters aren't a serial
        cand = core
        if i > 0:                          # maybe a separated letter-prefix precedes it
            prev = toks[i - 1]
            if (prev.isalpha() and 2 <= len(prev) <= 4
                    and prev not in LABEL_WORDS and "ODL" not in prev):
                cand = prev + core
        if len(cand) < MIN_LEN:
            continue
        digits = sum(c.isdigit() for c in cand)
        key = (digits, len(cand))
        if best is None or key > best[0]:
            best = (key, cand)
    return best[1] if best else ""


ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="temporary result only/roi_results.csv",
                help="the roi_results.csv exported from the review page")
ap.add_argument("--out", default="temporary result only/roi_ground_truth.csv")
args = ap.parse_args()

REVIEWED = {"correct", "wrong", "not_visible"}  # rows that have a human verdict

rows = list(csv.DictReader(open(args.csv)))
out = []
n_serial = n_blank = 0
for r in rows:
    if (r.get("verdict") or "").strip() not in REVIEWED:
        continue                                    # skip 'unchecked' (no ground truth)
    label = (r.get("correct_serial") or "").strip()  # human-verified full reading
    serial = best_serial(label) if label else ""
    out.append((r["image_path"], label, serial))
    if label:
        n_serial += 1
    else:
        n_blank += 1

with open(args.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["image_path", "correct_label", "correct_serial"])
    w.writerows(out)

print(f"Wrote {len(out)} reviewed rows -> {args.out}")
print(f"  {n_serial} with a verified label/serial, {n_blank} blank (not_visible)")
