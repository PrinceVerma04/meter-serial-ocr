#!/usr/bin/env python3
"""
Move (or copy) the images you flagged as "Not visible" in the review page into
a separate folder.

The review page (results/roi_review.html) exports roi_results.csv with a
`verdict` column. This script pulls every row whose verdict is "not_visible"
and puts those crops in one folder so the poor-quality images are set aside.

By default it COPIES (non-destructive). Pass --move to actually move them out
of the source folder.

Usage:
  # after clicking "Download results.csv" in the page, point at that CSV:
  python3 scripts/move_not_visible.py --csv ~/Downloads/roi_results.csv
  python3 scripts/move_not_visible.py --csv ~/Downloads/roi_results.csv --move
  python3 scripts/move_not_visible.py --csv results/roi_results.csv \
        --dest "dataset for test/ROI_not_visible"
"""
import csv, os, shutil, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/roi_results.csv",
                help="the roi_results.csv exported from the review page")
ap.add_argument("--src", default="dataset for test/ROI",
                help="fallback source folder if the CSV has no image_path column")
ap.add_argument("--dest", default="dataset for test/ROI_not_visible",
                help="folder to place the flagged crops in")
ap.add_argument("--move", action="store_true",
                help="MOVE the files (remove from source) instead of copying")
args = ap.parse_args()

if not os.path.exists(args.csv):
    raise SystemExit(f"CSV not found: {args.csv}\n"
                     f"Click 'Download results.csv' in the review page first, "
                     f"then pass its path with --csv.")

os.makedirs(args.dest, exist_ok=True)

rows = list(csv.DictReader(open(args.csv)))
flagged = [r for r in rows if (r.get("verdict") or "").strip() == "not_visible"]
print(f"{len(flagged)} images flagged 'not_visible' (out of {len(rows)} rows)")

done, missing = 0, 0
for r in flagged:
    # prefer the absolute path the page recorded; fall back to src/<image>
    src = (r.get("image_path") or "").strip()
    if not src or not os.path.exists(src):
        src = os.path.join(args.src, r["image"])
    if not os.path.exists(src):
        print(f"  ! missing, skipped: {r['image']}")
        missing += 1
        continue
    dst = os.path.join(args.dest, os.path.basename(src))
    (shutil.move if args.move else shutil.copy2)(src, dst)
    done += 1

verb = "moved" if args.move else "copied"
print(f"\n{verb} {done} images -> {args.dest}"
      + (f"  ({missing} missing/skipped)" if missing else ""))
