#!/usr/bin/env python3
"""
Merge the 3-model results with the bot ground truth for the TPDDL 1st-4th July
set, matched by image filename.

Inputs:
    Some_analysis_part/three_model_results_tpddl_julyset.csv
        columns: image_name, fp32, ptq_v3, ptq_v2, all_agree
    Some_analysis_part/1stjuly_4thjuly_2026_serialno_tpddl.csv
        columns incl. 'bigImg' (URL whose last path segment is the filename)
        and 'readings.KWH.bot_annotation.serial_value' (the bot's reading)

Output:
    Some_analysis_part/merged_3models_vs_bot_tpddl.csv
        image_name, fp32, ptq_v3, ptq_v2, all_agree, bot_serial, bigImg_url,
        fp32_eq_bot, ptq_v3_eq_bot, ptq_v2_eq_bot

Matching is by filename only (stdlib csv, no pandas needed).

Usage:
    python3 scripts/merge_3models_vs_bot_tpddl.py
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, "Some_analysis_part")

MODELS_CSV = os.path.join(ANALYSIS, "three_model_results_tpddl_julyset.csv")
BOT_CSV = os.path.join(ANALYSIS, "1stjuly_4thjuly_2026_serialno_tpddl.csv")
OUT_CSV = os.path.join(ANALYSIS, "merged_3models_vs_bot_tpddl.csv")

BOT_IMG_COL = "bigImg"
BOT_SERIAL_COL = "readings.KWH.bot_annotation.serial_value"


def norm(s):
    return (s or "").strip().upper().replace(" ", "")


def main():
    # ---- load bot ground truth, keyed by image filename ----
    bot = {}
    bot_url = {}
    with open(BOT_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            url = r[BOT_IMG_COL].strip()
            name = url.rstrip("/").split("/")[-1]
            bot[name] = (r.get(BOT_SERIAL_COL) or "").strip()
            bot_url[name] = url
    print(f"bot rows: {len(bot)}")

    # ---- load model results ----
    with open(MODELS_CSV, encoding="utf-8") as f:
        model_rows = list(csv.DictReader(f))
    print(f"model-result rows: {len(model_rows)}")

    model_cols = [c for c in model_rows[0].keys() if c not in ("image_name", "all_agree")]

    matched = 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = (["image_name"] + model_cols + ["all_agree", "bot_serial", "bigImg_url"]
                      + [f"{m}_eq_bot" for m in model_cols])
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in model_rows:
            name = row["image_name"]
            bot_serial = bot.get(name, "")
            if name in bot:
                matched += 1
            out = {"image_name": name, "all_agree": row["all_agree"], "bot_serial": bot_serial,
                   "bigImg_url": bot_url.get(name, "")}
            for m in model_cols:
                out[m] = row[m]
                out[f"{m}_eq_bot"] = int(norm(row[m]) == norm(bot_serial) and norm(bot_serial) != "")
            w.writerow(out)

    print(f"matched to bot csv: {matched} / {len(model_rows)}")
    print(f"wrote {OUT_CSV}")

    # quick accuracy summary vs bot, on matched rows only
    n = matched
    if n:
        with open(OUT_CSV, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["bot_serial"]]
        print("\n=== exact-match vs bot (matched rows only) ===")
        for m in model_cols:
            correct = sum(1 for r in rows if int(r[f"{m}_eq_bot"]))
            print(f"  {m:8s}  {correct:5d}/{len(rows)} = {100.0*correct/len(rows):6.2f}%")


if __name__ == "__main__":
    main()
