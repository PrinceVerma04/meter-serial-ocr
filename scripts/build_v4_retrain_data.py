#!/usr/bin/env python3
"""
Build the corrected ground truth and the v4 retraining data from both review rounds.

  round 1: your notes of 2026-11-04       (lists in scripts/apply_review_2026_11_04.py)
  round 2: your dashboard choices         (new testing data 4 nov/review_round2_decisions.csv,
                                           made with scripts/review_round2_server.py)
  round 3: doubtful labels in new crops   (new testing data 4 nov/review_round3_decisions.csv,
                                           made with scripts/review_round3_server.py) + the "07" rule
A later round overrides an earlier one for the same image. Every round-2 crop and the overlap
question must be decided, otherwise nothing is changed. Round 3 is applied once its page has been
opened (review_round3_items.csv exists) and then must be complete; until then the build says so and
datasets/merge_into_v4_train.py refuses to merge.

What it writes (always rebuilt from the untouched original, so safe to re-run):
  new testing data 4 nov/clean_extracted_serials.xlsx   corrected ground truth
                                                        + sheet review_changes (every change, both rounds)
  new testing data 4 nov/clean/ and removed/            images moved to match (nothing deleted)
  results/2026-11-04_all_models_test/model_reads.xlsx   rows of removed images dropped
  datasets/v4_retrain_nov4/images/ + labels.csv         the retraining data
        labels.csv: image_name, serial, in_v4_train (no / train / val / test), v4_train_serial

Run from the project root:
    .venv/bin/python scripts/build_v4_retrain_data.py --dry-run
    .venv/bin/python scripts/build_v4_retrain_data.py
Then refresh the test numbers (step 1 only reads crops brought back, if any):
    .venv/bin/python scripts/step1_run_all_models.py
    .venv/bin/python scripts/step2_check_accuracy.py
    .venv/bin/python scripts/step3_build_dashboard.py
"""
import argparse
import csv
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import apply_review_2026_11_04 as R1          # noqa: E402  round-1 lists + paths
import review_round2_server as R2             # noqa: E402  round-2 item list + decisions file
import review_round3_server as R3             # noqa: E402  round-3 items + decisions + 07 rule
import json                                   # noqa: E402

OUT_DS = os.path.join(ROOT, "datasets", "v4_retrain_nov4")
V4_LABELS = os.path.join(ROOT, "datasets", "v4_train", "labels.csv")


def norm(s):
    return (s or "").strip().upper().replace(" ", "")


def main():
    import openpyxl
    from openpyxl.styles import Font
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # ---- original ground truth (backup made by round 1)
    if not os.path.exists(R1.BACKUP):
        sys.exit(f"{R1.BACKUP} missing - run scripts/apply_review_2026_11_04.py first")
    wb = openpyxl.load_workbook(R1.BACKUP)
    ws = wb["roi_labels"] if "roi_labels" in wb.sheetnames else wb.worksheets[0]
    head = [R1.cell_text(c.value) for c in ws[1]]
    rows = [[R1.cell_text(c.value) for c in r] for r in ws.iter_rows(min_row=2) if r[0].value is not None]
    i_name, i_gt = head.index("image_name"), head.index("extracted_serial_number")
    orig = {r[i_name]: r[i_gt] for r in rows}

    # ---- round 1
    plan = {}   # name -> (action, new_serial, note, round)
    for prefix, gt in R1.REMOVE:
        m = [n for n in orig if n.startswith(prefix) and orig[n] == gt]
        if len(m) != 1:
            sys.exit(f"round 1: {prefix} {gt} matches {m}")
        plan[m[0]] = ("remove", "", "reviewer note: remove the image and related data", 1)
    for name, gt, action, new, note in R1.REVIEW:
        if orig.get(name) != gt:
            sys.exit(f"round 1: {name} ground truth is {orig.get(name)!r}, expected {gt!r}")
        plan[name] = (action, new, note, 1)

    # ---- round 2
    dec = R2.load_decisions() if os.path.exists(R2.DECISIONS) else {}
    missing = [n for n, *_ in R2.ITEMS if n not in dec]
    if missing or R2.OVERLAP_KEY not in dec:
        print(f"round 2 not finished: {len(missing)} crops undecided"
              f"{' + the overlap question' if R2.OVERLAP_KEY not in dec else ''}:")
        for n in missing:
            print("   ", n)
        sys.exit("open the review page (scripts/review_round2_server.py), decide them, then run this again")
    for name, *_ in R2.ITEMS:
        d = dec[name]
        if d["decision"] == "remove":
            plan[name] = ("remove", "", "review round 2: remove", 2)
        elif norm(d["serial"]) == norm(orig[name]):
            plan[name] = ("keep", orig[name], "review round 2: ground truth confirmed", 2)
        else:
            plan[name] = ("correct", d["serial"], "review round 2: serial chosen on the review page", 2)
    overlap = dec[R2.OVERLAP_KEY]["decision"]

    # ---- round 3 (doubtful labels in the new crops + the 07 rule)
    rule, r3_state = None, "not started"
    if os.path.exists(R3.ITEMS_FILE):
        items3 = list(csv.DictReader(open(R3.ITEMS_FILE, encoding="utf-8")))
        dec3 = R3.load_decisions()
        miss3 = [it["image_name"] for it in items3 if it["image_name"] not in dec3]
        if miss3 or R3.RULE_KEY not in dec3:
            print(f"round 3 not finished: {len(miss3)} of {len(items3)} crops undecided"
                  f"{' + the 07 rule' if R3.RULE_KEY not in dec3 else ''}")
            sys.exit("open the round-3 page (scripts/review_round3_server.py), finish it, then run this again")
        for it in items3:
            n, d = it["image_name"], dec3[it["image_name"]]
            if n not in orig:
                sys.exit(f"round 3: {n} is not in the original ground truth")
            if d["decision"] == "remove":
                plan[n] = ("remove", "", "review round 3: remove", 3)
            elif norm(d["serial"]) == norm(orig[n]):
                plan[n] = ("keep", orig[n], "review round 3: ground truth confirmed", 3)
            else:
                plan[n] = ("correct", d["serial"], "review round 3: serial chosen on the review page", 3)
        rule, r3_state = dec3[R3.RULE_KEY]["decision"], "complete"

    rem = {n for n, p in plan.items() if p[0] == "remove"}
    fix = {n: p[1] for n, p in plan.items() if p[0] == "correct"}
    final = {n: fix.get(n, g) for n, g in orig.items() if n not in rem}
    if rule:                                   # one way of writing every "07" serial
        for n in final:
            new = R3.apply_rule(final[n], rule)
            if new != final[n]:
                plan[n] = ("correct", new, f"07 rule: {R3.RULES[rule]}", 3)
                fix[n] = final[n] = new
    v4 = {r["image_name"]: r for r in csv.DictReader(open(V4_LABELS, encoding="utf-8"))}
    in_v4 = [n for n in final if n in v4]
    clash = [(n, final[n], v4[n]["serial"]) for n in in_v4 if norm(final[n]) != norm(v4[n]["serial"])]
    ds_names = [n for n in sorted(final, key=str.lower) if overlap == "all_flagged" or n not in v4]

    print(f"ground truth: {len(orig)} -> {len(final)}   (removed {len(rem)}, corrected {len(fix)})")
    for n, s in sorted(fix.items()):
        print(f"   correct {n}: {orig[n]} -> {s}")
    print(f"round 2: {sum(p[3] == 2 for p in plan.values())} crops decided; overlap choice: {overlap}")
    print(f"round 3: {r3_state}" + (f"; 07 rule: {R3.RULES[rule]}" if rule else ""))
    print(f"retraining data: {len(ds_names)} crops "
          f"({'only crops v4 never saw' if overlap == 'new_only' else f'all, {len(in_v4)} flagged as in v4_train'})")
    if clash:
        print(f"note: {len(clash)} kept crops have a different serial in datasets/v4_train:")
        for c in clash:
            print(f"   {c[0]}: here {c[1]}, v4_train {c[2]}")
    if a.dry_run:
        print("dry run - nothing changed")
        return

    # ---- ground truth file
    out = openpyxl.Workbook()
    o = out.active
    o.title = ws.title
    o.append(head)
    for r in rows:
        if r[i_name] in rem:
            continue
        r = list(r)
        r[i_gt] = final[r[i_name]]
        o.append(r)
    ch = out.create_sheet("review_changes")
    ch.append(["image_name", "old_ground_truth", "action", "new_ground_truth", "note", "round", "applied"])
    stamp = time.strftime("%Y-%m-%d %H:%M")
    for n, (act, new, note, rnd) in sorted(plan.items(), key=lambda kv: (kv[1][3], kv[1][0], kv[0].lower())):
        ch.append([n, orig[n], {"remove": "removed", "correct": "corrected", "keep": "kept"}[act],
                   new if act != "remove" else "", note, rnd, stamp])
    for sh in (o, ch):
        for c in sh[1]:
            c.font = Font(bold=True)
        for row in sh.iter_rows(min_row=2):
            for c in row:
                c.number_format = "@"
        sh.column_dimensions["A"].width = 52
    ch.column_dimensions["E"].width = 70
    tmp = R1.TRUTH + ".tmp.xlsx"
    out.save(tmp)
    os.replace(tmp, R1.TRUTH)
    print(f"saved {os.path.relpath(R1.TRUTH, ROOT)}")

    # ---- images: clean/ holds exactly the kept crops
    os.makedirs(R1.REMOVED, exist_ok=True)
    moved_out = moved_back = 0
    for n in orig:
        c, r = os.path.join(R1.IMAGES, n), os.path.join(R1.REMOVED, n)
        if n in rem and os.path.exists(c):
            shutil.move(c, r)
            moved_out += 1
        elif n not in rem and not os.path.exists(c) and os.path.exists(r):
            shutil.move(r, c)
            moved_back += 1
    print(f"images: {moved_out} moved to removed/, {moved_back} brought back to clean/")

    # ---- model reads: drop removed crops (brought-back crops are read again by step 1)
    if os.path.exists(R1.READS):
        rb = openpyxl.load_workbook(R1.READS)
        dropped = 0
        for sh in rb.worksheets:
            for i in range(sh.max_row, 1, -1):
                if R1.cell_text(sh.cell(row=i, column=1).value) in rem:
                    sh.delete_rows(i)
                    dropped += 1
        if dropped:
            tmp = R1.READS + ".tmp.xlsx"
            rb.save(tmp)
            os.replace(tmp, R1.READS)
        print(f"model reads: {dropped // max(len(rb.worksheets), 1)} removed crops dropped")

    # ---- retraining dataset (built aside, then swapped in)
    tmpd = OUT_DS + ".tmp"
    shutil.rmtree(tmpd, ignore_errors=True)
    os.makedirs(os.path.join(tmpd, "images"))
    with open(os.path.join(tmpd, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "serial", "in_v4_train", "v4_train_serial"])
        for n in ds_names:
            shutil.copy2(os.path.join(R1.IMAGES, n), os.path.join(tmpd, "images", n))
            w.writerow([n, final[n], v4[n]["split"] if n in v4 else "no", v4[n]["serial"] if n in v4 else ""])
    with open(os.path.join(tmpd, "BUILD_INFO.json"), "w", encoding="utf-8") as f:
        json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "round3": r3_state, "rule07": rule,
                   "overlap": overlap, "images": len(ds_names)}, f, indent=1)
    shutil.rmtree(OUT_DS, ignore_errors=True)
    os.replace(tmpd, OUT_DS)
    print(f"saved {os.path.relpath(OUT_DS, ROOT)}/  ({len(ds_names)} images + labels.csv)")
    print("\nNext (refresh the test numbers):\n"
          "  .venv/bin/python scripts/step1_run_all_models.py\n"
          "  .venv/bin/python scripts/step2_check_accuracy.py\n"
          "  .venv/bin/python scripts/step3_build_dashboard.py")


if __name__ == "__main__":
    main()
