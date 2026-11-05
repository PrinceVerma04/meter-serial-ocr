#!/usr/bin/env python3
"""
Apply the 2026-11-04 manual review of the "new testing data 4 nov" test set.

What it does (safe to re-run; already-applied items are skipped):
  remove  -> image moved from  new testing data 4 nov/clean/  to  new testing data 4 nov/removed/,
             its row deleted from clean_extracted_serials.xlsx and from
             results/2026-11-04_all_models_test/model_reads.xlsx
  correct -> ground truth changed in clean_extracted_serials.xlsx
Every change is listed in a "review_changes" sheet of clean_extracted_serials.xlsx.
The untouched original is kept once as clean_extracted_serials_before_review.xlsx.
Undo: move images back from removed/, copy the _before_review file over the ground truth,
re-run step 1.

After this, re-score and rebuild the dashboard (no model is run):
    .venv/bin/python scripts/step2_check_accuracy.py
    .venv/bin/python scripts/step3_build_dashboard.py

Run from the project root:
    .venv/bin/python scripts/apply_review_2026_11_04.py --dry-run   # show what would change
    .venv/bin/python scripts/apply_review_2026_11_04.py
"""
import argparse
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.path.join(ROOT, "new testing data 4 nov")
IMAGES = os.path.join(TEST, "clean")
REMOVED = os.path.join(TEST, "removed")
TRUTH = os.path.join(TEST, "clean_extracted_serials.xlsx")
BACKUP = os.path.join(TEST, "clean_extracted_serials_before_review.xlsx")
READS = os.path.join(ROOT, "results", "2026-11-04_all_models_test", "model_reads.xlsx")

# (image_name, ground truth it had, action, new ground truth, note)
REVIEW = [
    ("c5cKgbiYQJyaY91jna1gti_null_bigImg_1787472554.jpg", "1725713", "correct", "41725713",
     "ground truth wrong; all 6 models read 41725713 (faint first digit)"),
    ("dvimFBVvT8SMMd6cTWCUMZ_null_bigImg_1787304108.jpg", "137307", "correct", "35137307",
     "ground truth wrong; correct result 35137307 (reviewer). Also listed once as remove - correction kept"),
    ("dkUKKqkGSCe-qNf71FMUi7_null_bigImg_1787044235.jpg", "14388192", "remove", "",
     "ground truth wrong, v3 FP32 / v3 PTQ wrong too"),
]
REMOVE = """
cITLU3jSTkSOBv_0Fkffe1_null_bigImg_1786 3513372
cJdkUuS_REG9LZ1MEgMdnB_null_bigImg_1786 15868
cJz9_py9RYug6pB1P8mVZ__null_bigImg_1787 7740388
cKClmMrkRbe4Rhix3sbQiX_null_bigImg_1786 1768592
cmlGlBi_TxmwnXNaB14d3Q_null_bigImg_1785 1848921
cOnc7zctQgaLgovA7RPrnd_null_bigImg_1787 41665216
cq9Rtm-zSX2UDzG554aGXG_null_bigImg_1786 1685355
cU9jYdkpQda0Xwg2aiPXYd_null_bigImg_1786622480 0151858
cUDdnFioQPGQJdXpVImIPk_null_bigImg_1786 41647347
d_DyzXTOTd-pzZPlxO5rUc_null_bigImg_1787715532 158677
dbVQCHYuSAezJG44Bw__9p_null_bigImg_17869 1798850
dc-KlKAPQ42ybS2sLVZvm__null_bigImg_1788079957 4206557
dEAHPVrpRiyI1RiDG0UkCo_null_bigImg_178 7857569
dEAHPVrpRiyI1RiDG0UkCo_null_bigImg_1787282074 0334064
dfmqlOl_TTymQy_wQzdfTE_null_bigImg_1787 2159127
dhgURubQS8uu9xQVSwEMUK_null_bigImg_1787 520303
dhVQ-jn2Q5aop8xqRJGbGh_null_bigImg_1787995145 7750982
dLziqMQEQtSa5ZpNp_bP98_null_bigImg_178 14228793
dnCqf07fQnq1-PG93Fedkm_null_bigImg_1786 98107073
dnCqf07fQnq1-PG93Fedkm_null_bigImg_1788 4131925
dr08b2M6QbSr3D2InstsAC_null_bigImg_1787372672 111933
dT1OsydXQoqwsBM165mKR7_null_bigImg_1786 136070
dWgDbFhXQHie4QU3HJZwhs_null_bigImg_17881 5113038
dX9E4t0TSVGUBWuHuGtHSI_null_bigImg_1786705989 14027
e2slCLSeTtK8kQSkdsUgJY_null_bigImg_1786 07956
e7wf5JMEQHW1h7j6NdDJLa_null_bigImg_1787541526 417698
e8yoXjhjQ4KlK2zLj3e4DD_null_bigImg_17871 35100434
eAULxvUzTvaGMO-MdmQHE__null_bigImg_1786 414356
ebJerSApQECeQzSSp6KrN4_null_bigImg_1787 12001593
egXxxy6JR0Gbg66Z0f7dpL_null_bigImg_178 1719835
eh4lkKbjRguXRfZbhMkv9w_null_bigImg_178 741651007
ejaFbEYLRH64XRD-2yULAC_null_bigImg_1787 1928817
eKCsUp_6R2inPAnRmcVRH5_null_bigImg_1787 4404720
eljmWbM9QbiP4DYW-bwgPC_null_bigImg_1787 004219
emaqne18Rtm5cbSSZSSYjV_null_bigImg_1786 4101513
eOQa3NK2Rpyzzcokp1M33C_null_bigImg_1786 4107527
ewGNM_p-RzqTvSaQKt9pX2_null_bigImg_1786362502 075514
ewqvzn5hSWKyIwgq_93S7e_null_bigImg_1787288076 4189285
ezQpux9TRbCahY_szPT7PS_null_bigImg_178 400F2986
fEb-rlvrRQimuZMixy1rCn_null_bigImg_1786 0741647116
fhoR8JwQRJ6oZ08p8T9Cen_null_bigImg_178 38650
fjhd2GGXSQ-kAOKQdJwlfG_null_bigImg_1786162745 1872506
fJMNoVlRRCCZaRWlghHXLI_null_bigImg_178 417333
fk3Zr2nhR8O1i6Fa5Oiji5_null_bigImg_1787 173579
fMHJ3dRaQTaPcwqSg_wpR2_null_bigImg_1786 93734
fMHJ3dRaQTaPcwqSg_wpR2_null_bigImg_1788 064830
fp_1odiLQAeP70J85F0eLy_null_bigImg_178 417537
fQ0ISIiJQuGx_ECsVq1SYS_null_bigImg_1786 064850
fqDI_Q-8RAS_3WNLp6YOsi_null_bigImg_1787 9807543
fQeumS7xSZ61ExZe1izozA_null_bigImg_1787811660 1405837
ftw94LtCSCGWdqoCFGrHlc_null_bigImg_1786 032706
fVNE1M6qQCq5NhqlVvNs-7_null_bigImg_1788 419341
fyeCfg4tSQyujSBFSw_wun_null_bigImg_1786 418425
fyeCfg4tSQyujSBFSw_wun_null_bigImg_1787 4140019
fz2DUCWhTRW91IfNRyzKD7_null_bigImg_178 96103602
"""
# reviewer's name prefix + ground truth -> resolved to one exact file name at run time
REMOVE = [l.split() for l in REMOVE.strip().splitlines()]


def cell_text(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def main():
    import openpyxl
    from openpyxl.styles import Font
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = BACKUP if os.path.exists(BACKUP) else TRUTH      # always rebuild from the original
    wb = openpyxl.load_workbook(src)
    ws = wb["roi_labels"] if "roi_labels" in wb.sheetnames else wb.worksheets[0]
    head = [cell_text(c.value) for c in ws[1]]
    rows = [[cell_text(c.value) for c in r] for r in ws.iter_rows(min_row=2) if r[0].value is not None]
    i_name, i_gt = head.index("image_name"), head.index("extracted_serial_number")
    orig = {r[i_name]: r[i_gt] for r in rows}

    plan = []                                   # (name, old, action, new, note)
    errors = []
    for prefix, gt in REMOVE:
        m = [n for n in orig if n.startswith(prefix) and orig[n] == gt]
        if len(m) != 1:
            errors.append(f"remove {prefix} {gt}: matches {m}")
            continue
        plan.append((m[0], gt, "remove", "", "reviewer: remove the image and related data"))
    for name, gt, action, new, note in REVIEW:
        if orig.get(name) != gt:
            errors.append(f"{action} {name}: ground truth is {orig.get(name)!r}, expected {gt!r}")
            continue
        plan.append((name, gt, action, new, note))
    names = [p[0] for p in plan]
    if len(set(names)) != len(names):
        errors.append("an image is listed twice")
    if errors:
        sys.exit("NOT applied - fix these first:\n  " + "\n  ".join(errors))

    rem = {p[0] for p in plan if p[2] == "remove"}
    fix = {p[0]: p[3] for p in plan if p[2] == "correct"}
    print(f"review: remove {len(rem)} images, correct ground truth of {len(fix)} images")
    for n, old, act, new, _ in plan:
        if act == "correct":
            print(f"  correct {n}: {old} -> {new}")
    print(f"ground truth: {len(orig)} rows -> {len(orig) - len(rem)}")
    if a.dry_run:
        print("dry run - nothing changed")
        return

    # 1. backup original ground truth once
    if not os.path.exists(BACKUP):
        shutil.copy2(TRUTH, BACKUP)
        print(f"backup: {os.path.relpath(BACKUP, ROOT)}")

    # 2. move removed images out of the test folder
    os.makedirs(REMOVED, exist_ok=True)
    moved = 0
    for n in sorted(rem):
        s, d = os.path.join(IMAGES, n), os.path.join(REMOVED, n)
        if os.path.exists(s):
            shutil.move(s, d)
            moved += 1
        elif not os.path.exists(d):
            print(f"  warning: {n} not found in clean/ or removed/")
    print(f"moved {moved} images to {os.path.relpath(REMOVED, ROOT)}/ ({len(rem) - moved} were already there)")

    # 3. new ground truth file (+ review_changes sheet)
    out = openpyxl.Workbook()
    o = out.active
    o.title = ws.title
    o.append(head)
    for r in rows:
        if r[i_name] in rem:
            continue
        if r[i_name] in fix:
            r = list(r)
            r[i_gt] = fix[r[i_name]]
        o.append(r)
    for c in o[1]:
        c.font = Font(bold=True)
    for row in o.iter_rows(min_row=2):
        for c in row:
            c.number_format = "@"
    o.column_dimensions["A"].width = 52
    o.column_dimensions["B"].width = 24
    ch = out.create_sheet("review_changes")
    ch.append(["image_name", "old_ground_truth", "action", "new_ground_truth", "note", "applied"])
    stamp = time.strftime("%Y-%m-%d %H:%M")
    for n, old, act, new, note in sorted(plan, key=lambda p: (p[2], p[0].lower())):
        ch.append([n, old, "removed" if act == "remove" else "corrected", new, note, stamp])
    for c in ch[1]:
        c.font = Font(bold=True)
    for row in ch.iter_rows(min_row=2):
        for c in row:
            c.number_format = "@"
    ch.column_dimensions["A"].width = 52
    ch.column_dimensions["E"].width = 70
    tmp = TRUTH + ".tmp.xlsx"
    out.save(tmp)
    os.replace(tmp, TRUTH)
    print(f"ground truth updated: {os.path.relpath(TRUTH, ROOT)}  ({len(orig) - len(rem)} rows + review_changes sheet)")

    # 4. drop removed images from step-1 model reads
    if os.path.exists(READS):
        rb = openpyxl.load_workbook(READS)
        dropped = 0
        for sh in rb.worksheets:
            for i in range(sh.max_row, 1, -1):
                if cell_text(sh.cell(row=i, column=1).value) in rem:
                    sh.delete_rows(i)
                    dropped += 1
        tmp = READS + ".tmp.xlsx"
        rb.save(tmp)
        os.replace(tmp, READS)
        print(f"model reads: dropped {dropped // max(len(rb.worksheets), 1)} images from {os.path.relpath(READS, ROOT)}")

    left = len([f for f in os.listdir(IMAGES)])
    print(f"\ntest set now: {left} images in clean/.  Next:\n"
          "  .venv/bin/python scripts/step2_check_accuracy.py\n"
          "  .venv/bin/python scripts/step3_build_dashboard.py")


if __name__ == "__main__":
    main()
