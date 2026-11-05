# Datasets — which model version was trained and tested on what

Every dataset folder is `images/` + `labels.csv` (plus, where useful, the original label files).
Image names look like `<capture-session>_null_bigImg_<timestamp>.jpg`; each image is one
cropped meter serial (recognition only, no detector).

## Training data per version

| Version | Trained | Folder | Images | Split train / val / test | Ground truth |
|---|---|---|---|---|---|
| v1 | Jul 2026 | — (split lists not kept) | 1,144 | 773 / 175 / 196 | first hand-labelled TPWODL crops; an early subset of `v1_v2_labels/ground_truth.csv` |
| v2 | Jul 2026 | — (folder removed 2026-10-27) | 6,709 | 70 / 15 / 15 % by session | readable rows of `v1_v2_labels/ground_truth.csv`; 6,706 of its images are in `v3_train/` |
| **v3** | 2026-07-24 | **`v3_train/`** | **6,733** | 4,793 / 916 / 1,024 | `labels.csv` = the exact labels v3 trained on (74% contain letters) |
| **v4** | 2026-11-02 | **`v4_train/`** | **23,707** | 18,402 / 2,601 / 2,704 | `labels.csv`, see below |

### `v3_train/labels.csv`
`image_name, serial, split` — 6,733 rows. Characters 0–9 and A–X; 959 labels contain a space.

### `v4_train/labels.csv`
`image_name, serial, split, label_source` — 23,707 rows. `split` is the exact split v4 used.

| label_source | Images | Where the serial comes from | train / val / test |
|---|---|---|---|
| `v3_data` | 6,733 | the v3 training labels (same images as `v3_train/`) | 4,793 / 916 / 1,024 |
| `new_human_reviewed` | 1,807 | typed/confirmed by Prince Verma, Sathwik, Srinath Battu on the review dashboard (1,771 department serial confirmed, 36 corrected); images v3 FP32 or PTQ read wrong | 1,468 / 168 / 171 |
| `new_auto_4_way_agreement` | 15,167 | department serial, used only where EasyOCR, v3 FP32 and v3 PTQ all read exactly it with confidence ≥ 0.95; **not checked by a person per image** | 12,141 / 1,517 / 1,509 |
| **total** | **23,707** | | **18,402 / 2,601 / 2,704** |

New images (the two `new_*` sources, 16,974) come from `test/clean_test_oct27/` and were split
80/10/10 **by capture session** (seed 0), so one session — and every repeat of a serial — sits
in only one split. The 2,937 images reviewers marked "can't read" are not used.

The v4 test split is reported as three sets (`results/2026-11-02_v4/`):
`test_old` = 1,024 `v3_data` test images · `test_new` = 1,680 new test images ·
`test_new_hard` = the 171 human-reviewed ones inside `test_new`.

### Adding reviewed crops for v4.2 (`merge_into_v4_train.py`)
The 4-Nov clean crops (`new testing data 4 nov/`) are reviewed in three rounds
(`scripts/apply_review_2026_11_04.py`, `scripts/review_round2_server.py`, `scripts/review_round3_server.py`),
built into `datasets/v4_retrain_nov4/` by `scripts/build_v4_retrain_data.py`, and then merged here by
`datasets/merge_into_v4_train.py`:
- new rows get `label_source = nov4_reviewed`; each crop takes its meter's existing split, crops of new
  meters are split 80/10/10 by meter (seed 0);
- rows already here whose reviewed serial differs get the reviewed serial, the old one stays in `v4_serial`;
- `labels_v4.csv` = this file exactly as v4 was trained on it (kept once, before the merge).
v4.2 (`training/train_v4_2.py`) trains on every row; v4 used every row except `nov4_reviewed`
(with the `v4_serial` values).

### `v1_v2_labels/ground_truth.csv`
The original label bank: `image, true_label, verdict, tier` — 8,434 rows
(correct 6,191 + fixed 521 = 6,712 readable, unreadable 1,722). 6,726 of these images are in
`v3_train/images/`. Kept as the record of v1/v2 labels; the v1/v2 models were deleted.

## Test sets (`test/`)

| Folder | Images | In v3 training | In v4 training | Ground truth (`labels.csv`) |
|---|---|---|---|---|
| `clean_test_oct27/` | 21,042 | 0 | **16,974** | department serial + human review (below) |
| `final_test/` | 26,787 | 3,014 | 3,014 | `serial` (639 blank); `in_v3_training` flag |
| `tpddl/` | 3,460 | 0 | 0 | `true_label` + `verdict` (correct 2,862, fixed 138, unreadable 460) — second utility |
| `tpcodl_human_verified/` | 712 | 0 | 0 | human-verified TPCODL serials (Aug 2026 review) |
| `tpddl_july_2026/` | 6,153 | 0 | 0 | `serial`, hand-reviewed TPDDL July crops (below) |

### `clean_test_oct27/` (built 2026-10-27, reviewed 2026-10-30)
Clear, sharp crops (tenengrad ≥ 1,500) where EasyOCR read the department serial exactly; all
of them were misread by the department's old OCR. `labels.csv` columns:

| column | meaning |
|---|---|
| `ground_truth` | best label: the human serial where reviewed, else the department serial, empty if unreadable |
| `department_serial` | `aug_departgiven` from the department |
| `human_review` | `not_reviewed` 16,298 · `confirmed` 1,771 · `corrected` 36 · `unreadable` 2,937 |
| `human_serial`, `reviewer` | what the reviewer saved (4,744 reviewed = every image v3 FP32 or PTQ got wrong) |
| `old_system_read` | `aug_extracted_serial`, the department's old OCR (wrong on every row by construction) |
| `used_in_v4_training` | `train` 13,609 · `val` 1,685 · `test` 1,680 · `no` 4,068 |

Also here: `manual_review_4744.xlsx` (the raw review log: who, when, edits) and
`original_labels_with_quality.xlsx` (the build file with sharpness / contrast / glare columns).
**Because 16,974 of these images trained v4, this set is no longer a fair test for v4** — use
the v4 test split above, `tpddl/` or `tpcodl_human_verified/`.

### `tpddl_july_2026/` (kept 2026-11-04)
`image_name, serial` — 6,153 TPDDL "1st–4th July 2026" crops, hand-reviewed 17–19 Aug on local review
tools and kept where 3 ONNX models agreed with the bot's serial, so the labels are easy-biased. They
were used only by an August experiment that has since been removed; v3 and v4 never saw them.
Not included in August: 655 TPDDL images never finished review, and 728 TPCODL Aug labels that were
deleted before merging (both lost).

## Other
`other/tpcodl_1aug_4aug_2026_s3_manifest.csv` — S3 URL + bot serial for 11,423 TPCODL crops
(images not stored here; links were live on 2026-10-27).
