# Cleanup Log — deleted files reference

**Date:** 2026-07-16
**Reason:** free disk space by removing regenerable / redundant files.
**Result:** freed ~11 GB (disk 3.2 GB → 14 GB free; project 11 GB → 637 MB).

All deleted items are **regenerable** or **redundant** — nothing unique was lost, and
**nothing on GitHub was affected** (the repo only tracked code + docs + deploy packages).

---

## What was deleted (and how to regenerate it)

| # | Deleted path | Size | What it was | How to regenerate |
|---|---|---|---|---|
| 1 | `size_reduction/compat_env/venv/` | 9.0 GB | Paddle 3.0.0 GPU virtualenv (used for quantization/export) | `bash size_reduction/compat_env/1_setup_venv.sh` (then re-add GPU/onnx libs — see notes below) |
| 2 | `size_reduction/prune_qat/qat_output/` | 1.7 GB | QAT training checkpoints (`best_accuracy.pdparams`, `.pdopt`, `latest.*`, `best_model/`) | `bash size_reduction/prune_qat/2_qat.sh` (retrains QAT). **The deployable QAT model was already exported** to `deploy/qat_int8/` and `size_reduction/prune_qat/qat_inference/`, so this is only needed to re-train. |
| 3 | `finetune/inference/` | 81 MB | OLD v1 model (fine-tuned on 1,144 labels, ~87%) — superseded by `inference_v2` | Re-export from a checkpoint, or use `inference_v2` (the better model) |
| 4 | `finetune/inference_backup_2026-07-13/` | 81 MB | Backup copy of the OLD v1 model | redundant (was a copy of #3) |
| 5 | `finetune/inference_v2_backup_2026-07-13/` | 81 MB | Backup copy of the retrained FP32 model | redundant — `finetune/inference_v2/` (the original) is kept |
| 6 | `results/*.html` | ~3.5 MB | Review pages (`roi_review.html`, `tpddl_review.html`, `model_compare.html`, `clear500_compare.html`) | re-run the `scripts/build_*_page.py` / `size_reduction/build_3model_page.py` on the matching CSV |
| 7 | `__pycache__/` (project dirs) | tiny | Python bytecode cache | auto-regenerated on next run |
| 8 | `MODELS/meterocr_v1_fp32_old_81mb` | link | Dangling symlink (pointed to deleted #3) | n/a |

---

## What was KEPT (deliverables & essentials — NOT deleted)

- `deploy/ptq_int8/` — PTQ INT8 ONNX deploy package (21 MB) ✅
- `deploy/qat_int8/` — QAT INT8 Paddle deploy package (22 MB) ✅
- `finetune/inference_v2/` — the **FP32 original** model (81 MB) ✅
- `size_reduction/compat_env/output/best_accuracy.pdparams` — retrained checkpoint (205 MB, needed to re-quantize) ✅
- `size_reduction/models/server_int8.onnx` — PTQ INT8 model (21 MB) ✅
- `size_reduction/prune_qat/qat_inference/` — QAT INT8 inference model (22 MB) ✅
- All `scripts/`, `size_reduction/*.py|*.sh|*.md`, docs, and `results/*.csv` ✅
- The git repo + GitHub remote (unaffected) ✅

---

## Notes for regenerating the venv (item #1), if you ever re-quantize

The venv wasn't just `1_setup_venv.sh` — it also needed these extra manual steps that were
added during the work (for GPU export + ONNX quantization):

1. `bash size_reduction/compat_env/1_setup_venv.sh` — creates venv + Paddle 3.0.0 (CPU) + repo deps.
2. GPU Paddle (for fast retrain/export):
   `venv/bin/pip install --no-deps paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/`
   (plus the `nvidia-*-cu11` libs it depends on).
3. PaddleSlim (for QAT): `venv/bin/pip install paddleslim==2.6.0`.
4. ONNX tools: `venv/bin/pip install paddle2onnx onnx onnxruntime` (for benchmarking on
   CUDA 11.8, use `onnxruntime-gpu==1.18.1` + set `LD_LIBRARY_PATH` to the venv's nvidia libs).
5. Patched slim tools (for PP-OCRv5 NRTR head) live in the PaddleOCR repo at
   `deploy/slim/quantization/quant_v5.py`, `export_v5.py`, `export_qat_standalone.py`.

Full methodology: `PROJECT_SUMMARY.md` (§2A, §0, progress log) and `size_reduction/compat_env/README.md`.

---

# Cleanup — 2026-10-27

Goal: keep only the trained/deployed v3 models, the datasets they were trained on,
and the scripts needed to run, retrain and re-quantize them.

## Models removed
| Path | What it was |
|---|---|
| `MODELS/meterocr_v2_fp32_original_81mb/` | v2 FP32 (replaced by v3) |
| `size_reduction/models/server_int8.onnx` (+ `MODELS/meterocr_v2_int8_ptq_21mb.onnx` link) | v2 PTQ INT8 |
| `deploy/ptq_py38/` | Python-3.8 package holding the **v2** PTQ model — the v3 version is `deploy/meterocr_ptq_py38_v3.zip` |
| `deploy/qat_int8/` | QAT INT8, built from v2 |
| `size_reduction/compat_env/output/best_accuracy.pdparams/.states` | v2-era training checkpoint (205 MB) |

## Pages, dashboards and scripts removed (~80 files)
- All 15 generated HTML dashboards/review pages and `Some_analysis_part/tpcodl_dashboard_portable.zip`
- `webapp/` and `scripts/web_app.py` (demo apps that needed v2 FP32 + QAT)
- Every dashboard builder (`analysis/*/build_*.py`, `scripts/build_*.py`, `make_labeling_page.py`,
  `roi_review_page.py`, `size_reduction/build_3model_page.py`)
- One-off experiment scripts in `scripts/` (`test_*`, `run_*`, `compare_*`, `score_*`, rescore/fix/save, …)
- Old `size_reduction/` experiments: `compare_3models.py`, `test_compare.py`,
  `quantize_server_paddleslim.py`, `onnx_reader.py`, `paddle_reader.py`, `path_a_server_int8/`,
  `legacy_server/`, `prune_qat/`, `compat_env/{2_retrain.sh,4_quantize.py,setup_and_quantize.sh}`
- `finetune/run.py`, all `__pycache__/`

Results CSVs, logs, labeled image sets and all datasets were kept.

## How to restore
Files that were committed come back with `git checkout 3703b17 -- <path>`
(e.g. `deploy/qat_int8`, `deploy/ptq_py38`, `webapp`, most `scripts/`).
Untracked files (the v2 models, the checkpoint, the Aug–Sep dashboards and scripts) are gone.

## Kept
`MODELS/meterocr_v3_fp32_original_81mb/` · `size_reduction/models/meterocr_v3_{fp32,int8_ptq}.onnx` ·
`deploy/ptq_int8/` · `deploy/meterocr_ptq_py38_v3.zip` · `finetune/pretrained/` · all datasets ·
scripts: `extract_serials`, `infer`, `score_accuracy`, `run_newptq_puvnl`, `prepare_retrain_data`,
`quantize_v3_ptq` + `compat_env/{1_setup_venv,3_export_v1}.sh`, `finetune/retrain_fp32.sh`.

## Data folders removed (2026-10-27, later the same day)
| Folder | Size | What it was |
|---|---|---|
| `23 june analysis/` | 82 MB | 9,970 images (5,929 unique) + v2-era FP32/PTQ prediction CSVs, no labels |
| `25th_26th_july_2026_serialno_puvnl/` | 284 MB | 37,713 PUVNL images (only local copy), no labels (its predictions CSV `puvnl_newptq_results.csv` is also gone) |
| `26788 images for test/` | 325 MB | duplicate of `datasets/final_test/serialno_testing/images` + 18,544-image consensus set (`result_corrected.csv`, the "93.14%" v3 PTQ figure) and model-comparison CSVs |

None were in git — not recoverable locally. The 26,787 test images still exist in
`datasets/final_test/serialno_testing/images`.
| `bpcl/` | 3 MB | 643 BPCL serial crops (only copy) + 5-model prediction CSVs, no labels |
| `Some_analysis_part/` | 163 MB | TPCODL Aug 1–4 analysis: 11,423 images, bot-label CSV, 3-model consensus/comparison CSVs, blur copies, dashboards' leftovers |

The TPCODL manifest (S3 image URL + bot serial for all 11,423 images) was kept as
`datasets/tpcodl_1aug_4aug_2026_manifest.csv`, so the images can be re-downloaded from
the `mrdt` S3 bucket (links were live on 2026-10-27).

## Duplicate datasets and old results removed (2026-10-27)
| Path | What it was |
|---|---|
| `finetune/dataset/` | v2 training set (6,709; 6,706 of its images are in `dataset_retrain`) |
| `retraining dataset with label/` | byte-identical source copy of `finetune/dataset_retrain/` (same 6,733 images + labels) |
| `retraining_data/` | older 6,716-image version of the same set |
| `datasets/eval_1000/` (incl. `score_any_model.py`) | 1,004 verified labels — all already in `dataset_retrain` |
| `datasets/comparison_dev/`, `datasets/screenshots/` | unlabeled dev images + model CSVs; screenshots |
| `datasets/final_test/` old model-result CSV/XLSX | outputs of deleted models (labels `final_results*.xlsx` kept) |
| `results/*.csv` except `ground_truth.csv`, `feedback_tpddl.csv` | old model outputs |

Note: `scripts/prepare_retrain_data.py` read from `retraining dataset with label/`; the dataset it
built (`finetune/dataset_retrain/`) is kept.
| `finetune/ppocrv5_server_rec.yaml`, `rec_finetune.yml`, `eval_test.yml` | PaddleX example template + v1 train/eval configs (not related to the deployed v3 model) |
| `restart training/` | 14 MB | TPCODL Aug review: 2,193 crops, `annotations.csv` (1,222 reviewed), `review_app.py`. The 712 human-verified images + labels were first copied to `datasets/tpcodl_human_verified/` (`ground_truth.csv`: filename, ground_truth); blur flags, 971 unreviewed crops and the review app were removed. |
| `datasets/final_test/final_results_no_blanks.xlsx` | exact copy of `final_results.xlsx` minus its 639 blank rows |

---

# Cleanup and restructure — 2026-11-02

Goal: one clean project — data per version with its ground truth, models per version
(FP32 + PTQ INT8 + training weights), current docs. Nothing tied to v3 (deployed) or v4 was lost.

## Moved (new home)
| From | To |
|---|---|
| `finetune/dataset_retrain/` (images + train/val/test.txt) | `datasets/v3_train/` (images + `labels.csv`: image_name, serial, split) |
| `finetune/dataset_v4/` + `new test 27 october/correct images 30 october/` | `datasets/v4_train/` (23,707 images incl. a copy of v3's 6,733 + `labels.csv` with split and label_source) |
| `results/ground_truth.csv` | `datasets/v1_v2_labels/ground_truth.csv` |
| `new test 27 october/clean data/clean/` + `clean_labels.xlsx` + `manual_review/manual_review_live.xlsx` | `datasets/test/clean_test_oct27/` (images, merged `labels.csv`, `original_labels_with_quality.xlsx`, `manual_review_4744.xlsx`) |
| `datasets/final_test/serialno_testing/images` + `final_results.xlsx` | `datasets/test/final_test/` (images + `labels.csv`, adds `in_v3_training`) |
| `data/tpddl_test/` + `results/feedback_tpddl.csv` | `datasets/test/tpddl/` |
| `datasets/tpcodl_human_verified/` (`ground_truth.csv`) | `datasets/test/tpcodl_human_verified/` (`labels.csv`) |
| `datasets/tpcodl_1aug_4aug_2026_manifest.csv` | `datasets/other/tpcodl_1aug_4aug_2026_s3_manifest.csv` |
| `finetune/pretrained/PP-OCRv5_server_rec_pretrained.pdparams` | `models/base/` |
| `MODELS/meterocr_v3_fp32_original_81mb/`, `size_reduction/models/meterocr_v3_*.onnx`, `finetune/pretrained/v3_fp32_train_init.pdparams` | `models/v3/` (`meterocr_v3_fp32_paddle/`, `meterocr_v3_fp32.onnx`, `meterocr_v3_int8_ptq.onnx`, `meterocr_v3_train_weights.pdparams`) |
| `MODELS/meterocr_v4_fp32/`, `size_reduction/models/meterocr_v4_*.onnx`, `finetune/output_v4/best_kept/best_accuracy.pdparams` | `models/v4/` (same names with v4) |
| `finetune/train_v4.py`, `rec_retrain.yml`, `output_v4/rec_v4.yml`, `output_v4/train.log` | `training/train_v4.py`, `rec_v3.yml`, `rec_v4.yml`, `logs/v4_train.log` |
| `size_reduction/quantize_v3_ptq.py`, `quantize_v4_ptq.py`, README, requirements, `compat_env/{1_setup_venv.sh,README.md}` | `quantization/` |
| `new test 27 october/` result CSVs/summaries + `v3_model_test_results.pptx` | `results/2026-10-27_v3_clean_test/` |
| `finetune/output_v4/` eval + deploy-score files | `results/2026-11-02_v4/` |
| `meter_ocr_interview_qa.zip`, `meter_ocr_interview_report.zip` | `docs/` |
| `scripts/run_newptq_puvnl.py` | `scripts/run_ptq_folder.py` (any folder, `--model v3|v4`) |

All moves were checked: image sets match their label files 1:1; the 16,974 new v4 images and the
6,733 v3 copies match the originals by md5; model checksums unchanged; `score_vs_department.py`
reproduces the saved 27-Oct summary exactly; both pipelines run from the new paths.

## Deleted
| Path | Why |
|---|---|
| `finetune/output_v4/` `latest.*`, `best_accuracy.*`, `best_model/`, `config.yml`, `TRAINING_DONE` (~1.5 GB) | optimizer/duplicate checkpoints; v4 best weights kept in `models/v4/` |
| `new test 27 october/correct images 30 october/` | same 16,974 images as `datasets/v4_train` |
| `new test 27 october/results_dashboard.html`, `crop_preprocessing_preview.html` | dashboards |
| `new test 27 october/manual_review/reviews.db*` | review finished (4,744/4,744); all rows are in `manual_review_4744.xlsx` and `labels.csv` |
| `new test 27 october/clean data/clean_test_labels.xlsx` | subset of `original_labels_with_quality.xlsx` |
| `scripts/review_server.py`, `review_page.html`, `review_watchdog.py`, `build_results_dashboard.py`, `results_dashboard_template.html` | dashboard code |
| `scripts/prepare_retrain_data.py`, `build_correct_set.py`, `compare_filters.py`, `add_filter_slides.py`, `score_accuracy.py` | one-off scripts (results kept) / input files long gone |
| `analysis/`, `ntow_corrected_labels.csv`, `three_model_consensus_tpcodl.csv` | v2/QAT-era reviews of deleted models |
| `PROJECT_REPORT.md`, `SESSION_HANDOFF_2026-07-25.md`, `WORKLOG_2026-07-25_v3_ptq_requant.md` | superseded by `PROJECT_SUMMARY.md` |
| `size_reduction/models_compare.csv`, `compat_env/{output/,server_train.yml,3_export_v1.sh,pip_paddle.log}` | v2-era leftovers |
| `finetune/retrain_fp32.sh`, `finetune/dataset_retrain/*.txt` + `dict.txt` | old-machine paths; labels now in `datasets/v3_train/labels.csv` |
| `MODELS/README.md` | replaced by `models/README.md` |
| `v3_training_data_6733.zip` | same content as `datasets/v3_train/` |
| `__pycache__/`, LibreOffice lock file | junk |

## 2026-11-02 (later) — August v4 brought in, GitHub backup
- GitHub `main` had a commit this machine didn't (`55bb9eb4`, 19 Aug: an earlier "v4" trained from stock on
  12,886 images, `TRAINING_LOG.md`, TPDDL/TPCODL review scripts). Saved before pushing as branch
  `backup/github-main-2026-11-02` and tag `backup-github-main-2026-11-02`; also kept in `main`'s history
  (merged with this layout's files winning).
- That model is named **v4-aug** here. Copied from the old copy (md5-verified, originals untouched):

| From (`/home/coral/Desktop/Prince/Prince number detection in meter/`) | To |
|---|---|
| `finetune/inference_v4/` | `models/v4_aug/meterocr_v4aug_fp32_paddle/` |
| `size_reduction/models/meterocr_v4_fp32.onnx`, `meterocr_v4_int8_ptq.onnx` | `models/v4_aug/meterocr_v4aug_fp32.onnx`, `meterocr_v4aug_int8_ptq.onnx` |
| `finetune/output_retrain_v4/best_accuracy.pdparams` | `models/v4_aug/meterocr_v4aug_train_weights.pdparams` |
| `finetune/rec_retrain_v4.yml`, `output_retrain_v4/train.log` | `training/rec_v4_aug.yml`, `training/logs/v4_aug_train.log` |
| `retraining dataset with label/` (12,886 images + `ground_truth.csv`) | `datasets/v4_aug_train/` (`images/`, `original_ground_truth.csv`, new `labels.csv` from `finetune/dataset_retrain/*.txt`) |


## 2026-11-04 — v4-aug removed
- On request, the August experiment **v4-aug** was removed from this project: `models/v4_aug/` (FP32 Paddle,
  FP32 ONNX, INT8 PTQ ONNX, training weights; 387 MB), `training/rec_v4_aug.yml`, `training/logs/v4_aug_train.log`,
  and its entries in `dashboard/app.py`, `scripts/step1_run_all_models.py`, `step2_check_accuracy.py`,
  `step3_build_dashboard.py`, the READMEs and `results/2026-11-04_all_models_test/` (columns dropped, report and
  dashboard rebuilt).
- `datasets/v4_aug_train/` deleted: its 6,733 v3 images were byte-identical copies of `datasets/v3_train/`
  (same labels); its 6,153 TPDDL July crops were moved to `datasets/test/tpddl_july_2026/` (`image_name, serial`).
- Originals still exist in the old copy `/home/coral/Desktop/Prince/Prince number detection in meter/`
  (`finetune/inference_v4/`, `finetune/output_retrain_v4/`, `size_reduction/models/meterocr_v4_*.onnx`,
  `retraining dataset with label/`) and on GitHub branch `backup/github-main-2026-11-02`.
