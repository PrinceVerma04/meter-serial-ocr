#!/usr/bin/env python3
"""
v4 PTQ INT8 + deployment-style scoring of v3 vs v4.  ONE command:

    cd "/home/coral/Desktop/Prince/ocr/Prince number detection in meter"
    python3 quantization/quantize_v4_ptq.py

Run on 2026-11-02 (3 min); its outputs are in models/v4/ and results/2026-11-02_v4/, so a
re-run only re-prints the summary. Delete an output to redo that step.

Same recipe as the deployed v3 PTQ (quantization/quantize_v3_ptq.py):
  1. export   v4 training checkpoint -> Paddle inference, PIR v1 (Paddle 3.0.0 env;
              paddle2onnx can't read the Paddle 3.3.1 format)
  2. onnx     paddle2onnx -> ONNX FP32, constants folded in Python (the space in the
              project path breaks paddle2onnx's own folding), unused NRTR head pruned
  3. int8     static INT8, QDQ, PER-TENSOR (per-channel breaks this model),
              calibrated on 200 random v4 training crops (seed 42)
  4. score    v3 FP32, v3 PTQ (deployed), v4 FP32, v4 PTQ on the held-out test images,
              exactly the way the deployed server reads: deploy/ptq_int8/meterocr.py
              preprocessing (48x320 keep ratio + pad), greedy CTC, 0.95 accept gate, CPU

Outputs (v3 files and deploy/ are never changed):
  models/v4/meterocr_v4_fp32.onnx                          (~84 MB)
  models/v4/meterocr_v4_int8_ptq.onnx                      (~21 MB)  <- deploy candidate
  results/2026-11-02_v4/v3_v4_deploy_scores.csv            one row per test image, 4 models
  results/2026-11-02_v4/v3_v4_deploy_scores_summary.txt

Each step is skipped when its output exists; scoring is resumable (re-run the same
command after a stop / power cut).
"""
import csv, glob, os, random, shutil, subprocess, sys, time

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # project root
OLD = "/home/coral/Desktop/Prince/Prince number detection in meter"
COMPAT_PY = f"{OLD}/size_reduction/compat_env/venv/bin/python"            # paddle 3.0.0 + paddle2onnx + onnx + ort
VENV_PY = f"{P}/.venv/bin/python"                                         # paddle 3.3.1 CPU + ort (scoring)
REPO = f"{OLD}/finetune/PaddleOCR_repo"

CFG = f"{P}/training/rec_v4.yml"                                           # v4 training config (architecture)
CKPT = f"{P}/models/v4/meterocr_v4_train_weights"                          # v4 best (epoch 16)
DATA_DIR = f"{P}/datasets/v4_train"                                        # images/ + labels.csv
MODELS = f"{P}/models/v4"
CONVERTIBLE = f"{MODELS}/_v4_convertible"                                  # temporary, removed after step 2
V4_FP32_ONNX = f"{MODELS}/meterocr_v4_fp32.onnx"
V4_INT8 = f"{MODELS}/meterocr_v4_int8_ptq.onnx"

V3_FP32_DIR = f"{P}/models/v3/meterocr_v3_fp32_paddle"
V4_FP32_DIR = f"{P}/models/v4/meterocr_v4_fp32_paddle"
V3_PTQ = f"{P}/deploy/ptq_int8/model/meterocr_ptq_int8.onnx"               # deployed model (= models/v3/meterocr_v3_int8_ptq.onnx)
DEPLOY_CODE = f"{P}/deploy/ptq_int8"

OUT_CSV = f"{P}/results/2026-11-02_v4/v3_v4_deploy_scores.csv"
OUT_SUM = f"{P}/results/2026-11-02_v4/v3_v4_deploy_scores_summary.txt"

N_CALIB, CALIB_SEED = 200, 42
MODEL_KEYS = [("v3_fp32", "v3 FP32"), ("v3_ptq", "v3 PTQ INT8 (deployed)"),
              ("v4_fp32", "v4 FP32"), ("v4_ptq", "v4 PTQ INT8 (new)")]
SETS = [("test_old", "old v3 test split (74% serials with letters)"),
        ("test_new", "new correct images held out (never trained on)"),
        ("test_new_hard", "part of test_new labelled on the review dashboard")]


def say(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mb(p):
    return os.path.getsize(p) / 1e6 if os.path.exists(p) else 0.0


def run_self(py, *args):
    """run a step of this script in another python env (CPU only); stop on failure"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("PYTHONPATH", None)
    r = subprocess.run([py, os.path.abspath(__file__), *args], env=env, cwd=P)
    if r.returncode != 0:
        sys.exit(f"!! step '{args[0]}' failed - send me the output above")


# ============================================================ checks
def check():
    for f in (COMPAT_PY, VENV_PY, f"{REPO}/tools/export_model.py", CFG, CKPT + ".pdparams",
              f"{V3_FP32_DIR}/inference.json", f"{V4_FP32_DIR}/inference.json", V3_PTQ,
              f"{DEPLOY_CODE}/meterocr.py", f"{DATA_DIR}/labels.csv"):
        if not os.path.exists(f):
            sys.exit(f"!! missing: {f}")


# ============================================================ 1. export (Paddle 3.0.0)
def step_export():
    if os.path.exists(V4_FP32_ONNX) or os.path.exists(f"{CONVERTIBLE}/inference.pdiparams"):
        say("1/4 export: done earlier - skipped"); return
    say("1/4 export: v4 checkpoint -> Paddle inference (PIR v1) with Paddle 3.0.0")
    tmp = CONVERTIBLE + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    env = dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([COMPAT_PY, f"{REPO}/tools/export_model.py", "-c", CFG, "-o",
                        f"Global.pretrained_model={CKPT}", f"Global.save_inference_dir={tmp}",
                        "Global.use_gpu=False"], env=env, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(f"{tmp}/inference.pdiparams"):
        print((r.stdout + r.stderr)[-4000:]); sys.exit("!! export failed - send me the output above")
    head = open(f"{tmp}/inference.json", encoding="utf-8").read(200)
    if '"version":1' not in head.replace(" ", ""):
        sys.exit(f"!! exported PIR version is not 1 (paddle2onnx needs v1): {head[:120]}")
    os.replace(tmp, CONVERTIBLE)
    print(f"       PIR v1 export OK ({mb(CONVERTIBLE + '/inference.pdiparams'):.0f} MB)")


# ============================================================ 2 + 3. ONNX FP32 + INT8 (compat env)
def preprocess(path):
    import cv2, numpy as np
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    rw = min(320, int(np.ceil(48 * w / h)))
    r = cv2.resize(img, (rw, 48)).astype("float32").transpose(2, 0, 1) / 255.0
    r = (r - 0.5) / 0.5
    pad = np.zeros((3, 48, 320), dtype="float32"); pad[:, :, :rw] = r
    return pad[None]


def _onnx():
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if not os.path.exists(V4_FP32_ONNX):
        say("2/4 onnx: paddle2onnx -> ONNX FP32 (a 'constant folding failed' warning is expected)")
        raw = f"{MODELS}/_v4_fp32_raw.onnx"
        # call paddle2onnx through this env's python: the venv's own 'paddle2onnx' launcher
        # still points at the folder the venv was first made in (moved since) -> "not found"
        subprocess.run([sys.executable, "-c",
                        "import sys; from paddle2onnx.command import main; sys.argv[0] = 'paddle2onnx'; sys.exit(main())",
                        "--model_dir", CONVERTIBLE, "--model_filename", "inference.json",
                        "--params_filename", "inference.pdiparams", "--save_file", raw,
                        "--opset_version", "13"], check=True)
        print(f"       raw ONNX {mb(raw):.0f} MB -> folding constants in Python")
        quant_pre_process(raw, V4_FP32_ONNX + ".tmp", skip_symbolic_shape=True)
        os.replace(V4_FP32_ONNX + ".tmp", V4_FP32_ONNX)
        os.remove(raw)
        print(f"       v4 FP32 ONNX {mb(V4_FP32_ONNX):.1f} MB")
    else:
        say("2/4 onnx: meterocr_v4_fp32.onnx exists - skipped")
    shutil.rmtree(CONVERTIBLE, ignore_errors=True)

    if os.path.exists(V4_INT8):
        say("3/4 int8: meterocr_v4_int8_ptq.onnx exists - skipped"); return
    say(f"3/4 int8: static INT8 (QDQ, per-tensor), calibrating on {N_CALIB} random v4 training crops")
    rows = [r for r in csv.DictReader(open(f"{DATA_DIR}/labels.csv", encoding="utf-8")) if r["split"] == "train"]
    random.seed(CALIB_SEED); random.shuffle(rows); rows = rows[:N_CALIB]
    files = [f"{DATA_DIR}/images/{r['image_name']}" for r in rows]
    n_new = sum(r["label_source"] != "v3_data" for r in rows)
    print(f"       calibration crops: {len(files)} ({n_new} new, {len(files) - n_new} old)")
    name = onnx.load(V4_FP32_ONNX).graph.input[0].name

    class Calib(CalibrationDataReader):
        def __init__(self):
            self.i = 0
        def get_next(self):
            while self.i < len(files):
                x = preprocess(files[self.i]); self.i += 1
                if x is not None:
                    return {name: x}
            return None

    tmp = V4_INT8.replace(".onnx", ".tmp.onnx")
    quantize_static(V4_FP32_ONNX, tmp, calibration_data_reader=Calib(),
                    quant_format=QuantFormat.QDQ, per_channel=False,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    os.replace(tmp, V4_INT8)
    print(f"       v4 INT8 ONNX {mb(V4_INT8):.1f} MB   (v3 deployed: {mb(V3_PTQ):.1f} MB)")


# ============================================================ 4. score (project .venv, CPU workers)
_m = {}


def _init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import paddle
    paddle.disable_signal_handler()
    import paddle.inference as pi
    import onnxruntime as ort
    sys.path.insert(0, DEPLOY_CODE)
    from meterocr import MeterOCR

    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    for key, path in (("v3_ptq", V3_PTQ), ("v4_ptq", V4_INT8)):
        o = MeterOCR(model_path=path)
        o.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        o.inp = o.sess.get_inputs()[0].name
        _m[key] = o
    for key, d in (("v3_fp32", V3_FP32_DIR), ("v4_fp32", V4_FP32_DIR)):
        c = pi.Config(f"{d}/inference.json", f"{d}/inference.pdiparams")
        c.disable_gpu(); c.disable_mkldnn(); c.set_cpu_math_library_num_threads(1); c.disable_glog_info()
        pr = pi.create_predictor(c)
        _m[key] = (pr, pr.get_input_handle(pr.get_input_names()[0]), pr.get_output_handle(pr.get_output_names()[0]))


def _norm(s):
    return (s or "").strip().upper().replace(" ", "")


def _score_one(item):
    import cv2
    name, path, tset, hard, label = item
    row = {"image_name": name, "test_set": tset, "dashboard_label": hard, "label": label}
    bgr = cv2.imread(path)
    pre = _m["v3_ptq"]
    for key, _ in MODEL_KEYS:
        if bgr is None:
            text, score = "", 0.0
        else:
            t0 = time.perf_counter()
            if key.endswith("ptq"):
                text, score = _m[key].read(bgr)
            else:
                pr, inp, out = _m[key]
                inp.copy_from_cpu(pre._preprocess(bgr)); pr.run()
                text, score = pre._decode(out.copy_to_cpu())
            row[f"{key}_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        status = "blank" if not text else ("accepted" if score >= 0.95 else "review")
        row[f"{key}_result"] = text
        row[f"{key}_confidence"] = round(score, 4) if text else 0.0
        row[f"{key}_status"] = status
        row[f"{key}_correct"] = "yes" if text and _norm(text) == _norm(label) else "no"
    return row


def _score():
    from multiprocessing import Pool
    fields = ["image_name", "test_set", "dashboard_label", "label"]
    for k, _ in MODEL_KEYS:
        fields += [f"{k}_result", f"{k}_confidence", f"{k}_status", f"{k}_correct", f"{k}_ms"]

    items = []
    for r in csv.DictReader(open(f"{DATA_DIR}/labels.csv", encoding="utf-8")):
        if r["split"] != "test":
            continue
        tset = "test_old" if r["label_source"] == "v3_data" else "test_new"
        hard = "yes" if r["label_source"] == "new_human_reviewed" else "no"
        items.append((r["image_name"], f"{DATA_DIR}/images/{r['image_name']}", tset, hard, r["serial"]))
    done = set()
    if os.path.exists(OUT_CSV):
        done = {r["image_name"] for r in csv.DictReader(open(OUT_CSV, encoding="utf-8"))}
    todo = [it for it in items if it[0] not in done]
    workers = min(12, max(1, (os.cpu_count() or 4) - 4))
    say(f"4/4 score: {len(items)} test images x 4 models, deploy pipeline, CPU   "
        f"done {len(items) - len(todo)}, to run {len(todo)}, workers {workers}")
    if todo:
        new = not os.path.exists(OUT_CSV)
        start = time.time()
        pool = Pool(workers, initializer=_init_worker)
        try:
            with open(OUT_CSV, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                if new:
                    w.writeheader()
                for i, row in enumerate(pool.imap_unordered(_score_one, todo, chunksize=4), 1):
                    w.writerow(row)
                    if i % 100 == 0 or i == len(todo):
                        fh.flush(); os.fsync(fh.fileno())
                        rate = i / (time.time() - start)
                        print(f"       {i}/{len(todo)}  {rate:.1f} img/s  ~{(len(todo) - i) / rate / 60:.1f} min left", flush=True)
            pool.close()
        except KeyboardInterrupt:
            pool.terminate()
            sys.exit("\nstopped - results so far are saved; run the same command to continue")
        finally:
            pool.join()
    _summary()


def _summary():
    rows = list(csv.DictReader(open(OUT_CSV, encoding="utf-8")))
    groups = {"test_old": [r for r in rows if r["test_set"] == "test_old"],
              "test_new": [r for r in rows if r["test_set"] == "test_new"],
              "test_new_hard": [r for r in rows if r["dashboard_label"] == "yes"]}
    L = ["v3 vs v4 - deploy pipeline (deploy/ptq_int8/meterocr.py preprocessing, greedy CTC, 0.95 gate, CPU)",
         f"model sizes: v3 PTQ {mb(V3_PTQ):.1f} MB, v4 PTQ {mb(V4_INT8):.1f} MB, v4 FP32 ONNX {mb(V4_FP32_ONNX):.1f} MB", ""]
    for g, desc in SETS:
        rs = groups[g]; n = len(rs)
        L += [f"{g}  ({n} images) - {desc}",
              f"  {'model':<24}{'right':>14}{'accepted':>10}{'acc. right':>16}{'review':>8}{'blank':>7}"]
        for k, title in MODEL_KEYS:
            right = sum(r[f"{k}_correct"] == "yes" for r in rs)
            acc = [r for r in rs if r[f"{k}_status"] == "accepted"]
            acc_r = sum(r[f"{k}_correct"] == "yes" for r in acc)
            rev = sum(r[f"{k}_status"] == "review" for r in rs)
            blank = sum(r[f"{k}_status"] == "blank" for r in rs)
            L.append(f"  {title:<24}{right:>6} ({100 * right / max(n, 1):5.2f}%){len(acc):>10}"
                     f"{acc_r:>7} ({100 * acc_r / max(len(acc), 1):5.2f}%){rev:>8}{blank:>7}")
        L.append("")
    same = sum(_norm(r["v4_fp32_result"]) == _norm(r["v4_ptq_result"]) for r in rows)
    same3 = sum(_norm(r["v3_fp32_result"]) == _norm(r["v3_ptq_result"]) for r in rows)
    L.append(f"FP32 vs PTQ same read, all {len(rows)} images:  v3 {same3} ({100 * same3 / max(len(rows), 1):.1f}%)   "
             f"v4 {same} ({100 * same / max(len(rows), 1):.1f}%)")
    for k, title in MODEL_KEYS:
        ms = [float(r[f"{k}_ms"]) for r in rows if r.get(f"{k}_ms")]
        if ms:
            L.append(f"  avg time {title:<24} {sum(ms) / len(ms):6.1f} ms/image (1 CPU thread)")
    L += ["", "right = exact match with the label (upper-case, spaces ignored); accepted = confidence >= 0.95;",
          "acc. right = how many accepted reads are right (precision of auto-accept).",
          "test_new_hard is part of test_new. Columns add up: accepted + review + blank = images."]
    text = "\n".join(L)
    with open(OUT_SUM + ".tmp", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    os.replace(OUT_SUM + ".tmp", OUT_SUM)
    print("\n" + text)


# ============================================================ main
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_onnx":
        _onnx(); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "_score":
        _score(); sys.exit(0)
    t0 = time.time()
    check()
    step_export()
    run_self(COMPAT_PY, "_onnx")
    run_self(VENV_PY, "_score")
    say(f"ALL DONE in {(time.time() - t0) / 60:.0f} min.\n"
        f"   v4 PTQ INT8: models/v4/meterocr_v4_int8_ptq.onnx\n"
        f"   scores:      results/2026-11-02_v4/v3_v4_deploy_scores.csv + _summary.txt\n"
        f"   deploy/ptq_int8/ (v3, in production) was not changed")
