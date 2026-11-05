#!/usr/bin/env python3
"""
v4.2 PTQ INT8 + deployment-style scoring of v3 / v4 / v4.2.  ONE command (after training/train_v4_2.py):

    cd "/home/coral/Desktop/Prince/ocr/Prince number detection in meter"
    python3 quantization/quantize_v4_2_ptq.py

Same recipe as v4 PTQ (quantization/quantize_v4_ptq.py):
  1. export   v4.2 training weights -> Paddle inference, PIR v1 (Paddle 3.0.0 env)
  2. onnx     paddle2onnx -> ONNX FP32, constants folded in Python
  3. int8     static INT8, QDQ, PER-TENSOR (per-channel breaks this model). Several recipes are built
              and the one with the most right reads on the VALIDATION split (2,648 images - not the
              test sets) becomes meterocr_v4.2_int8_ptq.onnx:
                A every layer INT8, MinMax, 200 crops        (the first recipe, = v3/v4 recipe)
                B CTC head kept FP32, MinMax, 200 crops
                C CTC head FP32, MinMax, 1000 crops
                D CTC head + attention MatMuls FP32, MinMax, 1000 crops
                E CTC head + attention FP32, percentile 99.999, 1000 crops
              (the first INT8 of 5 Nov showed extra trailing characters the FP32 model does not make;
              the CTC head is 120 x 18,385 weights under one INT8 scale.) Calibration runs in chunks of
              50 crops (onnxruntime 1.23's own CalibMaxIntermediateOutputs drops data - not used).
  4. score    v3 FP32, v3 PTQ (deployed), v4 FP32, v4 PTQ, v4.2 FP32, v4.2 PTQ - all ONNX on CPU
              through deploy/ptq_int8/meterocr.py (48x320 keep ratio + pad, greedy CTC, 0.95 gate)
              on: test_old, test_new (+ hard), test_nov4 (datasets/v4_train test split) and
              tpddl_july (datasets/test/tpddl_july_2026, 6,153 crops no model was trained on)

Outputs (v3 files and deploy/ are never changed):
  models/v4.2/meterocr_v4.2_fp32.onnx       (~84 MB)
  models/v4.2/meterocr_v4.2_int8_ptq.onnx   (~21-28 MB, FP32 head adds ~7 MB)  <- deploy candidate
  results/<date>_v4.2/v4.2_ptq_choice.txt             the recipes on the validation split + which was chosen
  results/<date>_v4.2/v4.2_deploy_scores.csv           one row per test image, 6 models
  results/<date>_v4.2/v4.2_deploy_scores_summary.txt
Each step is skipped when its output exists; scoring is resumable.

After reading the summary, remove the old v4 (FP32 + PTQ + weights) with:
    python3 quantization/quantize_v4_2_ptq.py --remove-old-v4
"""
import csv, glob, os, random, shutil, subprocess, sys, time

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = "/home/coral/Desktop/Prince/Prince number detection in meter"
COMPAT_PY = f"{OLD}/size_reduction/compat_env/venv/bin/python"            # paddle 3.0.0 + paddle2onnx + onnx + ort
VENV_PY = f"{P}/.venv/bin/python"                                         # onnxruntime + opencv (scoring)
REPO = f"{OLD}/finetune/PaddleOCR_repo"

CFG = f"{P}/training/rec_v4.2.yml"
CKPT = f"{P}/models/v4.2/meterocr_v4.2_train_weights"
DATA_DIR = f"{P}/datasets/v4_train"
JULY_DIR = f"{P}/datasets/test/tpddl_july_2026"
MODELS = f"{P}/models/v4.2"
CONVERTIBLE = f"{MODELS}/_v4.2_convertible"
FP32_ONNX = f"{MODELS}/meterocr_v4.2_fp32.onnx"
INT8 = f"{MODELS}/meterocr_v4.2_int8_ptq.onnx"
OLD_V4 = f"{P}/models/v4"
DEPLOY_CODE = f"{P}/deploy/ptq_int8"
V3_PTQ = f"{DEPLOY_CODE}/model/meterocr_ptq_int8.onnx"

ONNX = [("v3_fp32", "v3 FP32", f"{P}/models/v3/meterocr_v3_fp32.onnx"),
        ("v3_ptq", "v3 PTQ (deployed)", V3_PTQ),
        ("v4_fp32", "v4 FP32", f"{OLD_V4}/meterocr_v4_fp32.onnx"),
        ("v4_ptq", "v4 PTQ", f"{OLD_V4}/meterocr_v4_int8_ptq.onnx"),
        ("v4.2_fp32", "v4.2 FP32", FP32_ONNX),
        ("v4.2_ptq", "v4.2 PTQ (new)", INT8)]
SETS = [("test_old", "old v3 test split"),
        ("test_old_letters", "  of which serials with letters"),
        ("test_new", "v4's new test images"),
        ("test_new_hard", "  of which human-reviewed (v3 had misread them)"),
        ("test_nov4", "test split of the reviewed 4-Nov crops"),
        ("tpddl_july", "TPDDL July 2026, no model trained on them")]
CALIB_SEED, CALIB_CHUNK = 42, 50
CAND_DIR = f"{MODELS}/_ptq_candidates"
# key, calibration crops, calibration method, layer groups kept in FP32, description
CANDIDATES = [
    ("A_all_int8_200", 200, "minmax", (), "every layer INT8, MinMax, 200 crops (first recipe)"),
    ("B_head_fp32_200", 200, "minmax", ("head",), "CTC head FP32, MinMax, 200 crops"),
    ("C_head_fp32_1000", 1000, "minmax", ("head",), "CTC head FP32, MinMax, 1000 crops"),
    ("D_head_attn_fp32_1000", 1000, "minmax", ("head", "attn"), "CTC head + attention FP32, MinMax, 1000 crops"),
    ("E_head_attn_fp32_pct_1000", 1000, "percentile", ("head", "attn"),
     "CTC head + attention FP32, percentile 99.999, 1000 crops"),
]


def out_dir():
    found = sorted(glob.glob(f"{P}/results/*_v4.2"))
    return found[-1] if found else f"{P}/results/{time.strftime('%Y-%m-%d')}_v4.2"


def choice_file():
    return f"{out_dir()}/v4.2_ptq_choice.txt"


def say(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def mb(p):
    return os.path.getsize(p) / 1e6 if os.path.exists(p) else 0.0


def norm(s):
    return (s or "").strip().upper().replace(" ", "")


def run_self(py, *args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.pop("PYTHONPATH", None)
    r = subprocess.run([py, os.path.abspath(__file__), *args], env=env, cwd=P)
    if r.returncode != 0:
        sys.exit(f"!! step '{args[0]}' failed - send me the output above")


def check():
    for f in (COMPAT_PY, VENV_PY, f"{REPO}/tools/export_model.py", CFG, CKPT + ".pdparams",
              f"{DEPLOY_CODE}/meterocr.py", f"{DATA_DIR}/labels.csv", f"{JULY_DIR}/labels.csv"):
        if not os.path.exists(f):
            sys.exit(f"!! missing: {f}" + ("   (train first: python3 training/train_v4_2.py)" if "v4.2" in f else ""))


# ============================================================ 1. export (Paddle 3.0.0)
def step_export():
    if os.path.exists(FP32_ONNX) or os.path.exists(f"{CONVERTIBLE}/inference.pdiparams"):
        say("1/4 export: done earlier - skipped"); return
    say("1/4 export: v4.2 training weights -> Paddle inference (PIR v1) with Paddle 3.0.0")
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

    if not os.path.exists(FP32_ONNX):
        say("2/4 onnx: paddle2onnx -> ONNX FP32 (a 'constant folding failed' warning is expected)")
        raw = f"{MODELS}/_v4.2_fp32_raw.onnx"
        subprocess.run([sys.executable, "-c",
                        "import sys; from paddle2onnx.command import main; sys.argv[0] = 'paddle2onnx'; sys.exit(main())",
                        "--model_dir", CONVERTIBLE, "--model_filename", "inference.json",
                        "--params_filename", "inference.pdiparams", "--save_file", raw,
                        "--opset_version", "13"], check=True)
        print(f"       raw ONNX {mb(raw):.0f} MB -> folding constants in Python")
        quant_pre_process(raw, FP32_ONNX + ".tmp", skip_symbolic_shape=True)
        os.replace(FP32_ONNX + ".tmp", FP32_ONNX)
        os.remove(raw)
        print(f"       v4.2 FP32 ONNX {mb(FP32_ONNX):.1f} MB")
    else:
        say("2/4 onnx: meterocr_v4.2_fp32.onnx exists - skipped")
    shutil.rmtree(CONVERTIBLE, ignore_errors=True)

    if os.path.exists(INT8):
        say("3/4 int8: meterocr_v4.2_int8_ptq.onnx exists - skipped"); return
    todo = [c for c in CANDIDATES if not os.path.exists(f"{CAND_DIR}/{c[0]}.onnx")]
    say(f"3/4 int8: {len(CANDIDATES)} INT8 recipes (QDQ, per-tensor) - "
        f"{len(CANDIDATES) - len(todo)} built earlier, {len(todo)} to build")
    if not todo:
        return
    from onnxruntime.quantization import CalibrationMethod
    _chunked_calibration()
    os.makedirs(CAND_DIR, exist_ok=True)
    rows = [r for r in csv.DictReader(open(f"{DATA_DIR}/labels.csv", encoding="utf-8")) if r["split"] == "train"]
    random.seed(CALIB_SEED); random.shuffle(rows)                  # same order as the first recipe:
    files_all = [f"{DATA_DIR}/images/{r['image_name']}" for r in rows[:max(c[1] for c in CANDIDATES)]]
    model = onnx.load(FP32_ONNX)
    name = model.graph.input[0].name
    groups = fp32_groups(model)
    print(f"       layers that can stay FP32: CTC head {groups['head']}, attention {groups['attn']}", flush=True)
    if len(groups["head"]) != 1:
        sys.exit(f"!! expected 1 CTC head MatMul, found {groups['head']}")
    for key, n, method, keep, desc in todo:
        say(f"   {key}: {desc}")
        excl = [x for g in keep for x in groups[g]]
        tmp, out = f"{CAND_DIR}/{key}.tmp.onnx", f"{CAND_DIR}/{key}.onnx"
        pct = method == "percentile"
        quantize_static(FP32_ONNX, tmp, calibration_data_reader=Calib(files_all[:n], name),
                        quant_format=QuantFormat.QDQ, per_channel=False,
                        weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                        op_types_to_quantize=["Conv", "MatMul", "Gemm"], nodes_to_exclude=excl,
                        calibrate_method=CalibrationMethod.Percentile if pct else CalibrationMethod.MinMax,
                        extra_options={"CalibPercentile": 99.999} if pct else {})
        os.replace(tmp, out)
        print(f"       {key}: {mb(out):.1f} MB", flush=True)


class Calib:
    """feeds preprocessed calibration crops (same preprocessing as deploy/ptq_int8/meterocr.py)"""
    def __init__(self, files, name):
        self.files, self.name, self.i = files, name, 0

    def get_next(self):
        while self.i < len(self.files):
            x = preprocess(self.files[self.i]); self.i += 1
            if x is not None:
                return {self.name: x}
        return None


def fp32_groups(model):
    """names of the MatMuls that may stay FP32: 'head' = the CTC classifier (weight with >1000 output
    classes), 'attn' = activation x activation MatMuls (attention, no weights - FP32 costs no size)"""
    w = {i.name: list(i.dims) for i in model.graph.initializer}
    head, attn = [], []
    for n in model.graph.node:
        if n.op_type != "MatMul":
            continue
        ws = [x for x in n.input if x in w]
        if ws and w[ws[0]][-1] > 1000:
            head.append(n.name)
        elif not ws:
            attn.append(n.name)
    return {"head": head, "attn": attn}


def _chunked_calibration():
    """make onnxruntime calibrate CALIB_CHUNK crops at a time and merge the ranges / histograms
    (both calibrators merge on every collect_data call), so 1000 crops fit in memory"""
    from onnxruntime.quantization import calibrate as C

    class _List:
        def __init__(self, xs):
            self.xs = iter(xs)

        def get_next(self):
            return next(self.xs, None)

    def wrap(orig):
        def collect(self, reader):
            n = 0
            while True:
                chunk = []
                while len(chunk) < CALIB_CHUNK:
                    x = reader.get_next()
                    if not x:
                        break
                    chunk.append(x)
                if not chunk:
                    break
                orig(self, _List(chunk)); n += len(chunk)
                print(f"       calibrated {n} crops", flush=True)
        collect._chunked = True
        return collect

    for cls in (C.MinMaxCalibrater, C.HistogramCalibrater):
        if not getattr(cls.collect_data, "_chunked", False):
            cls.collect_data = wrap(cls.collect_data)


# ============================================================ 4. score (project .venv)
_R = {}


def _init_worker(models):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import onnxruntime as ort
    sys.path.insert(0, DEPLOY_CODE)
    from meterocr import DICT, MeterOCR
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    chars = [l.rstrip("\n") for l in open(DICT, encoding="utf-8")]
    for key, path in models:
        r = MeterOCR.__new__(MeterOCR)
        r.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        r.inp = r.sess.get_inputs()[0].name
        r.character = ["blank"] + chars + [" "]
        _R[key] = r


def _score_one(item):
    import cv2
    name, path, tset, label = item
    row = {"image_name": name, "test_set": tset, "label": label}
    bgr = cv2.imread(path)
    for key, r in _R.items():
        t0 = time.perf_counter()
        text, score = r.read(bgr) if bgr is not None else ("", 0.0)
        row[f"{key}_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        row[f"{key}_result"] = text
        row[f"{key}_confidence"] = round(score, 4) if text else 0.0
        row[f"{key}_status"] = "blank" if not text else ("accepted" if score >= 0.95 else "review")
        row[f"{key}_correct"] = "yes" if text and norm(text) == norm(label) else "no"
    return row


def _score():
    from multiprocessing import Pool
    models = [(k, p) for k, _, p in ONNX if os.path.exists(p)]
    keys = [k for k, _ in models]
    od = out_dir(); os.makedirs(od, exist_ok=True)
    out_csv = f"{od}/v4.2_deploy_scores.csv"
    fields = ["image_name", "test_set", "label"]
    for k in keys:
        fields += [f"{k}_result", f"{k}_confidence", f"{k}_status", f"{k}_correct", f"{k}_ms"]
    items = []
    for r in csv.DictReader(open(f"{DATA_DIR}/labels.csv", encoding="utf-8")):
        if r["split"] != "test":
            continue
        tset = {"v3_data": "test_old", "nov4_reviewed": "test_nov4"}.get(r["label_source"], "test_new")
        if r["label_source"] == "new_human_reviewed":
            tset = "test_new_hard"
        items.append((r["image_name"], f"{DATA_DIR}/images/{r['image_name']}", tset, r["serial"]))
    for r in csv.DictReader(open(f"{JULY_DIR}/labels.csv", encoding="utf-8")):
        items.append((r["image_name"], f"{JULY_DIR}/images/{r['image_name']}", "tpddl_july", r["serial"]))
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != fields:
                sys.exit(f"!! {out_csv} was made with other models - delete it to score again")
            done = {(r["test_set"], r["image_name"]) for r in rd}
    todo = [it for it in items if (it[2], it[0]) not in done]
    workers = min(16, max(1, (os.cpu_count() or 4) - 4))
    say(f"4/4 score: {len(items)} images x {len(keys)} models ({', '.join(keys)}), CPU   "
        f"done {len(items) - len(todo)}, to run {len(todo)}, workers {workers}")
    if todo:
        new = not os.path.exists(out_csv)
        start = time.time()
        pool = Pool(workers, initializer=_init_worker, initargs=(models,))
        try:
            with open(out_csv, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                if new:
                    w.writeheader()
                for i, row in enumerate(pool.imap_unordered(_score_one, todo, chunksize=4), 1):
                    w.writerow(row)
                    if i % 200 == 0 or i == len(todo):
                        fh.flush(); os.fsync(fh.fileno())
                        rate = i / (time.time() - start)
                        print(f"       {i}/{len(todo)}  {rate:.1f} img/s  ~{(len(todo) - i) / rate / 60:.1f} min left", flush=True)
            pool.close()
        except KeyboardInterrupt:
            pool.terminate()
            sys.exit("\nstopped - results so far are saved; run the same command to continue")
        finally:
            pool.join()
    _summary(out_csv, keys)


def _pick():
    """score every INT8 recipe on the validation split, keep the best as meterocr_v4.2_int8_ptq.onnx"""
    if os.path.exists(INT8):
        say("pick: meterocr_v4.2_int8_ptq.onnx exists - skipped"); return
    from multiprocessing import Pool
    cands = [(c[0], f"{CAND_DIR}/{c[0]}.onnx") for c in CANDIDATES if os.path.exists(f"{CAND_DIR}/{c[0]}.onnx")]
    if not cands:
        sys.exit("!! no INT8 recipe was built - send me the output above")
    items = [(r["image_name"], f"{DATA_DIR}/images/{r['image_name']}", "val", r["serial"])
             for r in csv.DictReader(open(f"{DATA_DIR}/labels.csv", encoding="utf-8")) if r["split"] == "val"]
    workers = min(16, max(1, (os.cpu_count() or 4) - 4))
    say(f"pick: {len(cands)} INT8 recipes + FP32 on {len(items)} validation images "
        f"(the test sets are NOT used to choose), workers {workers}")
    pool = Pool(workers, initializer=_init_worker, initargs=([("fp32", FP32_ONNX)] + cands,))
    try:
        rows = list(pool.imap_unordered(_score_one, items, chunksize=8))
        pool.close()
    except KeyboardInterrupt:
        pool.terminate()
        sys.exit("\nstopped - run the same command again (built recipes are kept)")
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
    text, best = _pick_table(rows, cands)
    shutil.copy2(dict(cands)[best], INT8 + ".tmp")
    os.replace(INT8 + ".tmp", INT8)
    os.makedirs(out_dir(), exist_ok=True)
    with open(choice_file() + ".tmp", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    os.replace(choice_file() + ".tmp", choice_file())
    shutil.rmtree(CAND_DIR)
    print("\n" + text + f"\n\n       -> models/v4.2/meterocr_v4.2_int8_ptq.onnx ({mb(INT8):.1f} MB); other recipes deleted", flush=True)


def _pick_table(rows, cands):
    desc = {c[0]: c[4] for c in CANDIDATES}
    n = len(rows)
    fr = sum(r["fp32_correct"] == "yes" for r in rows)
    st = {}
    for k, p in cands:
        res = [(norm(r[f"{k}_result"]), norm(r["fp32_result"])) for r in rows]
        st[k] = dict(right=sum(r[f"{k}_correct"] == "yes" for r in rows),
                     same=sum(a == b for a, b in res),
                     accw=sum(r[f"{k}_status"] == "accepted" and r[f"{k}_correct"] != "yes" for r in rows),
                     extra=sum(bool(b) and len(a) == len(b) + 1 and a.startswith(b) for a, b in res),
                     size=mb(p))
    best = max(st, key=lambda k: (st[k]["right"], st[k]["same"], -st[k]["accw"]))
    L = [f"chosen: {best} - {desc[best]}",
         "",
         f"v4.2 INT8 recipes on the validation split ({n} images; v4.2 FP32 reads {fr} right = {100 * fr / max(n, 1):.2f}%)",
         f"  {'recipe':<27}{'right':>16}{'same as FP32':>14}{'accepted wrong':>16}{'extra last char':>17}{'size MB':>9}"]
    for k, _ in cands:
        x = st[k]
        L.append(f"  {k:<27}{x['right']:>7} ({100 * x['right'] / max(n, 1):6.2f}%){x['same']:>14}{x['accw']:>16}"
                 f"{x['extra']:>17}{x['size']:>9.1f}" + ("   <- chosen" if k == best else ""))
    L += ["", "  " + "\n  ".join(f"{k}: {desc[k]}" for k, _ in cands),
          "", "chosen = most right reads; ties -> most reads equal to FP32, then fewest accepted-but-wrong.",
          "extra last char = INT8 read is the FP32 read plus one character at the end."]
    return "\n".join(L), best


def _summary(out_csv, keys):
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8")))
    title = {k: t for k, t, _ in ONNX}
    letters = lambda s: any(c.isalpha() for c in s)
    groups = {"test_old": [r for r in rows if r["test_set"] == "test_old"],
              "test_old_letters": [r for r in rows if r["test_set"] == "test_old" and letters(r["label"])],
              "test_new": [r for r in rows if r["test_set"] in ("test_new", "test_new_hard")],
              "test_new_hard": [r for r in rows if r["test_set"] == "test_new_hard"],
              "test_nov4": [r for r in rows if r["test_set"] == "test_nov4"],
              "tpddl_july": [r for r in rows if r["test_set"] == "tpddl_july"]}
    L = ["v3 / v4 / v4.2 - deploy pipeline (deploy/ptq_int8/meterocr.py preprocessing, greedy CTC, 0.95 gate, ONNX CPU)",
         "model sizes: " + ", ".join(f"{title[k]} {mb(p):.1f} MB" for k, _, p in ONNX if k in keys)]
    if os.path.exists(choice_file()):
        L.append("v4.2 PTQ recipe: " + open(choice_file(), encoding="utf-8").readline().strip()[len("chosen: "):]
                 + "   (v4.2_ptq_choice.txt)")
    L.append("")
    for g, desc in SETS:
        rs = groups[g]; n = len(rs)
        if not n:
            continue
        L += [f"{g}  ({n} images) - {desc.strip()}",
              f"  {'model':<20}{'right':>16}{'accepted':>10}{'acc. right':>18}{'acc. wrong':>12}{'review':>8}{'blank':>7}"]
        for k in keys:
            right = sum(r[f"{k}_correct"] == "yes" for r in rs)
            acc = [r for r in rs if r[f"{k}_status"] == "accepted"]
            acc_r = sum(r[f"{k}_correct"] == "yes" for r in acc)
            rev = sum(r[f"{k}_status"] == "review" for r in rs)
            blank = sum(r[f"{k}_status"] == "blank" for r in rs)
            L.append(f"  {title[k]:<20}{right:>7} ({100 * right / n:6.2f}%){len(acc):>10}"
                     f"{acc_r:>8} ({100 * acc_r / max(len(acc), 1):6.2f}%){len(acc) - acc_r:>12}{rev:>8}{blank:>7}")
        L.append("")
    if "v4_ptq" in keys and "v4.2_ptq" in keys:
        for g in ("test_old", "test_nov4", "tpddl_july"):
            rs = groups[g]
            if not rs:
                continue
            a = sum(r["v4_ptq_correct"] == "yes" and r["v4.2_ptq_correct"] == "yes" for r in rs)
            o4 = sum(r["v4_ptq_correct"] == "yes" and r["v4.2_ptq_correct"] != "yes" for r in rs)
            o42 = sum(r["v4_ptq_correct"] != "yes" and r["v4.2_ptq_correct"] == "yes" for r in rs)
            L.append(f"v4 PTQ vs v4.2 PTQ on {g}: both right {a}, only v4 {o4}, only v4.2 {o42}, "
                     f"both wrong {len(rs) - a - o4 - o42}  (total {len(rs)})")
    for f, q in (("v3_fp32", "v3_ptq"), ("v4_fp32", "v4_ptq"), ("v4.2_fp32", "v4.2_ptq")):
        if f in keys and q in keys:
            same = sum(norm(r[f"{f}_result"]) == norm(r[f"{q}_result"]) for r in rows)
            L.append(f"FP32 vs PTQ same read, {f.split('_')[0]:<5} {same} of {len(rows)} ({100 * same / max(len(rows), 1):.1f}%)")
    for k in keys:
        ms = [float(r[f"{k}_ms"]) for r in rows if r.get(f"{k}_ms")]
        if ms:
            L.append(f"  avg time {title[k]:<20} {sum(ms) / len(ms):6.1f} ms/image (1 CPU thread)")
    L += ["", "right = exact match with the label (upper-case, spaces ignored); accepted = confidence >= 0.95;",
          "acc. right % = precision of auto-accept; acc. wrong = accepted but wrong (would go out unchecked).",
          "test_old_letters is part of test_old, test_new_hard part of test_new.",
          "Note: tpddl_july labels were kept only where 3 models agreed with the bot (easy-biased)."]
    text = "\n".join(L)
    out_sum = out_csv[:-4] + "_summary.txt"
    with open(out_sum + ".tmp", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    os.replace(out_sum + ".tmp", out_sum)
    print("\n" + text)


# ============================================================ first INT8 (5 Nov 11:50) -> recipe A
def migrate_first_ptq():
    """The first run built one INT8 (recipe A) and scored it. Keep that model as recipe A and move its
    scores to first_ptq_all_int8/, so this run builds the other recipes, picks one and scores again."""
    if not os.path.exists(INT8) or os.path.exists(choice_file()):
        return
    od = out_dir()
    os.makedirs(CAND_DIR, exist_ok=True)
    os.replace(INT8, f"{CAND_DIR}/{CANDIDATES[0][0]}.onnx")
    keep = f"{od}/first_ptq_all_int8"
    for f in ("v4.2_deploy_scores.csv", "v4.2_deploy_scores_summary.txt"):
        if os.path.exists(f"{od}/{f}"):
            os.makedirs(keep, exist_ok=True)
            os.replace(f"{od}/{f}", f"{keep}/{f}")
    say(f"first v4.2 INT8 (every layer INT8) kept as recipe {CANDIDATES[0][0]}; "
        f"its scores moved to {os.path.relpath(keep, P)}/")


# ============================================================ remove old v4
def remove_old_v4():
    od = out_dir()
    if not glob.glob(f"{od}/v4.2_deploy_scores_summary.txt"):
        sys.exit("!! score v4.2 first (run without --remove-old-v4), read the summary, then remove v4")
    for f in (FP32_ONNX, INT8, f"{MODELS}/meterocr_v4.2_fp32_paddle/inference.pdiparams"):
        if not os.path.exists(f):
            sys.exit(f"!! {f} missing - not removing v4 while v4.2 is incomplete")
    if not os.path.exists(OLD_V4):
        print("old v4 already removed"); return
    files = sorted(os.listdir(OLD_V4))
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OLD_V4) for f in fs) / 1e6
    shutil.rmtree(OLD_V4)
    print(f"removed models/v4/ ({size:.0f} MB): {', '.join(files)}")
    print("v3 (deployed) and v4.2 are untouched. Scores of the old v4 stay in the summary file.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_onnx":
        _onnx(); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "_pick":
        _pick(); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "_score":
        _score(); sys.exit(0)
    if "--remove-old-v4" in sys.argv:
        remove_old_v4(); sys.exit(0)
    t0 = time.time()
    check()
    migrate_first_ptq()
    step_export()
    run_self(COMPAT_PY, "_onnx")
    run_self(VENV_PY, "_pick")
    run_self(VENV_PY, "_score")
    od = os.path.relpath(out_dir(), P)
    say(f"ALL DONE in {(time.time() - t0) / 60:.0f} min.\n"
        f"   v4.2 PTQ INT8: models/v4.2/meterocr_v4.2_int8_ptq.onnx\n"
        f"   recipe choice: {od}/v4.2_ptq_choice.txt\n"
        f"   scores:        {od}/v4.2_deploy_scores.csv + _summary.txt\n"
        f"   send the summary first - only after deciding v4 vs v4.2: --remove-old-v4\n"
        f"   deploy/ptq_int8/ (v3, in production) was not changed")
