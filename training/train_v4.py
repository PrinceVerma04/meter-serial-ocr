#!/usr/bin/env python3
"""
v4 FP32 = v3 FP32 retrained on the new correct images + the v3 data.  ONE command:

    cd "/home/coral/Desktop/Prince/ocr/Prince number detection in meter"
    python3 training/train_v4.py

v4 was trained with this script on 2026-11-02 (22 min, RTX 5080; log: training/logs/v4_train.log).
It refuses to run while models/v4/meterocr_v4_fp32_paddle/ exists, so the trained v4 can't be
overwritten by accident; move that folder away first to train again.

If the PC turns off / the run is stopped, run the SAME command again: every finished
step is skipped and training resumes from the last saved epoch.

What it does (each step is skipped when already done):
  1. data    datasets/v4_train/labels.csv -> PaddleOCR lists in training/output_v4/lists/
             (split column = the exact split v4 used; images in datasets/v4_train/images/)
  2. init    v3 FP32 inference model -> models/v3/meterocr_v3_train_weights.pdparams
             (only saved if it reads exactly like v3; already present)
  3. train   from v3 weights, 20 epochs, biggest batch that fits the GPU (starts at 128,
             steps down automatically on out-of-memory), 16 loader workers, checkpoint
             every epoch, auto-retry/resume after a crash -> training/output_v4/
  4. export  best model -> models/v4/meterocr_v4_fp32_paddle/
  5. eval    v3 vs v4 on test_old / test_new / test_new_hard -> training/output_v4/eval_v3_vs_v4.txt

Runs with the GPU training env of the older project copy (python 3.10, paddle 3.3.1,
CUDA 13): the script switches to it by itself.
"""
import csv, glob, json, math, os, pickle, random, re, shutil, subprocess, sys, time

# ---------------------------------------------------------------- paths
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # project root
OLD = "/home/coral/Desktop/Prince/Prince number detection in meter/finetune"
TRAIN_PY = f"{OLD}/train_env/bin/python3.10"
REPO = f"{OLD}/PaddleOCR_repo"
BASE_CFG = f"{P}/training/rec_v3.yml"                      # the v3 recipe
DATA_DIR = f"{P}/datasets/v4_train"                         # images/ + labels.csv
V3 = f"{P}/models/v3/meterocr_v3_fp32_paddle/inference"
STOCK = f"{P}/models/base/PP-OCRv5_server_rec_pretrained.pdparams"
INIT = f"{P}/models/v3/meterocr_v3_train_weights"
OUT = f"{P}/training/output_v4"                             # work dir (checkpoints, lists, log)
DATA = f"{OUT}/lists"                                       # PaddleOCR label lists made from labels.csv
BEST = f"{OUT}/best_kept/best_accuracy"                     # our own copy of the best model
CFG = f"{OUT}/rec_v4.yml"
STATE = f"{OUT}/run_state.json"
EXPORT = f"{P}/models/v4/meterocr_v4_fp32_paddle"

# ---------------------------------------------------------------- settings
EPOCHS = 20
BS_STEPS = [128, 96, 64, 48, 32]       # first_bs tried in this order (OOM -> next)
BASE_LR, BASE_BS = 2e-5, 32             # v3 recipe; LR scaled by sqrt(bs / 32)
WORKERS = min(16, max(4, (os.cpu_count() or 8) // 2))
MAX_FAILS = 5                           # consecutive failed attempts without progress
SEED = 0

# switch to the training env (works when started with any python3)
if os.path.realpath(sys.prefix) != os.path.realpath(f"{OLD}/train_env"):
    if not os.path.exists(TRAIN_PY):
        sys.exit(f"!! training env not found: {TRAIN_PY}")
    os.execv(TRAIN_PY, [TRAIN_PY, os.path.abspath(__file__), *sys.argv[1:]])

import yaml  # noqa: E402  (available in the training env)

OOM = re.compile(r"out of memory|ResourceExhausted|cudaErrorMemoryAllocation|CUDNN_STATUS_ALLOC_FAILED", re.I)


def say(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def cpu_env():
    e = dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES="")
    return e


def gpu_env():
    return dict(os.environ, PYTHONPATH=REPO,
                FLAGS_cudnn_exhaustive_search="1",
                FLAGS_cudnn_batchnorm_spatial_persistent="1")


def helper(*args):
    """run a paddle helper of this script on CPU in a separate process; returns stdout"""
    r = subprocess.run([TRAIN_PY, os.path.abspath(__file__), *args], env=cpu_env(),
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


# ================================================================ 0. check
def step_check():
    for f in (TRAIN_PY, f"{REPO}/tools/train.py", f"{REPO}/ppocr/utils/dict/ppocrv5_dict.txt",
              BASE_CFG, V3 + ".json", V3 + ".pdiparams", STOCK, f"{DATA_DIR}/labels.csv"):
        if not os.path.exists(f):
            sys.exit(f"!! missing: {f}")
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                        "--format=csv,noheader"], capture_output=True, text=True)
    say(f"GPU: {r.stdout.strip()}   loader workers: {WORKERS}")


# ================================================================ 1. data
def step_data():
    """labels.csv (image_name, serial, split, label_source) -> PaddleOCR lists.
    The split column is the exact split v4 was trained with (new images split 80/10/10
    by capture session, seed 0; v3 data kept its own train/val/test)."""
    if os.path.exists(f"{DATA}/.complete"):
        say("data: label lists ready - skipped"); return
    say("data: datasets/v4_train/labels.csv -> training/output_v4/lists/")
    os.makedirs(DATA, exist_ok=True)
    rows = list(csv.DictReader(open(f"{DATA_DIR}/labels.csv", encoding="utf-8")))
    for r in rows:
        if not os.path.isfile(f"{DATA_DIR}/images/{r['image_name']}"):
            sys.exit(f"!! image in labels.csv but not in images/: {r['image_name']}")

    def pick(split, sources):
        return [(f"images/{r['image_name']}", r["serial"]) for r in rows
                if r["split"] == split and r["label_source"] in sources]

    new_src = ("new_auto_4_way_agreement", "new_human_reviewed")
    every = new_src + ("v3_data",)
    files = {"train.txt": pick("train", every),
             "val.txt": pick("val", every),
             "test_new.txt": pick("test", new_src),
             "test_new_hard.txt": pick("test", ("new_human_reviewed",)),
             "test_old.txt": pick("test", ("v3_data",))}
    random.Random(SEED).shuffle(files["train.txt"])
    for name, rs in files.items():
        atomic_write(f"{DATA}/{name}", "".join(f"{p}\t{l}\n" for p, l in rs))
    atomic_write(f"{DATA}/.complete", "ok\n")
    nt = {k: len(v) for k, v in files.items()}
    say(f"data: train {nt['train.txt']}, val {nt['val.txt']}, test_new {nt['test_new.txt']} "
        f"(hard {nt['test_new_hard.txt']}), test_old {nt['test_old.txt']}")


# ================================================================ config
def write_config(first_bs):
    c = yaml.safe_load(open(BASE_CFG, encoding="utf-8"))
    lr = BASE_LR * math.sqrt(first_bs / BASE_BS)
    n_train = sum(1 for _ in open(f"{DATA}/train.txt", encoding="utf-8"))
    iters_epoch = max(50, int(n_train / (first_bs * 0.72)))   # avg batch over the 3 scales
    g = c["Global"]
    g.update(epoch_num=EPOCHS, save_model_dir=OUT, save_epoch_step=1,
             eval_batch_step=[0, iters_epoch], print_batch_step=20,
             pretrained_model=INIT, checkpoints=None,
             character_dict_path=f"{REPO}/ppocr/utils/dict/ppocrv5_dict.txt",
             save_res_path=f"{OUT}/predicts.txt")
    c["Optimizer"]["lr"]["learning_rate"] = lr
    for part, lf in (("Train", "train.txt"), ("Eval", "val.txt")):
        c[part]["dataset"]["data_dir"] = DATA_DIR
        c[part]["dataset"]["label_file_list"] = [f"{DATA}/{lf}"]
    c["Train"]["sampler"]["first_bs"] = first_bs
    c["Train"]["loader"].update(batch_size_per_card=first_bs, num_workers=WORKERS, use_shared_memory=True)
    c["Eval"]["loader"].update(batch_size_per_card=128, num_workers=min(8, WORKERS))
    os.makedirs(OUT, exist_ok=True)
    atomic_write(CFG, "# generated by training/train_v4.py - do not edit, edit the script\n"
                 + yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
    return lr


# ================================================================ 2. init (v3 -> training weights)
def step_init():
    if os.path.exists(INIT + ".pdparams"):
        say("init: v3 training weights ready - skipped"); return
    say("init: converting v3 FP32 into training weights (CPU, ~1 min)")
    write_config(BS_STEPS[0])
    rc, out = helper("_init")
    print(out.strip()[-3000:])
    if rc != 0 or not os.path.exists(INIT + ".pdparams"):
        sys.exit("!! init failed - send me the output above")


def _init():
    import cv2, numpy as np, paddle
    sys.path.insert(0, REPO)
    from ppocr.data.imaug.rec_img_aug import resize_norm_img
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process
    paddle.set_device("cpu")

    lines = [l.rstrip("\n").split("\t") for l in open(f"{DATA}/test_old.txt", encoding="utf-8") if l.strip()][:64]
    batch = np.stack([resize_norm_img(cv2.imread(f"{DATA_DIR}/{p}"), [3, 48, 320])[0] for p, _ in lines]).astype("float32")

    paddle.enable_static()                                   # v3 inference model
    exe = paddle.static.Executor(paddle.CPUPlace())
    prog, feeds, fetches = paddle.static.load_inference_model(V3, exe)
    scope = paddle.static.global_scope()
    v3 = {p.name: np.array(scope.find_var(p.name).get_tensor()) for p in prog.global_block().all_parameters()}
    ref = np.concatenate([exe.run(prog, feed={feeds[0]: batch[i:i + 16]}, fetch_list=fetches)[0]
                          for i in range(0, len(batch), 16)])
    paddle.disable_static()

    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))       # training model, same as tools/train.py
    post = build_post_process(cfg["PostProcess"], cfg["Global"])
    n = len(post.character)
    cfg["Architecture"]["Head"]["out_channels_list"] = {"CTCLabelDecode": n, "SARLabelDecode": n + 2, "NRTRLabelDecode": n + 3}
    with paddle.utils.unique_name.guard():
        model = build_model(cfg["Architecture"])

    stock = paddle.load(STOCK)
    sd, nv3, nstock, fresh, used = {}, 0, 0, 0, set()
    for k, t in model.state_dict().items():
        shape = list(t.shape)
        if t.name in v3 and list(v3[t.name].shape) == shape:
            sd[k] = paddle.to_tensor(v3[t.name]); nv3 += 1; used.add(t.name)
        elif k in stock and list(stock[k].shape) == shape:
            sd[k] = stock[k]; nstock += 1
        elif k.startswith(("head.gtc_head", "head.before_gtc")):
            sd[k] = t; fresh += 1        # NRTR helper head, different size in stock file: fresh init (as PaddleOCR does)
        else:
            sys.exit(f"!! no weights for {k} {shape}")
    print(f"weights from v3 {nv3}, from stock {nstock}, fresh NRTR helper {fresh}, v3 unused {len(set(v3) - used)}")
    if nv3 != len(v3):
        sys.exit("!! not all v3 weights placed - not saving")
    model.set_state_dict(sd); model.eval()
    with paddle.no_grad():
        out = np.concatenate([model(paddle.to_tensor(batch[i:i + 16])).numpy() for i in range(0, len(batch), 16)])
    a, b = [t for t, _ in post(ref)], [t for t, _ in post(out)]
    same, diff = sum(x == y for x, y in zip(a, b)), float(np.abs(out - ref).max())
    right = sum(x.replace(" ", "") == l for x, (_, l) in zip(b, lines))
    print(f"check on {len(b)} images: same read as v3 {same}/{len(b)}, max prob diff {diff:.1e}, right vs label {right}/{len(b)}")
    if same != len(b) or diff > 1e-3:
        sys.exit("!! rebuilt model does not read like v3 - not saving")
    paddle.save(model.state_dict(), INIT + ".pdparams.tmp")
    os.replace(INIT + ".pdparams.tmp", INIT + ".pdparams")
    print("PASS - saved", INIT + ".pdparams")


# ================================================================ checkpoints
def _ckcheck(prefix, need_opt):
    import paddle
    paddle.set_device("cpu")
    paddle.load(prefix + ".pdparams")
    if need_opt == "1":
        paddle.load(prefix + ".pdopt")
    st = pickle.load(open(prefix + ".states", "rb"), encoding="latin1")
    print("EPOCH", st.get("epoch", 0), "ACC", st.get("best_model_dict", {}).get("acc", -1))


def ck_info(prefix, need_opt=True):
    """(epoch, acc) if the checkpoint loads completely, else None"""
    if not all(os.path.exists(prefix + e) for e in (".pdparams", ".states") + ((".pdopt",) if need_opt else ())):
        return None
    rc, out = helper("_ckcheck", prefix, "1" if need_opt else "0")
    m = re.search(r"EPOCH (\S+) ACC (\S+)", out)
    return (int(m.group(1)), float(m.group(2))) if rc == 0 and m else None


def resume_point():
    """newest checkpoint that loads fully (latest, else iter_epoch_N)"""
    cands = [f"{OUT}/latest"] + sorted(
        (p[:-9] for p in glob.glob(f"{OUT}/iter_epoch_*.pdparams")),
        key=lambda p: -int(p.rsplit("_", 1)[1]))
    best = None
    for c in cands:
        info = ck_info(c)
        if info and (best is None or info[0] > best[1]):
            best = (c, info[0])
    return best                                              # (prefix, epoch) or None


def keep_best():
    """PaddleOCR resets the best score to 0 on resume, so keep our own copy of the best model"""
    cur = ck_info(f"{OUT}/best_accuracy", need_opt=False)
    if not cur:
        return
    kept = ck_info(BEST, need_opt=False)
    if kept is None or cur[1] > kept[1]:
        os.makedirs(os.path.dirname(BEST), exist_ok=True)
        for e in (".pdparams", ".states"):
            shutil.copy2(f"{OUT}/best_accuracy{e}", BEST + e + ".tmp")
        for e in (".pdparams", ".states"):
            os.replace(BEST + e + ".tmp", BEST + e)
        say(f"best model so far: val acc {cur[1]:.4f} (epoch {cur[0]}) -> kept in output_v4/best_kept/")


def prune(keep):
    eps = sorted(glob.glob(f"{OUT}/iter_epoch_*.pdparams"), key=lambda p: int(p[:-9].rsplit("_", 1)[1]))
    for p in eps[:max(0, len(eps) - keep)]:
        for f in glob.glob(p[:-9] + ".*"):
            os.remove(f)


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {"bs_index": 0}


def gpu_line():
    r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        u, m, t = [x.strip() for x in r.stdout.strip().split(",")]
        return f"[GPU] busy {u}%  memory {int(m)/1024:.1f}/{int(t)/1024:.1f} GB"
    except Exception:
        return ""


# ================================================================ 3. train
def step_train():
    if os.path.exists(f"{OUT}/TRAINING_DONE"):
        say("train: already finished - skipped"); return
    st, fails = load_state(), 0
    while True:
        bs = BS_STEPS[st["bs_index"]]
        lr = write_config(bs)
        keep_best(); prune(keep=2)
        rp = resume_point()
        done_epoch = rp[1] if rp else 0
        if done_epoch >= EPOCHS:
            break
        cmd = [TRAIN_PY, f"{REPO}/tools/train.py", "-c", CFG]
        if rp:
            cmd += ["-o", f"Global.checkpoints={rp[0]}"]
            say(f"train: RESUMING after epoch {done_epoch}/{EPOCHS} from {os.path.basename(rp[0])}   batch {bs}, lr {lr:.1e}")
        else:
            say(f"train: starting from v3 weights   epochs {EPOCHS}, batch {bs}, lr {lr:.1e}, workers {WORKERS}")
        print("       log: training/output_v4/train.log   (Ctrl+C is safe - run the same command to continue)", flush=True)

        proc = subprocess.Popen(cmd, env=gpu_env(), cwd=REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        oom, last_gpu = False, 0.0
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                if OOM.search(line):
                    oom = True
                if time.time() - last_gpu > 60:
                    last_gpu = time.time(); print(gpu_line(), flush=True)
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate(); proc.wait()
            keep_best()
            sys.exit("\nstopped by you - run the same command to continue from the last saved epoch")

        rp2 = resume_point()
        progress = (rp2[1] if rp2 else 0) > done_epoch
        if rc == 0 and rp2 and rp2[1] >= EPOCHS:
            break
        fails = 0 if progress else fails + 1
        if oom and not progress and st["bs_index"] < len(BS_STEPS) - 1:
            st["bs_index"] += 1
            atomic_write(STATE, json.dumps(st))
            say(f"GPU out of memory at batch {bs} -> trying batch {BS_STEPS[st['bs_index']]}")
            fails = 0
            continue
        if fails >= MAX_FAILS:
            sys.exit(f"!! training failed {fails} times in a row without progress - send me the last lines above")
        say(f"training process stopped (code {rc}) - retrying from the last saved epoch in 20 s")
        time.sleep(20)

    keep_best(); prune(keep=0)
    atomic_write(f"{OUT}/TRAINING_DONE", time.strftime("%Y-%m-%d %H:%M:%S\n"))
    say("train: finished all epochs")


# ================================================================ 4. export
def step_export():
    if os.path.exists(f"{EXPORT}/inference.pdiparams"):
        say("export: models/v4/meterocr_v4_fp32_paddle exists - skipped"); return
    keep_best()
    if not ck_info(BEST, need_opt=False):
        sys.exit("!! no valid best model to export")
    tmp = EXPORT + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    say("export: best model -> models/v4/meterocr_v4_fp32_paddle/")
    r = subprocess.run([TRAIN_PY, f"{REPO}/tools/export_model.py", "-c", CFG, "-o",
                        f"Global.pretrained_model={BEST}", f"Global.save_inference_dir={tmp}"],
                       env=gpu_env(), cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(f"{tmp}/inference.pdiparams"):
        print((r.stdout + r.stderr)[-3000:]); sys.exit("!! export failed")
    os.replace(tmp, EXPORT)
    print("      ", ", ".join(sorted(os.listdir(EXPORT))))


# ================================================================ 5. eval
def step_eval():
    res = f"{OUT}/eval_v3_vs_v4.txt"
    if os.path.exists(res):
        say("eval: already done"); print(open(res).read()); return
    say("eval: v3 vs v4 (6 short runs)")
    lines = [f"{'test set':<15} {'images':>7} {'v3 acc':>9} {'v4 acc':>9}"]
    for t in ("test_old", "test_new", "test_new_hard"):
        n = sum(1 for _ in open(f"{DATA}/{t}.txt", encoding="utf-8"))
        accs = []
        for w in (INIT, BEST):
            r = subprocess.run([TRAIN_PY, f"{REPO}/tools/eval.py", "-c", CFG, "-o",
                                f"Global.pretrained_model={w}", f"Eval.dataset.label_file_list=['{DATA}/{t}.txt']"],
                               env=gpu_env(), cwd=REPO, capture_output=True, text=True)
            out = (r.stdout + r.stderr).split("metric eval")[-1]
            m = re.search(r"acc:([0-9.]+)", out)
            accs.append(f"{float(m.group(1)) * 100:.2f}%" if m else "FAILED")
        lines.append(f"{t:<15} {n:>7} {accs[0]:>9} {accs[1]:>9}")
        print(lines[-1], flush=True)
    lines += ["", "test_old      = old v3 test split (74% serials with letters)",
              "test_new      = new correct images held out (never trained on)",
              "test_new_hard = part of test_new labelled on the review dashboard (v3 FP32 or PTQ read them wrong)",
              "acc = exact match, fixed 48x320 resize (training-eval pipeline, not the deploy pipeline)"]
    atomic_write(res, "\n".join(lines) + "\n")
    print("\n".join([lines[0]] + lines[1:4]))


# ================================================================ main
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_init":
        _init(); sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "_ckcheck":
        _ckcheck(sys.argv[2], sys.argv[3]); sys.exit(0)
    if os.path.exists(f"{EXPORT}/inference.pdiparams") and not os.path.exists(OUT):
        sys.exit("v4 is already trained and exported to models/v4/ (2026-11-02).\n"
                 "To train it again, move models/v4/meterocr_v4_fp32_paddle/ somewhere else first.")
    t0 = time.time()
    step_check()
    step_data()
    step_init()
    step_train()
    step_export()
    step_eval()
    say(f"ALL DONE in {(time.time() - t0) / 60:.0f} min.  v4 FP32: models/v4/meterocr_v4_fp32_paddle/   "
        f"(v3 untouched: models/v3/)")
