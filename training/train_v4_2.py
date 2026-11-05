#!/usr/bin/env python3
"""
v4.2 FP32 = the v4 recipe re-run on the merged collection (v4's data + the reviewed 4-Nov crops):
same start (v3 weights), same 20 epochs, same LR - only the data is new.  ONE command:

    cd "/home/coral/Desktop/Prince/ocr/Prince number detection in meter"
    python3 training/train_v4_2.py

Before it: datasets/merge_into_v4_train.py must have merged the new crops (label_source nov4_reviewed).
v4 and v3 are not changed by this script (old v4 is removed later, after scoring, by
quantization/quantize_v4_2_ptq.py --remove-old-v4).

If the PC turns off / the run is stopped, run the SAME command again: every finished step is
skipped and training resumes from the last saved epoch. It refuses to run while
models/v4.2/meterocr_v4.2_fp32_paddle/ exists and there is no unfinished run.

What it does (each step is skipped when already done):
  1. data    datasets/v4_train/labels.csv -> PaddleOCR lists in training/output_v4.2/lists/
             train = every train row + the new train crops repeated 3x (so 829 crops count
             next to 18,402 without making the model forget letter serials); val = every val row
             tests: test_old (v3 data), test_new (+ hard), test_nov4 (new crops' test split)
  2. init    start weights = v3 training weights models/v3/meterocr_v3_train_weights.pdparams (as v4)
  3. train   20 epochs, same recipe as v4 (Cosine LR 2e-5 x sqrt(batch/32), biggest batch that
             fits, 16 loader workers, checkpoint every epoch, auto-retry/resume)
  4. export  best model -> models/v4.2/meterocr_v4.2_fp32_paddle/
             + models/v4.2/meterocr_v4.2_train_weights.pdparams (for PTQ) + training/rec_v4.2.yml
  5. eval    v3 vs v4 vs v4.2 per test list -> training/output_v4.2/eval_v3_v4_v4.2.txt

(Try 1 on 2026-11-04 fine-tuned FROM v4 for 8 epochs and came out slightly worse than v4;
its model was deleted on 2026-11-05; config training/rec_v4.2_try1.yml and log are kept.)

Gentle on the PC (settings below):
  - GPU busy ~90%: training runs 0.9 s, pauses 0.1 s, every second (GPU_SHARE)
  - GPU memory capped at 90% of the card (GPU_MEM_SHARE); CPU at low priority (nice 10)
  - one summary line after every epoch (val acc, best, elapsed, time left)
  - resume = from the last FINISHED epoch (~1 min of work lost at most)

Runs with the GPU training env of the older project copy (python 3.10, paddle 3.3.1, CUDA 13):
the script switches to it by itself.
"""
import csv, glob, json, math, os, pickle, random, re, shutil, signal, subprocess, sys, threading, time

# ---------------------------------------------------------------- paths
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # project root
OLD = "/home/coral/Desktop/Prince/Prince number detection in meter/finetune"
TRAIN_PY = f"{OLD}/train_env/bin/python3.10"
REPO = f"{OLD}/PaddleOCR_repo"
BASE_CFG = f"{P}/training/rec_v4.yml"                      # the v4 recipe (architecture, augmentation)
DATA_DIR = f"{P}/datasets/v4_train"                         # images/ + labels.csv (merged)
INIT = f"{P}/models/v3/meterocr_v3_train_weights"           # start = v3 (same start as v4)
V4_W = f"{P}/models/v4/meterocr_v4_train_weights"           # only for the comparison at the end
OUT = f"{P}/training/output_v4.2"                           # work dir (checkpoints, lists, log)
DATA = f"{OUT}/lists"
BEST = f"{OUT}/best_kept/best_accuracy"
CFG = f"{OUT}/rec_v4.2.yml"
STATE = f"{OUT}/run_state.json"
MODELS = f"{P}/models/v4.2"
EXPORT = f"{MODELS}/meterocr_v4.2_fp32_paddle"
TRAIN_WEIGHTS = f"{MODELS}/meterocr_v4.2_train_weights"
CFG_KEEP = f"{P}/training/rec_v4.2.yml"
LOG_KEEP = f"{P}/training/logs/v4.2_train.log"

# ---------------------------------------------------------------- settings
EPOCHS = 20                             # same as v4 (v4 best was epoch 16)
OVERSAMPLE = 3                          # new crops repeated in the train list
NEW_SOURCE = "nov4_reviewed"
BS_STEPS = [128, 96, 64, 48, 32]
BASE_LR, BASE_BS = 2e-5, 32             # v3/v4 recipe; LR scaled by sqrt(bs / 32)
WORKERS = min(16, max(4, (os.cpu_count() or 8) // 2))
MAX_FAILS = 5
GPU_SHARE = 0.90                        # training may use the GPU 90% of the time (1.0 = no pause)
CYCLE_S = 1.0                           # run GPU_SHARE*CYCLE_S seconds, then pause the rest
GPU_MEM_SHARE = 0.90                    # max share of GPU memory paddle may take
NICE = 10                               # CPU priority of training (0 normal, 19 lowest)
SEED = 0

if os.path.realpath(sys.prefix) != os.path.realpath(f"{OLD}/train_env"):
    if not os.path.exists(TRAIN_PY):
        sys.exit(f"!! training env not found: {TRAIN_PY}")
    os.execv(TRAIN_PY, [TRAIN_PY, os.path.abspath(__file__), *sys.argv[1:]])

import yaml  # noqa: E402

OOM = re.compile(r"out of memory|ResourceExhausted|cudaErrorMemoryAllocation|CUDNN_STATUS_ALLOC_FAILED", re.I)


def say(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def cpu_env():
    return dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES="")


def gpu_total_mb():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def gpu_env():
    env = dict(os.environ, PYTHONPATH=REPO, FLAGS_cudnn_exhaustive_search="1",
               FLAGS_cudnn_batchnorm_spatial_persistent="1")
    total = gpu_total_mb()
    if total and GPU_MEM_SHARE < 1:
        env["FLAGS_gpu_memory_limit_mb"] = str(int(total * GPU_MEM_SHARE))
    return env


def low_priority():
    os.nice(NICE)


# ---------------------------------------------------------------- GPU share (pause/continue) + usage meter
class Throttle:
    """Pauses the training process (SIGSTOP) for (1-GPU_SHARE) of every CYCLE_S seconds and
    measures GPU busy % every 2 s. Always leaves the process running (SIGCONT) when stopped."""
    def __init__(self, proc):
        self.proc, self.stop_evt, self.samples = proc, threading.Event(), []
        self.threads = [threading.Thread(target=self._meter, daemon=True)]
        if GPU_SHARE < 1:
            self.threads.append(threading.Thread(target=self._cycle, daemon=True))
        for t in self.threads:
            t.start()

    def _signal(self, sig):
        try:
            os.kill(self.proc.pid, sig); return True
        except ProcessLookupError:
            return False

    def _cycle(self):
        run, pause = CYCLE_S * GPU_SHARE, CYCLE_S * (1 - GPU_SHARE)
        while not self.stop_evt.wait(run) and self.proc.poll() is None:
            if not self._signal(signal.SIGSTOP):
                break
            time.sleep(pause)
            if not self._signal(signal.SIGCONT):
                break
        self._signal(signal.SIGCONT)

    def _meter(self):
        while not self.stop_evt.wait(2):
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
            try:
                u, m, t = [int(x.strip()) for x in r.stdout.strip().splitlines()[0].split(",")]
                self.samples = (self.samples + [(time.time(), u, m, t)])[-60:]
            except Exception:
                pass

    def line(self):
        rec = [x for x in self.samples if time.time() - x[0] <= 60]
        if not rec:
            return ""
        u = sum(x[1] for x in rec) / len(rec)
        return (f"[GPU] busy {u:.0f}% (avg last minute, target {GPU_SHARE * 100:.0f}%)  "
                f"memory {rec[-1][2] / 1024:.1f}/{rec[-1][3] / 1024:.1f} GB (cap {GPU_MEM_SHARE * 100:.0f}%)")

    def close(self):
        self.stop_evt.set()
        for t in self.threads:
            t.join(timeout=5)
        self._signal(signal.SIGCONT)


def already_running():
    me = os.getpid()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except Exception:
            continue
        if any(c.endswith(b"tools/train.py") for c in cmd) and CFG.encode() in cmd:
            return int(d)
    return None


def helper(*args):
    r = subprocess.run([TRAIN_PY, os.path.abspath(__file__), *args], env=cpu_env(), capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


# ================================================================ 0. check
def step_check():
    for f in (TRAIN_PY, f"{REPO}/tools/train.py", f"{REPO}/ppocr/utils/dict/ppocrv5_dict.txt",
              BASE_CFG, INIT + ".pdparams", V4_W + ".pdparams", f"{DATA_DIR}/labels.csv"):
        if not os.path.exists(f):
            sys.exit(f"!! missing: {f}")
    rows = list(csv.DictReader(open(f"{DATA_DIR}/labels.csv", encoding="utf-8")))
    if not any(r["label_source"] == NEW_SOURCE for r in rows):
        sys.exit("!! the new crops are not merged yet - run: .venv/bin/python datasets/merge_into_v4_train.py")
    pid = already_running()
    if pid:
        sys.exit(f"!! a v4.2 training is already running (process {pid}). Wait for it, or stop it with: kill {pid}")
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True)
    say(f"GPU: {r.stdout.strip()}   loader workers: {WORKERS}\n"
        f"       gentle mode: GPU busy ~{GPU_SHARE * 100:.0f}%, GPU memory cap {GPU_MEM_SHARE * 100:.0f}%, CPU priority nice {NICE}")


# ================================================================ 1. data
def step_data():
    if os.path.exists(f"{DATA}/.complete"):
        say("data: label lists ready - skipped"); return
    say("data: datasets/v4_train/labels.csv -> training/output_v4.2/lists/")
    os.makedirs(DATA, exist_ok=True)
    rows = list(csv.DictReader(open(f"{DATA_DIR}/labels.csv", encoding="utf-8")))
    for r in rows:
        if not os.path.isfile(f"{DATA_DIR}/images/{r['image_name']}"):
            sys.exit(f"!! image in labels.csv but not in images/: {r['image_name']}")

    def pick(split, sources):
        return [(f"images/{r['image_name']}", r["serial"]) for r in rows
                if r["split"] == split and r["label_source"] in sources]

    v4_new = ("new_auto_4_way_agreement", "new_human_reviewed")
    every = v4_new + ("v3_data", NEW_SOURCE)
    train = pick("train", v4_new + ("v3_data",)) + pick("train", (NEW_SOURCE,)) * OVERSAMPLE
    random.Random(SEED).shuffle(train)
    files = {"train.txt": train,
             "val.txt": pick("val", every),
             "test_old.txt": pick("test", ("v3_data",)),
             "test_new.txt": pick("test", v4_new),
             "test_new_hard.txt": pick("test", ("new_human_reviewed",)),
             "test_nov4.txt": pick("test", (NEW_SOURCE,))}
    unknown = {r["label_source"] for r in rows} - set(every)
    if unknown:
        sys.exit(f"!! unknown label_source in labels.csv: {unknown}")
    for name, rs in files.items():
        atomic_write(f"{DATA}/{name}", "".join(f"{p}\t{l}\n" for p, l in rs))
    atomic_write(f"{DATA}/.complete", "ok\n")
    nt = {k: len(v) for k, v in files.items()}
    n_new = len(pick("train", (NEW_SOURCE,)))
    say(f"data: train {nt['train.txt']} (incl. {n_new} new crops x{OVERSAMPLE}), val {nt['val.txt']}, "
        f"test_old {nt['test_old.txt']}, test_new {nt['test_new.txt']} (hard {nt['test_new_hard.txt']}), "
        f"test_nov4 {nt['test_nov4.txt']}")


# ================================================================ config
def write_config(first_bs):
    c = yaml.safe_load(open(BASE_CFG, encoding="utf-8"))
    lr = BASE_LR * math.sqrt(first_bs / BASE_BS)
    n_train = sum(1 for _ in open(f"{DATA}/train.txt", encoding="utf-8"))
    iters_epoch = max(50, int(n_train / (first_bs * 0.72)))
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
    atomic_write(CFG, "# generated by training/train_v4_2.py - do not edit, edit the script\n"
                 + yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
    return lr


# ================================================================ checkpoints (same as train_v4.py)
def _ckcheck(prefix, need_opt):
    import paddle
    paddle.set_device("cpu")
    paddle.load(prefix + ".pdparams")
    if need_opt == "1":
        paddle.load(prefix + ".pdopt")
    st = pickle.load(open(prefix + ".states", "rb"), encoding="latin1")
    print("EPOCH", st.get("epoch", 0), "ACC", st.get("best_model_dict", {}).get("acc", -1))


def ck_info(prefix, need_opt=True):
    if not all(os.path.exists(prefix + e) for e in (".pdparams", ".states") + ((".pdopt",) if need_opt else ())):
        return None
    rc, out = helper("_ckcheck", prefix, "1" if need_opt else "0")
    m = re.search(r"EPOCH (\S+) ACC (\S+)", out)
    return (int(m.group(1)), float(m.group(2))) if rc == 0 and m else None


def resume_point():
    cands = [f"{OUT}/latest"] + sorted((p[:-9] for p in glob.glob(f"{OUT}/iter_epoch_*.pdparams")),
                                       key=lambda p: -int(p.rsplit("_", 1)[1]))
    best = None
    for c in cands:
        info = ck_info(c)
        if info and (best is None or info[0] > best[1]):
            best = (c, info[0])
    return best


def keep_best():
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
        say(f"best model so far: val acc {cur[1]:.4f} (epoch {cur[0]}) -> kept in output_v4.2/best_kept/")


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
            say(f"train: starting from v3 weights (v4 recipe)   epochs {EPOCHS}, batch {bs}, lr {lr:.1e}, workers {WORKERS}")
        print("       log: training/output_v4.2/train.log   (Ctrl+C is safe - run the same command to continue)", flush=True)
        proc = subprocess.Popen(cmd, env=gpu_env(), cwd=REPO, stdout=subprocess.PIPE, preexec_fn=low_priority,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        thr = Throttle(proc)
        kept = ck_info(BEST, need_opt=False)
        best = (kept[1], kept[0]) if kept else (-1.0, 0)            # (acc, epoch)
        oom, last_gpu, ep, t_start, ep_start = False, time.time(), done_epoch, time.time(), done_epoch
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                if OOM.search(line):
                    oom = True
                m = re.search(r"epoch: \[(\d+)/\d+\]", line)
                if m:
                    ep = int(m.group(1))
                m = re.search(r"cur metric, acc: ([0-9.]+)", line)
                if m:
                    acc = float(m.group(1))
                    if acc > best[0]:
                        best = (acc, ep)
                    el = time.time() - t_start
                    left = el / max(1, ep - ep_start) * (EPOCHS - ep)
                    print(f"\n==== epoch {ep}/{EPOCHS} finished   val acc {acc * 100:.2f}%   "
                          f"best {best[0] * 100:.2f}% (epoch {best[1]})   elapsed {el / 60:.1f} min   "
                          f"left ~{left / 60:.0f} min   (saved - a restart continues after this epoch)\n", flush=True)
                if time.time() - last_gpu > 60:
                    last_gpu = time.time(); print(thr.line(), flush=True)
            rc = proc.wait()
        except KeyboardInterrupt:
            thr.close()
            proc.terminate(); proc.wait()
            keep_best()
            sys.exit("\nstopped by you - run the same command to continue from the last saved epoch")
        finally:
            thr.close()
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
        say("export: models/v4.2/meterocr_v4.2_fp32_paddle exists - skipped"); return
    keep_best()
    if not ck_info(BEST, need_opt=False):
        sys.exit("!! no valid best model to export")
    os.makedirs(MODELS, exist_ok=True)
    tmp = EXPORT + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    say("export: best model -> models/v4.2/meterocr_v4.2_fp32_paddle/")
    r = subprocess.run([TRAIN_PY, f"{REPO}/tools/export_model.py", "-c", CFG, "-o",
                        f"Global.pretrained_model={BEST}", f"Global.save_inference_dir={tmp}"],
                       env=gpu_env(), cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(f"{tmp}/inference.pdiparams"):
        print((r.stdout + r.stderr)[-3000:]); sys.exit("!! export failed")
    shutil.copy2(BEST + ".pdparams", TRAIN_WEIGHTS + ".pdparams.tmp")
    os.replace(TRAIN_WEIGHTS + ".pdparams.tmp", TRAIN_WEIGHTS + ".pdparams")
    shutil.copy2(CFG, CFG_KEEP)
    if os.path.exists(f"{OUT}/train.log"):
        os.makedirs(os.path.dirname(LOG_KEEP), exist_ok=True)
        shutil.copy2(f"{OUT}/train.log", LOG_KEEP)
    os.replace(tmp, EXPORT)
    print("      ", ", ".join(sorted(os.listdir(EXPORT))), "+ meterocr_v4.2_train_weights.pdparams, training/rec_v4.2.yml")


# ================================================================ 5. eval
def step_eval():
    res = f"{OUT}/eval_v3_v4_v4.2.txt"
    if os.path.exists(res):
        say("eval: already done"); print(open(res).read()); return
    say("eval: v3 vs v4 vs v4.2 (12 short runs)")
    lines = [f"{'test set':<15} {'images':>7} {'v3 acc':>9} {'v4 acc':>9} {'v4.2 acc':>9}"]
    for t in ("test_old", "test_new", "test_new_hard", "test_nov4"):
        n = sum(1 for _ in open(f"{DATA}/{t}.txt", encoding="utf-8"))
        accs = []
        for w in (INIT, V4_W, BEST):
            r = subprocess.run([TRAIN_PY, f"{REPO}/tools/eval.py", "-c", CFG, "-o",
                                f"Global.pretrained_model={w}", f"Eval.dataset.label_file_list=['{DATA}/{t}.txt']"],
                               env=gpu_env(), cwd=REPO, capture_output=True, text=True)
            out = (r.stdout + r.stderr).split("metric eval")[-1]
            m = re.search(r"acc:([0-9.]+)", out)
            accs.append(f"{float(m.group(1)) * 100:.2f}%" if m else "FAILED")
        lines.append(f"{t:<15} {n:>7} {accs[0]:>9} {accs[1]:>9} {accs[2]:>9}")
        print(lines[-1], flush=True)
    lines += ["", "test_old      = old v3 test split (74% serials with letters)",
              "test_new      = v4's new test images (never trained on)",
              "test_new_hard = part of test_new labelled on the review dashboard",
              "test_nov4     = test split of the reviewed 4-Nov crops (never trained on)",
              "acc = exact match, fixed 48x320 resize (training-eval pipeline, not the deploy pipeline)"]
    atomic_write(res, "\n".join(lines) + "\n")


# ================================================================ main
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_ckcheck":
        _ckcheck(sys.argv[2], sys.argv[3]); sys.exit(0)
    if os.path.exists(f"{EXPORT}/inference.pdiparams") and not os.path.exists(OUT):
        sys.exit("v4.2 is already trained and exported to models/v4.2/.\n"
                 "To train it again, move models/v4.2/ somewhere else first.")
    t0 = time.time()
    step_check()
    step_data()
    step_train()
    step_export()
    step_eval()
    say(f"ALL DONE in {(time.time() - t0) / 60:.0f} min.  v4.2 FP32: models/v4.2/meterocr_v4.2_fp32_paddle/\n"
        f"   next: python3 quantization/quantize_v4_2_ptq.py   (v3 and v4 untouched)")
