#!/usr/bin/env python3
"""
Run FP32 (ONNX) + PTQ INT8 (ONNX) + QAT INT8 (Paddle) over the TPCODL Aug 1-4 set.

Consensus rule (per user):
  - if all THREE models agree (normalized: upper, no spaces) -> final_value = that serial
  - else                                                     -> final_value = FP32 output

Writes a NEW csv (does NOT touch the source manifest):
    columns: image_name, fp32, qat, ptq, all_agree, final_value
"""
import os, csv, glob, argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(ROOT, "Some_analysis_part", "1staug_4thaug_2026_serialno_tpcodl")
FP32 = os.path.join(ROOT, "size_reduction", "models", "meterocr_v3_fp32.onnx")
PTQ  = os.path.join(ROOT, "deploy", "ptq_int8", "model", "meterocr_ptq_int8.onnx")
QAT_M = os.path.join(ROOT, "deploy", "qat_int8", "model", "meterocr_qat_int8.pdmodel")
QAT_P = os.path.join(ROOT, "deploy", "qat_int8", "model", "meterocr_qat_int8.pdiparams")
DICT = os.path.join(ROOT, "deploy", "ptq_int8", "model", "ppocrv5_dict.txt")
IMG_H, IMG_W = 48, 320

VOCAB = ["blank"] + [l.rstrip("\n") for l in open(DICT, encoding="utf-8")] + [" "]

def pre(bgr):
    h, w = bgr.shape[:2]
    rw = min(IMG_W, int(np.ceil(IMG_H * w / h)))
    r = cv2.resize(bgr, (rw, IMG_H)).astype("float32").transpose(2, 0, 1) / 255.0
    r = (r - 0.5) / 0.5
    pad = np.zeros((1, 3, IMG_H, IMG_W), dtype="float32"); pad[0, :, :, :rw] = r
    return pad

def decode(logits):
    p = logits[0]; idx = p.argmax(1); prev = -1; out = []
    for i in idx:
        if i != prev and i != 0:
            out.append(VOCAB[i] if i < len(VOCAB) else "")
        prev = i
    return "".join(out)

# ---------- ONNX worker ----------
_S = _N = None
def _onnx_init(model):
    global _S, _N
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    _S = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    _N = _S.get_inputs()[0].name
def _onnx_work(path):
    b = cv2.imread(path)
    if b is None: return os.path.basename(path), ""
    return os.path.basename(path), decode(_S.run(None, {_N: pre(b)})[0])

# ---------- Paddle (QAT) worker ----------
_P = _IN = _OUT = None
def _pad_init(pdmodel, pdiparams):
    global _P, _IN, _OUT
    import paddle.inference as pi
    cfg = pi.Config(pdmodel, pdiparams); cfg.disable_gpu(); cfg.disable_glog_info()
    cfg.set_cpu_math_library_num_threads(1)
    _P = pi.create_predictor(cfg)
    _IN = _P.get_input_names()[0]; _OUT = _P.get_output_names()[0]
def _pad_work(path):
    b = cv2.imread(path)
    if b is None: return os.path.basename(path), ""
    h = _P.get_input_handle(_IN); h.copy_from_cpu(pre(b)); _P.run()
    return os.path.basename(path), decode(_P.get_output_handle(_OUT).copy_to_cpu())

def run_pool(files, init, args, workers, ctx=None, tag=""):
    res = {}
    kw = dict(max_workers=workers, initializer=init, initargs=args)
    if ctx: kw["mp_context"] = ctx
    with ProcessPoolExecutor(**kw) as ex:
        done = 0
        for name, text in ex.map(_onnx_work if init is _onnx_init else _pad_work,
                                 files, chunksize=16):
            res[name] = text; done += 1
            if done % 2000 == 0: print("  [%s] %d/%d" % (tag, done, len(files)), flush=True)
    print("  [%s] done %d" % (tag, len(files)), flush=True)
    return res

norm = lambda s: s.strip().upper().replace(" ", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "Some_analysis_part",
                    "three_model_consensus_tpcodl.csv"))
    ap.add_argument("--images", default=IMAGES)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.images, "*.jpg")))
    print("images: %d  |  workers: %d" % (len(files), a.workers), flush=True)

    print("== FP32 (ONNX) =="); fp = run_pool(files, _onnx_init, (FP32,), a.workers, tag="fp32")
    print("== PTQ  (ONNX) =="); pq = run_pool(files, _onnx_init, (PTQ,),  a.workers, tag="ptq")
    print("== QAT  (Paddle) =="); qa = run_pool(files, _pad_init, (QAT_M, QAT_P),
                                                max(4, a.workers // 2),
                                                ctx=mp.get_context("spawn"), tag="qat")

    agree_n = 0
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_name", "fp32", "qat", "ptq", "all_agree", "final_value"])
        for name in (os.path.basename(p) for p in files):
            f32, q, p = fp.get(name, ""), qa.get(name, ""), pq.get(name, "")
            alla = norm(f32) == norm(q) == norm(p) and norm(f32) != ""
            if alla: agree_n += 1
            final = f32 if not alla else f32
            w.writerow([name, f32, q, p, "yes" if alla else "no", final])
    print("wrote %s" % a.out)
    print("all-3-agree: %d / %d (%.1f%%)" % (agree_n, len(files), 100.0*agree_n/len(files)))

if __name__ == "__main__":
    main()
