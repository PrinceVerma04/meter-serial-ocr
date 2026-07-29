#!/usr/bin/env python3
"""
Test the RETRAINED FP32 model (finetune/inference_v3) + the PTQ INT8 ONNX model
on the 23-June unseen dataset. QAT was removed, so only these two run.

Dataset (no ground truth): 23 june analysis/merged_unseen_images/  (~9970 crops)

Models:
    FP32 : finetune/inference_v3/            (Paddle, retrained, GPU)
    PTQ  : deploy/ptq_int8/model/meterocr_ptq_int8.onnx  (ONNX INT8, CPU)

Outputs (under "23 june analysis/"):
    fp32_results.csv        filename,label
    ptq_results.csv         filename,label
    two_model_results.csv   filename,fp32_result,ptq_result

Resumable: re-running continues from what's already in two_model_results.csv.
"""
import os, glob, csv, re, time
import numpy as np
for _a in ("float", "int", "bool", "object", "str"):
    if not hasattr(np, _a):
        setattr(np, _a, getattr(__builtins__, _a, None) if not isinstance(__builtins__, dict)
                else __builtins__.get(_a))
import cv2
import paddle.inference as pi
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

IMGDIR = "23 june analysis/merged_unseen_images"
OUTDIR = "23 june analysis"
FP32_MODEL = "finetune/inference_v3/inference.json"
FP32_PARAM = "finetune/inference_v3/inference.pdiparams"
PTQ_ONNX = "deploy/ptq_int8/model/meterocr_ptq_int8.onnx"
DICT = "deploy/ptq_int8/model/ppocrv5_dict.txt"

IMG_H, IMG_W = 48, 320
CHARS = ["blank"] + [l.rstrip("\n") for l in open(DICT, encoding="utf-8")] + [" "]


def preprocess(bgr):
    h, w = bgr.shape[:2]
    rw = min(IMG_W, max(1, int(np.ceil(IMG_H * w / h))))
    r = cv2.resize(bgr, (rw, IMG_H)).astype("float32").transpose(2, 0, 1) / 255.0
    r = (r - 0.5) / 0.5
    pad = np.zeros((1, 3, IMG_H, IMG_W), "float32")
    pad[0, :, :, :rw] = r
    return pad


def decode(logits):
    idx = logits[0].argmax(1)
    out, prev = [], -1
    for i in idx:
        if i != prev and i != 0:
            out.append(CHARS[i] if i < len(CHARS) else "")
        prev = i
    return "".join(out)


def clean(text):
    return re.sub(r"[^A-Z0-9]+", " ", (text or "").upper()).strip()


class PaddleRec:
    def __init__(self, model, param):
        cfg = pi.Config(model, param)
        cfg.enable_use_gpu(1000, 0)
        cfg.disable_glog_info()
        self.p = pi.create_predictor(cfg)
        self.i = self.p.get_input_names()[0]
        self.o = self.p.get_output_names()[0]
    def read(self, bgr):
        h = self.p.get_input_handle(self.i); h.copy_from_cpu(preprocess(bgr)); self.p.run()
        return clean(decode(self.p.get_output_handle(self.o).copy_to_cpu()))


class OnnxRec:
    def __init__(self, path):
        self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.n = self.s.get_inputs()[0].name
    def read(self, bgr):
        return clean(decode(self.s.run(None, {self.n: preprocess(bgr)})[0]))


def load_done(path):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) == 3 and row[0] not in ("", "filename"):
                done[row[0]] = (row[1], row[2])
    return done


def main():
    files = sorted(glob.glob(os.path.join(IMGDIR, "*.jpg")))
    print(f"{len(files)} images in {IMGDIR}")

    combined = os.path.join(OUTDIR, "two_model_results.csv")
    done = load_done(combined)
    print(f"resuming: {len(done)} already done")

    print("loading FP32 (retrained, GPU) ...")
    fp32 = PaddleRec(FP32_MODEL, FP32_PARAM)
    print("loading PTQ (ONNX, CPU) ...")
    ptq = OnnxRec(PTQ_ONNX)

    results = dict(done)
    t0 = time.time()
    for n, path in enumerate(files, 1):
        name = os.path.basename(path)
        if name in results:
            continue
        bgr = cv2.imread(path)
        if bgr is None:
            results[name] = ("", "")
        else:
            results[name] = (fp32.read(bgr), ptq.read(bgr))
        if n % 250 == 0 or n == len(files):
            el = time.time() - t0
            print(f"  {n}/{len(files)}  ({el:.0f}s, {n/el:.1f} img/s)", flush=True)

    # write all three CSVs fresh, in filename order
    order = [os.path.basename(p) for p in files]
    with open(combined, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["filename", "fp32_result", "ptq_result"])
        for name in order:
            f, p = results.get(name, ("", "")); w.writerow([name, f, p])
    for col, out in ((0, "fp32_results.csv"), (1, "ptq_results.csv")):
        with open(os.path.join(OUTDIR, out), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["filename", "label"])
            for name in order:
                w.writerow([name, results.get(name, ("", ""))[col]])

    # quick summary
    agree = sum(1 for n in order if results[n][0] == results[n][1] and results[n][0])
    fp32_blank = sum(1 for n in order if not results[n][0])
    ptq_blank = sum(1 for n in order if not results[n][1])
    print(f"\nDONE {len(order)} images in {time.time()-t0:.0f}s")
    print(f"  FP32 blank reads: {fp32_blank}   PTQ blank reads: {ptq_blank}")
    print(f"  FP32==PTQ agreement (non-blank): {agree}/{len(order)} = {100*agree/len(order):.1f}%")
    print(f"  wrote: two_model_results.csv, fp32_results.csv, ptq_results.csv")


if __name__ == "__main__":
    main()
