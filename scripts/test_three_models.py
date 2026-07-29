#!/usr/bin/env python3
"""
Run all THREE meter-OCR models over an image folder and write ONE CSV:
    filename, fp32_result, qat_result, ptq_result

Models (paths relative to project root):
    FP32 : finetune/inference_v2/            (Paddle,  81 MB)  -> PaddleOCR TextRecognition
    QAT  : deploy/qat_int8/model/   (Paddle INT8, 22 MB) -> paddle.inference
    PTQ  : deploy/ptq_int8/model/   (ONNX  INT8, 21 MB)  -> onnxruntime

SETUP (uses your system packages — already present: paddle 3.3.1, onnxruntime, cv2, numpy):
    pip install paddlepaddle paddleocr onnxruntime opencv-python numpy

RUN (foreground):
    python3 test_three_models.py \
        --images "datasets/final_test/serialno_testing/images" \
        --out    "datasets/final_test/three_model_results.csv" \
        --resume

--resume keeps the rows an earlier run already produced and only reads the images
that are still missing. It also repairs a file left behind by a killed run: the
tail of such a file is NUL bytes (the write buffer never flushed), which are
dropped along with any row that is not a complete 4-field record. Omit --resume
to redo every image from scratch (the existing --out is overwritten).

TIMING (26,787 images): FP32 on GPU is fast (~4 min). QAT & PTQ are INT8 and run on
CPU here (INT8 GPU kernels aren't available on older GPUs) -> QAT is the slow one
(~0.5 s/img). Expect a few hours total on this machine; much faster on a modern GPU
server.  Progress prints every 200 images so you can see it working.
"""
import os, glob, csv, io, re, argparse, time
import numpy as np
# numpy-alias shim (old openpyxl in the import chain on some boxes)
for _a in ("float", "int", "bool", "object", "str"):
    if not hasattr(np, _a):
        setattr(np, _a, getattr(__builtins__, _a, None) if not isinstance(__builtins__, dict)
                else __builtins__.get(_a))
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
DICT = "deploy/ptq_int8/model/ppocrv5_dict.txt"
IMG_H, IMG_W = 48, 320
CHARS = ["blank"] + [l.rstrip("\n") for l in open(DICT, encoding="utf-8")] + [" "]


# ---------- shared preprocessing + CTC decode (for QAT & PTQ) ----------
def preprocess(bgr):
    h, w = bgr.shape[:2]
    rw = min(IMG_W, int(np.ceil(IMG_H * w / h)))
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
    """Normalize a raw reading to a tidy serial string (upper, alnum + single spaces)."""
    return re.sub(r"[^A-Z0-9]+", " ", (text or "").upper()).strip()


# ---------- model wrappers ----------
class OnnxRec:              # PTQ
    def __init__(self, path, use_gpu=False):
        import onnxruntime as ort
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.s = ort.InferenceSession(path, providers=prov)
        self.n = self.s.get_inputs()[0].name
    def read(self, bgr):
        return clean(decode(self.s.run(None, {self.n: preprocess(bgr)})[0]))

class PaddleRec:           # QAT (legacy .pdmodel)
    def __init__(self, pdmodel, pdiparams, use_gpu=False):
        import paddle.inference as pi
        cfg = pi.Config(pdmodel, pdiparams)
        cfg.enable_use_gpu(500, 0) if use_gpu else cfg.disable_gpu()
        cfg.disable_glog_info()
        self.p = pi.create_predictor(cfg)
        self.i = self.p.get_input_names()[0]
        self.o = self.p.get_output_names()[0]
    def read(self, bgr):
        h = self.p.get_input_handle(self.i); h.copy_from_cpu(preprocess(bgr)); self.p.run()
        return clean(decode(self.p.get_output_handle(self.o).copy_to_cpu()))

class TextRec:             # FP32 (via PaddleOCR)
    def __init__(self, model_dir, use_gpu=True):
        from paddleocr import TextRecognition
        kw = dict(device="gpu") if use_gpu else dict(enable_mkldnn=False)
        self.m = TextRecognition(model_name="PP-OCRv5_server_rec", model_dir=model_dir, **kw)
    def read(self, bgr):
        r = self.m.predict(bgr)
        d = r[0] if r else {}
        d = d if isinstance(d, dict) else getattr(d, "res", d)
        return clean(d.get("rec_text", ""))


def load_done(path):
    """Filenames already read by an earlier run, repairing the file if that run was killed.

    A killed run leaves a NUL tail (unflushed buffer) and possibly a torn final row,
    so keep only complete 4-field records and rewrite the file from those.
    """
    if not os.path.exists(path):
        return set()
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.replace(b"\x00", b"").decode("utf-8", "replace")

    rows, done = [], set()
    for row in csv.reader(io.StringIO(text)):
        if len(row) != 4 or row[0] in ("", "filename"):
            continue
        if row[0] in done:                       # keep the first read of a duplicate
            continue
        done.add(row[0])
        rows.append(row)

    dropped = len(raw) - len(text.encode("utf-8", "replace"))
    print(f"resume: {len(rows)} usable rows in {path}" + (f"  (repaired: dropped {dropped} NUL bytes)" if dropped else ""))

    with open(path, "w", newline="") as fh:      # rewrite clean, so we can append safely
        w = csv.writer(fh)
        w.writerow(["filename", "fp32_result", "qat_result", "ptq_result"])
        w.writerows(rows)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of .jpg crops")
    ap.add_argument("--out", default="three_model_results.csv")
    ap.add_argument("--resume", action="store_true",
                    help="keep existing rows in --out and only read the missing images")
    ap.add_argument("--fp32-gpu", action="store_true", default=True, help="run FP32 on GPU (fast)")
    ap.add_argument("--qat-gpu", action="store_true", help="try QAT on GPU (fails on old GPUs w/o INT8 kernels)")
    ap.add_argument("--ptq-gpu", action="store_true", help="run PTQ on GPU (needs onnxruntime-gpu + matching CUDA)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    print(f"{len(files)} images from {args.images}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = load_done(args.out) if args.resume else set()
    todo = [f for f in files if os.path.basename(f) not in done]
    if done:
        print(f"{len(done)} already done, {len(todo)} to go")
    if not todo:
        print("nothing to do — every image is already in the CSV.")
        return

    print("loading FP32 ...");  fp32 = TextRec("finetune/inference_v2", use_gpu=args.fp32_gpu)
    print("loading QAT  ...");  qat  = PaddleRec("deploy/qat_int8/model/meterocr_qat_int8.pdmodel",
                                                 "deploy/qat_int8/model/meterocr_qat_int8.pdiparams",
                                                 use_gpu=args.qat_gpu)
    print("loading PTQ  ...");  ptq  = OnnxRec("deploy/ptq_int8/model/meterocr_ptq_int8.onnx",
                                               use_gpu=args.ptq_gpu)

    t0 = time.time()
    with open(args.out, "a" if done else "w", newline="") as fh:
        w = csv.writer(fh)
        if not done:
            w.writerow(["filename", "fp32_result", "qat_result", "ptq_result"])
        for i, f in enumerate(todo, 1):
            bgr = cv2.imread(f)
            if bgr is None:
                w.writerow([os.path.basename(f), "", "", ""])
            else:
                w.writerow([os.path.basename(f), fp32.read(bgr), qat.read(bgr), ptq.read(bgr)])
            fh.flush()               # a kill must not lose buffered rows (see --resume)
            if i % 200 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(todo)}  ({el/i*1000:.0f} ms/img, eta {el/i*(len(todo)-i)/60:.0f} min)", flush=True)
    print(f"\nDone. Wrote {args.out}  ({len(done) + len(todo)} rows total, {len(todo)} new)  in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
