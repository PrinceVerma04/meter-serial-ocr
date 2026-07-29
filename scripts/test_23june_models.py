#!/usr/bin/env python3
"""
Test all THREE meter-OCR models on the 23-June unseen dataset and summarise.

Dataset (no ground truth): 23 june analysis/merged_unseen_images/  (~9970 crops)

Models (paths relative to project root):
    FP32 : finetune/inference_v2/                         (Paddle,      81 MB)
    QAT  : deploy/qat_int8/model/meterocr_qat_int8.*      (Paddle INT8, 22 MB)
    PTQ  : deploy/ptq_int8/model/meterocr_ptq_int8.onnx   (ONNX  INT8,  21 MB)

Outputs (all under "23 june analysis/"):
    fp32_results.csv                filename,label      (per-model, named by model)
    qat_results.csv                 filename,label
    ptq_results.csv                 filename,label
    three_model_results.csv         filename,fp32_result,qat_result,ptq_result  (combined)

Then build the dashboard from the combined CSV:
    python3 scripts/build_comparison_dashboard.py \
        --csv    "23 june analysis/three_model_results.csv" \
        --images "23 june analysis/merged_unseen_images" \
        --out    "23 june analysis/comparison_dashboard.html"

RUN:
    python3 scripts/test_23june_models.py --resume
    # (default paths already point at 23 june analysis/; --resume continues a killed run)

There is NO ground truth here, so "performance" is measured as cross-model
AGREEMENT (when all three independent models agree, the read is almost certainly
right) plus read/blank rates. Absolute accuracy needs human review of the dashboard.
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
    """Filenames already read by an earlier run, repairing a file left by a killed run."""
    if not os.path.exists(path):
        return set()
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.replace(b"\x00", b"").decode("utf-8", "replace")
    rows, done = [], set()
    for row in csv.reader(io.StringIO(text)):
        if len(row) != 4 or row[0] in ("", "filename") or row[0] in done:
            continue
        done.add(row[0]); rows.append(row)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "fp32_result", "qat_result", "ptq_result"])
        w.writerows(rows)
    print(f"resume: {len(rows)} rows already in {path}")
    return done


def write_per_model(combined_csv, out_dir):
    """Split the combined CSV into one CSV per model, named by the model."""
    rows = list(csv.DictReader(open(combined_csv)))
    for key, name in [("fp32_result", "fp32"), ("qat_result", "qat"), ("ptq_result", "ptq")]:
        p = os.path.join(out_dir, f"{name}_results.csv")
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["filename", "label"])
            for r in rows:
                w.writerow([r["filename"], r[key]])
        print("wrote", p)


def summarise(combined_csv):
    rows = list(csv.DictReader(open(combined_csv)))
    n = len(rows)
    def norm(x): return (x or "").strip()
    all3 = two = none = 0
    blanks = {"fp32": 0, "qat": 0, "ptq": 0}
    for r in rows:
        a, b, c = norm(r["fp32_result"]), norm(r["qat_result"]), norm(r["ptq_result"])
        for k, v in (("fp32", a), ("qat", b), ("ptq", c)):
            if not v: blanks[k] += 1
        vals = [a, b, c]
        if a == b == c:            all3 += 1
        elif len({a, b, c}) == 2:  two += 1
        else:                       none += 1
    print("\n" + "=" * 48)
    print(f" SUMMARY  ({n} images, no ground truth)")
    print("=" * 48)
    print(f" all 3 models agree : {all3:5d}  ({100*all3/n:.1f}%)   <- high-confidence correct")
    print(f" exactly 2 agree    : {two:5d}  ({100*two/n:.1f}%)")
    print(f" all 3 differ       : {none:5d}  ({100*none/n:.1f}%)   <- needs review")
    print(f" blanks  fp32={blanks['fp32']}  qat={blanks['qat']}  ptq={blanks['ptq']}")
    print("=" * 48)
    print(" Absolute accuracy = review the dashboard and tag each read.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="23 june analysis/merged_unseen_images")
    ap.add_argument("--out-dir", default="23 june analysis")
    ap.add_argument("--resume", action="store_true", help="continue a previous/killed run")
    ap.add_argument("--fp32-gpu", action="store_true", default=True)
    ap.add_argument("--qat-gpu", action="store_true", help="try QAT on GPU (needs INT8 kernels)")
    ap.add_argument("--ptq-gpu", action="store_true", help="PTQ on GPU (needs onnxruntime-gpu)")
    args = ap.parse_args()

    combined = os.path.join(args.out_dir, "three_model_results.csv")
    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    print(f"{len(files)} images from {args.images}")
    os.makedirs(args.out_dir, exist_ok=True)

    done = load_done(combined) if args.resume else set()
    todo = [f for f in files if os.path.basename(f) not in done]
    print(f"{len(done)} done, {len(todo)} to go")

    if todo:
        print("loading FP32 ..."); fp32 = TextRec("finetune/inference_v2", use_gpu=args.fp32_gpu)
        print("loading QAT  ..."); qat  = PaddleRec("deploy/qat_int8/model/meterocr_qat_int8.pdmodel",
                                                    "deploy/qat_int8/model/meterocr_qat_int8.pdiparams",
                                                    use_gpu=args.qat_gpu)
        print("loading PTQ  ..."); ptq  = OnnxRec("deploy/ptq_int8/model/meterocr_ptq_int8.onnx",
                                                  use_gpu=args.ptq_gpu)
        t0 = time.time()
        with open(combined, "a" if done else "w", newline="") as fh:
            w = csv.writer(fh)
            if not done:
                w.writerow(["filename", "fp32_result", "qat_result", "ptq_result"])
            for i, f in enumerate(todo, 1):
                bgr = cv2.imread(f)
                if bgr is None:
                    w.writerow([os.path.basename(f), "", "", ""])
                else:
                    w.writerow([os.path.basename(f), fp32.read(bgr), qat.read(bgr), ptq.read(bgr)])
                fh.flush()
                if i % 200 == 0:
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  ({el/i*1000:.0f} ms/img, "
                          f"eta {el/i*(len(todo)-i)/60:.0f} min)", flush=True)
        print(f"\nInference done in {(time.time()-t0)/60:.1f} min -> {combined}")

    write_per_model(combined, args.out_dir)
    summarise(combined)
    print(f"\nNext, build the dashboard:\n"
          f'  python3 scripts/build_comparison_dashboard.py \\\n'
          f'      --csv    "{combined}" \\\n'
          f'      --images "{args.images}" \\\n'
          f'      --out    "{os.path.join(args.out_dir, "comparison_dashboard.html")}"')


if __name__ == "__main__":
    main()
