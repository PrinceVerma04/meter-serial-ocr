#!/usr/bin/env python3
"""Score finetune/inference_v3 on the held-out test split (exact-match, normalized)."""
import os, re, numpy as np
for _a in ("float","int","bool","object","str"):
    if not hasattr(np,_a): setattr(np,_a,getattr(__builtins__,_a,None) if not isinstance(__builtins__,dict) else __builtins__.get(_a))
import cv2, paddle.inference as pi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS   = os.path.join(ROOT, "finetune/dataset_retrain")
MODEL= os.path.join(ROOT, "finetune/inference_v3/inference.json")
PARAM= os.path.join(ROOT, "finetune/inference_v3/inference.pdiparams")
DICT = os.path.join(ROOT, "deploy/ptq_int8/model/ppocrv5_dict.txt")
IMG_H, IMG_W = 48, 320
CHARS = ["blank"] + [l.rstrip("\n") for l in open(DICT, encoding="utf-8")] + [" "]

def norm(s): return re.sub(r"[^A-Z0-9]", "", s.upper())

def preprocess(bgr):
    h,w = bgr.shape[:2]
    rw = min(IMG_W, max(1, int(np.ceil(IMG_H*w/h))))
    r = cv2.resize(bgr,(rw,IMG_H)).astype("float32").transpose(2,0,1)/255.0
    r = (r-0.5)/0.5
    pad = np.zeros((1,3,IMG_H,IMG_W),dtype="float32"); pad[0,:,:,:rw]=r
    return pad

def ctc(logits):
    probs = logits[0]; idx = probs.argmax(1); out=[]; prev=-1
    for i in idx:
        if i!=prev and i!=0: out.append(CHARS[i] if i<len(CHARS) else "")
        prev=i
    return "".join(out)

cfg = pi.Config(MODEL, PARAM); cfg.enable_use_gpu(1000,0); cfg.disable_glog_info()
pred = pi.create_predictor(cfg)
iname = pred.get_input_names()[0]; oname = pred.get_output_names()[0]
def infer(x):
    h=pred.get_input_handle(iname); h.copy_from_cpu(x); pred.run()
    return pred.get_output_handle(oname).copy_to_cpu()

rows = [l.rstrip("\n").split("\t") for l in open(f"{DS}/test.txt") if l.strip()]
correct=0; total=0; errors=[]
for rel,label in rows:
    bgr = cv2.imread(os.path.join(DS, rel))
    if bgr is None: continue
    ptext = ctc(infer(preprocess(bgr)))
    total+=1
    if norm(ptext)==norm(label): correct+=1
    elif len(errors)<15: errors.append((rel.split("/")[-1], label, ptext))

print(f"\ninference_v3 (new FP32) on held-out test: {correct}/{total} = {100*correct/total:.2f}% exact-match (normalized)")
print("\nfirst mismatches (file | truth | pred):")
for f,t,p in errors: print(f"  {f}  {t!r} -> {p!r}")
