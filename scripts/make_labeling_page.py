#!/usr/bin/env python3
"""
Build results/label_tool.html — a local page to hand-label a representative
sample of images (ground truth for measuring accuracy).

It picks a stratified sample across the confidence range (high/mid/low/blank)
from results/test_serials.csv, pre-fills each with the model's current guess so
you only correct mistakes, and lets you export results/labels.csv.

Usage:
  python3 scripts/make_labeling_page.py                 # ~180 images
  python3 scripts/make_labeling_page.py --n 250 --data data/serial_roi_test
"""
import csv, os, json, html, argparse, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/test_serials.csv")
ap.add_argument("--data", default="data/serial_roi_test")
ap.add_argument("--out", default="results/label_tool.html")
ap.add_argument("--n", type=int, default=180, help="total images to label")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed)

rows = [r for r in csv.DictReader(open(args.csv)) if "(Copy)" not in r["image"]]

# stratify by confidence so the sample spans easy -> hard -> blank
buckets = {"high (>=0.90)": [], "mid (0.70-0.90)": [], "low (<0.70)": [], "blank": []}
for r in rows:
    txt = r["display_text"].strip()
    sc = float(r["score"]) if r["score"] else 0.0
    if not txt:
        buckets["blank"].append(r)
    elif sc >= 0.90:
        buckets["high (>=0.90)"].append(r)
    elif sc >= 0.70:
        buckets["mid (0.70-0.90)"].append(r)
    else:
        buckets["low (<0.70)"].append(r)

# proportions of the sample (blank + low over-sampled: that's where errors live)
share = {"high (>=0.90)": 0.30, "mid (0.70-0.90)": 0.22, "low (<0.70)": 0.24, "blank": 0.24}
sample = []
for name, frac in share.items():
    pool = buckets[name]
    k = min(len(pool), round(args.n * frac))
    sample += random.sample(pool, k)
random.shuffle(sample)

IMG_REL = os.path.relpath(args.data, os.path.dirname(args.out))
data = [{"image": r["image"], "guess": r["display_text"].strip()} for r in sample]
print(f"sampled {len(data)} images "
      + ", ".join(f"{k}:{min(len(v), round(args.n*share[k]))}" for k, v in buckets.items()))

DOC = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Serial Labeling</title><style>
:root{--bg:#0f1115;--panel:#161a22;--line:#2a2f3a;--text:#e6e8ec;--muted:#9aa3b2;--accent:#4a9eff;--good:#2ecc71}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);
display:flex;flex-direction:column;height:100vh}
#bar{display:flex;align-items:center;gap:14px;padding:12px 20px;border-bottom:1px solid var(--line);flex-wrap:wrap}
#bar h1{font-size:15px;margin:0;flex:1}.btn{padding:9px 14px;border-radius:8px;border:1px solid var(--line);
background:var(--panel);color:var(--text);cursor:pointer;font-size:14px}.btn:hover{background:#20293a}
.btn.pri{background:var(--accent);border-color:var(--accent);color:#04121f;font-weight:600}
.btn.good{background:var(--good);border-color:var(--good);color:#062;font-weight:600}
#prog{color:var(--muted);font-size:13px}
#stage{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:20px;padding:24px}
.imgbox{background:#000;border:1px solid var(--line);border-radius:10px;padding:18px}
.imgbox img{max-width:820px;width:100%;height:auto;image-rendering:auto}
#guess{color:var(--muted);font-size:13px;font-family:ui-monospace,monospace}
#inp{font-family:ui-monospace,monospace;font-size:26px;letter-spacing:2px;text-align:center;padding:12px 16px;
border-radius:10px;border:2px solid var(--accent);background:var(--panel);color:var(--text);width:min(560px,90vw);text-transform:uppercase}
.row{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
.hint{color:#6b7280;font-size:12px;max-width:640px;text-align:center}
</style></head><body>
<div id="bar"><h1>Ground-truth labeling</h1><span id="prog"></span>
<button class="btn" id="prev">&larr; Prev</button>
<button class="btn good" id="dl">Download labels.csv</button></div>
<div id="stage">
<div class="imgbox"><img id="img"></div>
<div id="guess"></div>
<input id="inp" autocomplete="off" spellcheck="false" placeholder="type the serial exactly as shown">
<div class="row">
<button class="btn pri" id="save">Save &amp; Next  (Enter)</button>
<button class="btn" id="unread">Unreadable / blank  (\\)</button>
<button class="btn" id="acc">Guess is correct  (Tab)</button></div>
<div class="hint">Type what you see on the meter, exactly (keep letters, prefix and spacing, e.g. <b>TPWODL 1050390</b>).
If nothing is legible, click <b>Unreadable</b>. Progress is saved in your browser; click
<b>Download labels.csv</b> anytime and move it to the <b>results/</b> folder.</div>
</div>
<script>
const DATA=__DATA__, IMGDIR="__IMGDIR__", KEY="serial_labels_v1";
let cur=0;
const store=JSON.parse(localStorage.getItem(KEY)||"{}");
const img=document.getElementById('img'),inp=document.getElementById('inp'),
guess=document.getElementById('guess'),prog=document.getElementById('prog');
function render(){const d=DATA[cur];img.src=IMGDIR+'/'+d.image;
guess.textContent="model guess: "+(d.guess||"(none)");
inp.value=(d.image in store)?store[d.image]:d.guess;inp.focus();inp.select();
const done=Object.keys(store).length;prog.textContent=`${cur+1} / ${DATA.length}  ·  labeled ${done}`;}
function save(val){store[DATA[cur].image]=val;localStorage.setItem(KEY,JSON.stringify(store));
if(cur<DATA.length-1){cur++;render();}else{alert("Reached the end. Click Download labels.csv");}}
document.getElementById('save').onclick=()=>save(inp.value.trim().toUpperCase());
document.getElementById('unread').onclick=()=>save("");
document.getElementById('acc').onclick=()=>save((DATA[cur].guess||"").toUpperCase());
document.getElementById('prev').onclick=()=>{if(cur>0){cur--;render();}};
inp.addEventListener('keydown',e=>{
 if(e.key==='Enter'){e.preventDefault();save(inp.value.trim().toUpperCase());}
 if(e.key==='Tab'){e.preventDefault();save((DATA[cur].guess||"").toUpperCase());}
 if(e.key==='\\\\'){e.preventDefault();save("");}});
document.getElementById('dl').onclick=()=>{
 let csv="image,true_label\\n";
 for(const d of DATA){const v=(d.image in store)?store[d.image]:"";
   csv+='"'+d.image+'","'+v.replace(/"/g,'""')+'"\\n';}
 const b=new Blob([csv],{type:"text/csv"});const a=document.createElement('a');
 a.href=URL.createObjectURL(b);a.download="labels.csv";a.click();};
render();
</script></body></html>"""
open(args.out, "w").write(DOC.replace("__DATA__", json.dumps(data)).replace("__IMGDIR__", html.escape(IMG_REL)))
print(f"Wrote {args.out}")
