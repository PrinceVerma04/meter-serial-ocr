#!/usr/bin/env python3
"""
Run every OCR model on the test dataset and summarise the reading per image.

Models (each optional; skipped if its library/model isn't available):
  paddle_mobile  - PaddleOCR PP-OCRv5  (server detector + English MOBILE recognizer)
  paddle_server  - PaddleOCR PP-OCRv5  (server detector + SERVER recognizer)
  easyocr        - EasyOCR (CRAFT + CRNN), different architecture
  trocr          - Microsoft TrOCR base printed (transformer, heavy, GPU if available)

Output (written next to this script):
  test_results.csv   - one row per image, one column per model (+ score + agreement)
  test_results.html  - single-image viewer: image + every model's reading, prev/next

USAGE
  python3 run_test_models.py                     # all models, all images
  python3 run_test_models.py --limit 50          # quick trial on first 50 images
  python3 run_test_models.py --skip-copies        # ignore the "(Copy)" duplicates
  python3 run_test_models.py --models paddle_server,trocr
"""
import os, re, csv, sys, glob, json, html, argparse

# ---- shim: an old openpyxl on this box references removed numpy aliases ----
import numpy as np
for _a in ("float", "int", "bool", "object", "str"):
    if not hasattr(np, _a):
        setattr(np, _a, __builtins__[_a] if isinstance(__builtins__, dict) else getattr(__builtins__, _a))
import cv2

# run from the project root regardless of where this script is invoked
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ======================= CONFIG (overridable via --data / --out) =======================
IMG_DIR = "data/serial_roi_test"             # default: the test dataset
MIN_LEN = 5                                  # ignore text shorter than this (tiny labels)
OUT_CSV = "results/test_results.csv"
OUT_HTML = "results/test_results.html"


# ======================= extraction =======================
def display_text(lines):
    """Verbatim serial-area text: keep letters/digits and single spaces, drop
    stray punctuation, and pick the line with the most content (skips tiny
    labels like 'Sr.No.'). Returns (text, score)."""
    best = None
    for text, score in lines:
        norm = re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()
        alnum = sum(c.isalnum() for c in norm)
        if alnum < MIN_LEN:
            continue
        key = (alnum, score)
        if best is None or key > best[0]:
            best = (key, norm, score)
    return (best[1], best[2]) if best else ("", 0.0)


# ======================= model builders (lazy) =======================
def make_paddle(rec_name, det_name="PP-OCRv5_server_det"):
    import paddle
    from paddleocr import PaddleOCR
    on_gpu = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    kw = dict(device="gpu") if on_gpu else dict(enable_mkldnn=False)  # mkldnn fix is CPU-only
    print(f"    paddle recognizer={rec_name} on {'GPU' if on_gpu else 'CPU'}")
    eng = PaddleOCR(
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name=det_name, text_recognition_model_name=rec_name,
        **kw,
    )

    def run(bgr):
        out = []
        for r in eng.predict(bgr):
            d = r if isinstance(r, dict) else getattr(r, "res", r)
            out += [(t, float(s)) for t, s in zip(d.get("rec_texts", []), d.get("rec_scores", []))]
        return out
    return run


def make_easyocr():
    import torch, easyocr
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    def run(bgr):
        return [(t, float(c)) for _, t, c in reader.readtext(bgr, detail=1, paragraph=False)]
    return run


def make_trocr():
    # transformers 5.x's fast tokenizer for this checkpoint is broken, so build
    # from the slow RobertaTokenizer + image processor manually.
    import torch
    from transformers import RobertaTokenizer, AutoImageProcessor, VisionEncoderDecoderModel
    name = "microsoft/trocr-base-printed"
    tok = RobertaTokenizer.from_pretrained(name)
    ip = AutoImageProcessor.from_pretrained(name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionEncoderDecoderModel.from_pretrained(name).to(dev).eval()

    def run(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pix = ip(images=rgb, return_tensors="pt").pixel_values.to(dev)
        with torch.no_grad():
            ids = model.generate(pix, max_new_tokens=32)
        return [(tok.batch_decode(ids, skip_special_tokens=True)[0], 1.0)]
    return run


BUILDERS = {
    "paddle_mobile": lambda: make_paddle("en_PP-OCRv5_mobile_rec"),
    "paddle_server": lambda: make_paddle("PP-OCRv5_server_rec"),
    "easyocr":       make_easyocr,
    "trocr":         make_trocr,
}


# ======================= main =======================
def main():
    global IMG_DIR, OUT_CSV, OUT_HTML
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=IMG_DIR, help="folder of .jpg crops to run on")
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--out-html", default=OUT_HTML)
    ap.add_argument("--models", default="paddle_mobile,paddle_server,easyocr,trocr")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N images")
    ap.add_argument("--skip-copies", action="store_true", help="ignore '(Copy)' duplicate files")
    args = ap.parse_args()
    IMG_DIR, OUT_CSV, OUT_HTML = args.data, args.out_csv, args.out_html
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)

    if not os.path.isdir(IMG_DIR):
        sys.exit(f"ERROR: folder not found: {IMG_DIR!r} (run this from the project directory)")

    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    if args.skip_copies:
        files = [f for f in files if "(Copy)" not in os.path.basename(f)]
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} images from {IMG_DIR}")

    want = [x.strip() for x in args.models.split(",") if x.strip()]
    engines = {}
    for name in want:
        try:
            print(f"loading {name} ...", flush=True)
            engines[name] = BUILDERS[name]()
        except Exception as e:
            print(f"  !! skipping {name}: {str(e)[:130]}")
    names = list(engines)
    if not names:
        sys.exit("No models could be loaded.")

    rows = []
    for i, path in enumerate(files, 1):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rec = {"image": os.path.basename(path)}
        reads = []
        for name in names:
            try:
                t, sc = display_text(engines[name](bgr))
            except Exception:
                t, sc = "ERR", 0.0
            rec[name] = t
            rec[name + "_score"] = round(sc, 3)
            if t:
                reads.append(t)
        rec["agree"] = "yes" if reads and len(set(reads)) == 1 else ("none" if not reads else "no")
        rows.append(rec)
        if i % 25 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {rec['image'][-26:]}  " +
                  " | ".join(f"{n}={rec[n] or '-'}" for n in names), flush=True)

    # ---- CSV ----
    cols = ["image"] + sum(([n, n + "_score"] for n in names), []) + ["agree"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- HTML single-image viewer ----
    write_html(rows, names)
    agree = sum(1 for r in rows if r["agree"] == "yes")
    print(f"\nDone. {len(rows)} images. all-agree: {agree}. "
          f"Wrote {OUT_CSV} and {OUT_HTML}")


def write_html(rows, names):
    data = [{
        "image": r["image"], "agree": r["agree"],
        "results": [{"model": n, "text": r.get(n, ""), "score": r.get(n + "_score", 0)} for n in names],
    } for r in rows]
    # image path is relative to this html file
    # image src is relative to the HTML file (which lives in results/)
    imgdir = html.escape(os.path.relpath(IMG_DIR, os.path.dirname(OUT_HTML)))
    doc = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR Test-Set Results</title><style>
:root{--bg:#0f1115;--panel:#161a22;--line:#2a2f3a;--text:#e6e8ec;--muted:#9aa3b2;
--good:#2ecc71;--lo:#e67e22;--none:#e74c3c;--accent:#4a9eff;}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,sans-serif;background:var(--bg);
color:var(--text);display:flex;height:100vh;overflow:hidden}
#side{width:290px;flex:none;border-right:1px solid var(--line);display:flex;flex-direction:column;background:#12151b}
#side h2{font-size:12px;margin:0;padding:14px 14px 8px;color:var(--muted);text-transform:uppercase}
#filter{margin:0 14px 10px;padding:7px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--text)}
#list{overflow-y:auto;flex:1}.item{padding:8px 14px;border-bottom:1px solid #1d222b;cursor:pointer;
font-size:11px;display:flex;gap:8px;align-items:center;color:var(--muted);word-break:break-all}
.item:hover{background:#1b2029}.item.active{background:#20293a;color:var(--text);border-left:3px solid var(--accent);padding-left:11px}
.dot{width:8px;height:8px;border-radius:50%;flex:none}.dot.yes{background:var(--good)}.dot.no{background:#f1c40f}.dot.none{background:var(--none)}
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#top{display:flex;align-items:center;gap:14px;padding:14px 20px;border-bottom:1px solid var(--line)}
.btn{padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--panel);color:var(--text);cursor:pointer}
.btn:disabled{opacity:.4}#pos{color:var(--muted);font-size:13px;min-width:80px;text-align:center}
#stage{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;align-items:center;gap:18px}
.imgbox{background:#000;border:1px solid var(--line);border-radius:10px;padding:16px}
.imgbox img{max-width:760px;width:100%;height:auto}
.fname{display:flex;gap:10px;align-items:center;font-family:ui-monospace,monospace;font-size:13px;
color:var(--muted);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
#copy{padding:5px 12px;border-radius:6px;border:1px solid var(--line);background:#20293a;color:var(--text);cursor:pointer;font-size:12px}
#copy.done{background:var(--good);color:#062}
table{width:100%;max-width:760px;border-collapse:collapse}td,th{padding:12px 14px;border:1px solid var(--line);text-align:left}
th{font-size:12px;color:var(--muted);text-transform:uppercase}.mname{font-weight:600}
.ser{font-family:ui-monospace,monospace;font-size:20px;font-weight:600;letter-spacing:1px}
.ser.ok{color:var(--good)}.ser.lo{color:var(--lo)}.ser.none{color:var(--none)}.sc{color:var(--muted);text-align:right;width:80px}
</style></head><body>
<div id="side"><h2>Images (<span id="cnt"></span>)</h2>
<select id="filter"><option value="all">All</option><option value="no">Disagreements</option>
<option value="yes">All agree</option><option value="none">No read</option></select>
<div id="list"></div></div>
<div id="main"><div id="top">
<button class="btn" id="prev">&larr; Prev</button><div id="pos"></div><button class="btn" id="next">Next &rarr;</button>
<span style="color:#6b7280;font-size:12px">&larr;/&rarr; keys</span></div>
<div id="stage"><div class="imgbox"><img id="img"></div>
<div class="fname"><span id="fn"></span><button id="copy">Copy name</button></div>
<table><thead><tr><th>Model</th><th>Reading</th><th class="sc">Score</th></tr></thead><tbody id="tb"></tbody></table>
</div></div>
<script>
const DATA=__DATA__, IMGDIR="__IMGDIR__";
let view=DATA.map((_,i)=>i), cur=0;
const listEl=document.getElementById('list');
function buildList(){listEl.innerHTML='';view.forEach((di,vi)=>{const d=DATA[di];const e=document.createElement('div');
e.className='item'+(vi===cur?' active':'');e.innerHTML=`<span class="dot ${d.agree}"></span>${d.image.slice(-24)}`;
e.onclick=()=>{cur=vi;render()};listEl.appendChild(e);});}
function render(){const d=DATA[view[cur]];
document.getElementById('img').src=IMGDIR+'/'+d.image;
document.getElementById('fn').textContent=d.image;
document.getElementById('pos').textContent=(cur+1)+' / '+view.length;
const tb=document.getElementById('tb');tb.innerHTML='';
d.results.forEach(r=>{const cls=!r.text?'none':(r.score>0&&r.score<0.7?'lo':'ok');
tb.innerHTML+=`<tr><td class="mname">${r.model}</td><td><span class="ser ${cls}">${r.text||'—'}</span></td>
<td class="sc">${r.score?r.score.toFixed(3):''}</td></tr>`;});
[...listEl.children].forEach((c,i)=>c.classList.toggle('active',i===cur));
const a=listEl.children[cur];if(a)a.scrollIntoView({block:'nearest'});
document.getElementById('prev').disabled=cur<=0;document.getElementById('next').disabled=cur>=view.length-1;
const cp=document.getElementById('copy');cp.textContent='Copy name';cp.classList.remove('done');}
function applyFilter(v){view=DATA.map((_,i)=>i).filter(i=>v==='all'||DATA[i].agree===v);if(!view.length)view=[0];cur=0;
document.getElementById('cnt').textContent=view.length;buildList();render();}
document.getElementById('prev').onclick=()=>{if(cur>0){cur--;render()}};
document.getElementById('next').onclick=()=>{if(cur<view.length-1){cur++;render()}};
document.getElementById('filter').onchange=e=>applyFilter(e.target.value);
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'&&cur>0){cur--;render()}if(e.key==='ArrowRight'&&cur<view.length-1){cur++;render()}});
document.getElementById('copy').onclick=()=>{const n=DATA[view[cur]].image;const done=()=>{const c=document.getElementById('copy');c.textContent='Copied ✓';c.classList.add('done')};
if(navigator.clipboard)navigator.clipboard.writeText(n).then(done).catch(fb);else fb();
function fb(){const t=document.createElement('textarea');t.value=n;document.body.appendChild(t);t.select();try{document.execCommand('copy');done()}catch(e){}document.body.removeChild(t);}};
document.getElementById('cnt').textContent=view.length;buildList();render();
</script></body></html>"""
    doc = doc.replace("__DATA__", json.dumps(data)).replace("__IMGDIR__", imgdir)
    open(OUT_HTML, "w").write(doc)


if __name__ == "__main__":
    main()
