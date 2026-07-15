#!/usr/bin/env python3
"""
Web interface for meter serial-number detection.

Start it, open the printed URL in a browser, then either:
  * drag/select a SINGLE image, or
  * select a WHOLE FOLDER of images (the "Upload folder" button),
and the backend runs the fine-tuned PP-OCRv5 recognizer (same pipeline as
scripts/extract_serials.py) and shows the detected alphanumeric serial for each
image, with confidence and accept/review/blank status. Results are downloadable
as CSV.

Run:
  python3 scripts/web_app.py                 # http://127.0.0.1:5000
  python3 scripts/web_app.py --port 8000
  python3 scripts/web_app.py --base-model    # stock recognizer instead of fine-tuned
  python3 scripts/web_app.py --host 0.0.0.0  # reachable from other machines
"""
import os, io, base64, argparse
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response

# reuse the exact engine + logic from the main pipeline
import extract_serials as es

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB per upload batch
THRESHOLD = 0.95

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meter Serial Detection</title>
<style>
  :root{--bg:#0f1115;--card:#1a1d24;--fg:#e8eaed;--mut:#9aa0aa;--line:#2a2e37;
        --acc:#3b82f6;--ok:#22c55e;--warn:#f59e0b;--bad:#6b7280}
  @media (prefers-color-scheme: light){:root{--bg:#f5f6f8;--card:#fff;--fg:#1a1d24;
        --mut:#666;--line:#e2e4e8}}
  *{box-sizing:border-box}
  body{margin:0;font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
  header{padding:22px 26px;border-bottom:1px solid var(--line)}
  h1{margin:0;font-size:19px}
  .sub{color:var(--mut);font-size:13px;margin-top:3px}
  main{max-width:1100px;margin:0 auto;padding:26px}
  .drop{border:2px dashed var(--line);border-radius:14px;padding:34px;text-align:center;
        background:var(--card);transition:.15s}
  .drop.hover{border-color:var(--acc);background:rgba(59,130,246,.06)}
  .btns{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:16px}
  button,.filebtn{font:inherit;font-weight:600;border:1px solid var(--line);background:var(--card);
        color:var(--fg);padding:10px 18px;border-radius:10px;cursor:pointer}
  .primary{background:var(--acc);border-color:var(--acc);color:#fff}
  .filebtn input{display:none}
  #bar{height:6px;background:var(--line);border-radius:6px;overflow:hidden;margin:18px 0;display:none}
  #bar>div{height:100%;width:0;background:var(--acc);transition:width .2s}
  .stats{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}
  .pill{padding:6px 12px;border-radius:20px;font-size:13px;font-weight:600;border:1px solid var(--line)}
  table{width:100%;border-collapse:collapse;margin-top:8px;font-size:14px}
  th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
  th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
  td img{height:34px;border-radius:5px;background:#000;cursor:zoom-in}
  .serial{font-family:ui-monospace,Menlo,monospace;font-weight:700;font-size:15px}
  .tag{font-size:11px;font-weight:700;padding:3px 8px;border-radius:20px;text-transform:uppercase}
  .accepted{background:rgba(34,197,94,.15);color:var(--ok)}
  .review{background:rgba(245,158,11,.15);color:var(--warn)}
  .blank{background:rgba(107,114,128,.18);color:var(--mut)}
  .toolbar{display:flex;gap:12px;align-items:center;justify-content:space-between;margin-top:22px}
  a.dl{color:var(--acc);font-weight:600;text-decoration:none}
  #zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;cursor:zoom-out}
  #zoom img{max-width:92vw;max-height:92vh}
</style></head><body>
<header><h1>⚡ Meter Serial Number Detection</h1>
<div class="sub">Upload a single image or a whole folder — the fine-tuned model reads the alphanumeric serial on each.</div></header>
<main>
  <div class="drop" id="drop">
    <div>Drag &amp; drop an image or folder here, or:</div>
    <div class="btns">
      <label class="filebtn primary">Upload image(s)<input id="fileInput" type="file" accept="image/*" multiple></label>
      <label class="filebtn">Upload folder<input id="dirInput" type="file" webkitdirectory multiple></label>
    </div>
  </div>
  <div id="bar"><div></div></div>
  <div class="stats" id="stats"></div>
  <div class="toolbar" id="toolbar" style="display:none">
    <div id="count"></div><a class="dl" id="dl" href="#">⬇ Download results.csv</a>
  </div>
  <table id="tbl" style="display:none">
    <thead><tr><th>Image</th><th>Reading</th><th>Conf.</th><th>Status</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
</main>
<div id="zoom"><img></div>
<script>
const $=s=>document.querySelector(s);
const drop=$("#drop"),bar=$("#bar"),barIn=bar.firstElementChild;
let results=[];

function esc(s){return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

async function send(files){
  if(!files.length) return;
  const imgs=[...files].filter(f=>f.type.startsWith("image/")||/\.(jpe?g|png|bmp|tiff?|webp)$/i.test(f.name));
  if(!imgs.length){alert("No image files found.");return;}
  results=[]; $("#rows").innerHTML=""; $("#tbl").style.display="none"; $("#toolbar").style.display="none";
  bar.style.display="block"; barIn.style.width="3%"; $("#stats").innerHTML="";
  // send in chunks so big folders stream in and the UI updates
  const CHUNK=40; let done=0;
  for(let i=0;i<imgs.length;i+=CHUNK){
    const fd=new FormData();
    imgs.slice(i,i+CHUNK).forEach(f=>fd.append("images",f,f.webkitRelativePath||f.name));
    const r=await fetch("/infer",{method:"POST",body:fd});
    const j=await r.json();
    results=results.concat(j.results);
    addRows(j.results);
    done+=imgs.slice(i,i+CHUNK).length;
    barIn.style.width=(done/imgs.length*100).toFixed(0)+"%";
    renderStats();
  }
  bar.style.display="none";
  $("#toolbar").style.display="flex"; $("#count").textContent=results.length+" image(s)";
}

function addRows(rows){
  $("#tbl").style.display="table";
  const tb=$("#rows");
  for(const r of rows){
    const tr=document.createElement("tr");
    tr.innerHTML=`<td><img src="${r.thumb}" title="${esc(r.name)}"></td>
      <td class="serial">${esc(r.reading)||'<span style="color:var(--mut)">—</span>'}</td>
      <td>${r.score.toFixed(3)}</td>
      <td><span class="tag ${r.status}">${r.status}</span></td>`;
    tb.appendChild(tr);
  }
}

function renderStats(){
  const c={accepted:0,review:0,blank:0};
  results.forEach(r=>c[r.status]++);
  $("#stats").innerHTML=
    `<span class="pill accepted">${c.accepted} accepted</span>
     <span class="pill review">${c.review} review</span>
     <span class="pill blank">${c.blank} blank</span>`;
}

// CSV download
$("#dl").onclick=e=>{
  e.preventDefault();
  const head=["image","reading","score","status"];
  const rows=results.map(r=>[r.name,r.reading,r.score,r.status]
     .map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(","));
  const blob=new Blob([head.join(",")+"\n"+rows.join("\n")],{type:"text/csv"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="results.csv";a.click();
};

// zoom
$("#zoom").onclick=()=>$("#zoom").style.display="none";
document.addEventListener("click",e=>{if(e.target.tagName==="IMG"&&e.target.closest("#rows")){
  $("#zoom").firstElementChild.src=e.target.src;$("#zoom").style.display="flex";}});

// wiring
$("#fileInput").onchange=e=>send(e.target.files);
$("#dirInput").onchange=e=>send(e.target.files);
["dragenter","dragover"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add("hover");}));
["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove("hover");}));
drop.addEventListener("drop",e=>send(e.dataTransfer.files));
</script></body></html>"""


def infer_bgr(bgr):
    """Run the pipeline on a decoded BGR image -> (reading, score, status)."""
    lines_raw = es.run_ocr(bgr)
    lines_pre = es.run_ocr(es.preprocess(bgr))
    text, tscore = es.choose_text(es.display_text(lines_raw), es.display_text(lines_pre))
    if not text:
        status = "blank"
    elif tscore >= THRESHOLD:
        status = "accepted"
    else:
        status = "review"
    return text, tscore, status


def thumb(bgr, h=48):
    """Small base64 JPEG thumbnail for display in the table."""
    scale = h / max(1, bgr.shape[0])
    small = cv2.resize(bgr, (max(1, int(bgr.shape[1] * scale)), h))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/infer", methods=["POST"])
def infer():
    out = []
    for f in request.files.getlist("images"):
        data = np.frombuffer(f.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        name = f.filename or "image"
        if bgr is None:
            out.append(dict(name=name, reading="", score=0.0,
                            status="blank", thumb=""))
            continue
        text, score, status = infer_bgr(bgr)
        out.append(dict(name=name, reading=text,
                        score=round(float(score), 3), status=status, thumb=thumb(bgr)))
    return jsonify(results=out)


def main():
    ap = argparse.ArgumentParser(description="Web UI for meter serial detection.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--base-model", action="store_true",
                    help="use the stock recognizer instead of the fine-tuned one")
    args = ap.parse_args()

    global THRESHOLD
    THRESHOLD = args.threshold

    print("Loading model ...", flush=True)
    es.build_engine("recog", use_finetuned=not args.base_model)
    print(f"\n  Meter serial detection UI  ->  http://{args.host}:{args.port}\n", flush=True)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
