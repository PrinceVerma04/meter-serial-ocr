#!/usr/bin/env python3
"""
Flask front-end shared by app_cpu.py and app_gpu.py.

`create_app(device)` builds the same web page for either device — the ONLY
difference between the CPU and GPU builds is the string passed here, which the
Engine uses to pin every model to that device. Upload a single image or a whole
folder; the page shows each of the 3 models' extracted serial number and its
per-image inference time, plus average ms per model and a CSV export.
"""
import os
import io
import base64
import time
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response

from engine import Engine, MODEL_SPECS, ACCEPT_THRESHOLD

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _thumb(bgr, box=120):
    h, w = bgr.shape[:2]
    s = box / max(h, w)
    small = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def create_app(device):
    engine = Engine(device)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB / batch

    dev_label = "GPU" if device == "gpu" else "CPU"

    @app.route("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.route("/meta")
    def meta():
        return jsonify({"device": device, "dev_label": dev_label,
                        "threshold": ACCEPT_THRESHOLD, "models": engine.status})

    @app.route("/infer", methods=["POST"])
    def infer():
        files = request.files.getlist("images")
        out = []
        for f in files:
            name = f.filename or "image"
            if not name.lower().endswith(IMG_EXTS):
                continue
            data = np.frombuffer(f.read(), np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                out.append({"name": name, "error": "unreadable", "thumb": "", "models": []})
                continue
            out.append({"name": os.path.basename(name), "thumb": _thumb(bgr),
                        "models": engine.read_image(bgr)})
        return jsonify({"results": out})

    return app


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meter OCR — inference-time benchmark</title>
<style>
  :root{--bg:#0f1115;--card:#1a1d24;--fg:#e8eaed;--mut:#9aa0aa;--line:#2a2e37;
        --acc:#3b82f6;--ok:#22c55e;--warn:#f59e0b;--bad:#ef4444;--gpu:#a855f7;}
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
  header{position:sticky;top:0;z-index:20;background:#12151b;border-bottom:1px solid var(--line);padding:14px 20px}
  h1{margin:0;font-size:17px;display:flex;align-items:center;gap:10px}
  .devchip{font-size:13px;font-weight:700;padding:3px 12px;border-radius:20px}
  .devchip.cpu{background:rgba(59,130,246,.18);color:var(--acc)}
  .devchip.gpu{background:rgba(168,85,247,.18);color:var(--gpu)}
  .models{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
  .mchip{font-size:12px;padding:5px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card);display:flex;gap:8px;align-items:center}
  .dot{width:8px;height:8px;border-radius:50%}
  .dot.on{background:var(--ok)} .dot.off{background:var(--bad)} .dot.fallback{background:var(--warn)}
  .mchip small{color:var(--mut)}
  .controls{margin-top:12px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  .btn{padding:9px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer;font-size:14px}
  .btn.primary{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
  .btn.dl{border-color:var(--ok);color:var(--ok)}
  .hint{color:var(--mut);font-size:12px}
  .summary{padding:14px 20px;display:flex;gap:14px;flex-wrap:wrap}
  .scard{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;min-width:180px}
  .scard b{font-size:22px} .scard .lbl{color:var(--mut);font-size:12px}
  .scard .avg{color:var(--gpu);font-family:ui-monospace,monospace}
  table{width:100%;border-collapse:collapse;margin-top:6px}
  th,td{padding:8px 10px;border-bottom:1px solid var(--line);font-size:13px;text-align:left;vertical-align:middle}
  th{color:var(--mut);font-weight:600;position:sticky;top:0;background:#12151b}
  td.serial{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:15px;font-weight:600;letter-spacing:.5px}
  .ms{font-family:ui-monospace,monospace;color:var(--gpu)}
  .sc{color:var(--mut);font-size:11px}
  .st{font-size:10px;font-weight:700;padding:1px 6px;border-radius:4px;margin-left:6px}
  .st.accepted{background:rgba(34,197,94,.18);color:var(--ok)}
  .st.review{background:rgba(245,158,11,.18);color:var(--warn)}
  .st.blank,.st.unavailable{background:rgba(107,114,128,.2);color:var(--mut)}
  .thumb{width:64px;height:auto;border-radius:6px;background:#000;cursor:zoom-in}
  .wrap{padding:0 20px 40px}
  #drop{margin:0 20px;border:2px dashed var(--line);border-radius:12px;padding:26px;text-align:center;color:var(--mut)}
  #drop.hover{border-color:var(--acc);color:var(--fg)}
  #bar{height:6px;background:#222833;border-radius:4px;overflow:hidden;margin:12px 20px 0;display:none}
  #bar>i{display:block;height:100%;width:0;background:var(--acc);transition:width .2s}
  .modelcol{border-left:1px solid var(--line)}
</style></head><body>
<header>
  <h1>Meter OCR — inference-time benchmark <span id="devchip" class="devchip">…</span></h1>
  <div class="models" id="models"></div>
  <div class="controls">
    <input id="fileInput" type="file" accept="image/*" multiple hidden>
    <input id="dirInput" type="file" webkitdirectory directory multiple hidden>
    <button class="btn primary" onclick="fileInput.click()">Upload image(s)</button>
    <button class="btn" onclick="dirInput.click()">Upload folder</button>
    <button class="btn dl" id="dlBtn" onclick="downloadCSV()" disabled>⬇ Download CSV</button>
    <span class="hint">All 3 models run on <b id="devName">…</b>. Times are per-image forward-pass ms.</span>
  </div>
</header>

<div id="drop">Drag &amp; drop an image or a folder here</div>
<div id="bar"><i></i></div>
<div class="summary" id="summary"></div>
<div class="wrap"><table id="tbl"><thead></thead><tbody></tbody></table></div>

<script>
let META=null, ROWS=[];
const el=id=>document.getElementById(id);

fetch('/meta').then(r=>r.json()).then(m=>{
  META=m;
  el('devchip').textContent=m.dev_label;
  el('devchip').className='devchip '+m.device;
  el('devName').textContent=m.dev_label;
  el('models').innerHTML=m.models.map(s=>{
    let cls='off',txt='unavailable';
    if(s.available){ if(s.device_used===m.device){cls='on';txt=s.device_used.toUpperCase();}
                     else {cls='fallback';txt=s.device_used.toUpperCase()+' (fallback)';} }
    return `<span class="mchip"><span class="dot ${cls}"></span><b>${s.label}</b>
            <small>${s.size} · ${s.runtime} · ${txt}</small>
            ${s.note?`<small title="${s.note}">ⓘ</small>`:''}</span>`;
  }).join('');
  buildHead();
});

function buildHead(){
  const th=document.querySelector('#tbl thead');
  let h='<tr><th>#</th><th>Image</th><th>File</th>';
  for(const s of META.models){ h+=`<th class="modelcol">${s.label}<br><small class="sc">${s.size} · ${s.runtime}</small></th>`; }
  h+='</tr>';
  th.innerHTML=h;
}

// ---- upload wiring ----
el('fileInput').onchange=e=>send(e.target.files);
el('dirInput').onchange=e=>send(e.target.files);
const drop=el('drop');
['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('hover');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('hover');}));
drop.addEventListener('drop',e=>{ if(e.dataTransfer.files.length) send(e.dataTransfer.files); });

async function send(fileList){
  const files=[...fileList].filter(f=>/\.(jpe?g|png|bmp|tif?f|webp)$/i.test(f.name));
  if(!files.length){ alert('No image files found.'); return; }
  ROWS=[]; document.querySelector('#tbl tbody').innerHTML=''; el('summary').innerHTML='';
  el('dlBtn').disabled=true;
  const bar=el('bar'); bar.style.display='block'; bar.firstElementChild.style.width='0';
  const CHUNK=25;                              // stream in batches so the UI updates live
  let done=0;
  for(let i=0;i<files.length;i+=CHUNK){
    const fd=new FormData();
    files.slice(i,i+CHUNK).forEach(f=>fd.append('images',f));
    const res=await fetch('/infer',{method:'POST',body:fd}).then(r=>r.json());
    for(const row of res.results){ ROWS.push(row); addRow(row); }
    done=Math.min(files.length,i+CHUNK);
    bar.firstElementChild.style.width=(done/files.length*100)+'%';
    renderSummary();
  }
  setTimeout(()=>bar.style.display='none',600);
  el('dlBtn').disabled=ROWS.length===0;
}

function addRow(row){
  const tb=document.querySelector('#tbl tbody');
  const tr=document.createElement('tr');
  let cells=`<td>${tb.children.length+1}</td>
             <td>${row.thumb?`<img class="thumb" src="${row.thumb}" onclick="window.open(this.src)">`:''}</td>
             <td>${row.name}</td>`;
  for(const s of META.models){
    const m=(row.models||[]).find(x=>x.key===s.key)||{};
    if(m.status==='unavailable' || m.ms===null || m.ms===undefined){
      cells+=`<td class="modelcol"><span class="st unavailable">n/a</span></td>`;
    }else{
      cells+=`<td class="modelcol serial">${m.serial||'<span class="sc">(blank)</span>'}
              <span class="st ${m.status}">${m.status}</span>
              <div class="sc">score ${m.score} · <span class="ms">${m.ms} ms</span></div></td>`;
    }
  }
  tr.innerHTML=cells; tb.appendChild(tr);
}

function renderSummary(){
  const agg={};
  for(const s of META.models) agg[s.key]={label:s.label,size:s.size,sum:0,n:0};
  for(const row of ROWS) for(const m of (row.models||[]))
    if(m.ms!=null){ agg[m.key].sum+=m.ms; agg[m.key].n++; }
  el('summary').innerHTML=`<div class="scard"><div class="lbl">Images</div><b>${ROWS.length}</b></div>`+
    META.models.map(s=>{const a=agg[s.key];const avg=a.n?(a.sum/a.n).toFixed(2):'—';
      return `<div class="scard"><div class="lbl">${a.label} · ${a.size}</div>
              <b class="avg">${avg} ms</b><div class="lbl">avg / image (n=${a.n})</div></div>`;}).join('');
}

function downloadCSV(){
  const esc=s=>'"'+String(s==null?'':s).replace(/"/g,'""')+'"';
  let head=['image','device'];
  for(const s of META.models){ head.push(s.key+'_serial',s.key+'_score',s.key+'_status',s.key+'_ms',s.key+'_device'); }
  let csv=head.join(',')+'\n';
  for(const row of ROWS){
    let line=[esc(row.name),esc(META.device)];
    for(const s of META.models){ const m=(row.models||[]).find(x=>x.key===s.key)||{};
      line.push(esc(m.serial),m.score??'',esc(m.status),m.ms??'',esc(m.device)); }
    csv+=line.join(',')+'\n';
  }
  const b=new Blob([csv],{type:'text/csv'});const a=document.createElement('a');
  a.href=URL.createObjectURL(b);a.download='inference_'+META.device+'.csv';a.click();
}
</script></body></html>
"""
