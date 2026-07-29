#!/usr/bin/env python3
"""Interactive REVIEW dashboard for the 175 N->W confusion images.

Lets the user correct each true serial, tracks review progress, and exports the
confirmed labels to a CSV (downloaded by the browser). Corrections persist in
localStorage so progress survives reloads.

Output: analysis/N_to_W_confusion_test/review_dashboard.html
"""
import csv, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "analysis", "N_to_W_confusion_test")
LABELS = os.path.join(DIR, "labels.csv")
OUT = os.path.join(DIR, "review_dashboard.html")

rows = []
for r in csv.DictReader(open(LABELS)):
    rows.append({
        "f": r["filename"],
        "t": r["true_serial"],
        "p": r["new_ptq_pred"],
        "o": r["old_v2_pred"],
        "q": r["qat_pred"],
        "clean": r["clean_N_to_W"],
    })
# clean cases first
rows.sort(key=lambda x: (x["clean"] != "yes", x["f"]))
total = len(rows)

html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>N&rarr;W Review &mdash; correct &amp; confirm serials</title>
<style>
:root{--bg:#0f1115;--card:#1a1d24;--line:#2a2e37;--txt:#e6e8ec;--mut:#9aa2af;
--ok:#2ecc71;--warn:#f1c40f;--err:#e74c3c;--accent:#4aa3ff;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--txt)}
.bar{position:sticky;top:0;z-index:10;background:#12151b;border-bottom:1px solid var(--line);padding:14px 22px}
.bar h1{margin:0 0 8px;font-size:17px}
.track{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.pbar{flex:1;min-width:180px;height:12px;background:#0b0d11;border:1px solid var(--line);border-radius:20px;overflow:hidden}
.pbar>i{display:block;height:100%;width:0;background:var(--ok);transition:width .2s}
.stat{font-size:13px;color:var(--mut)}
.stat b{color:var(--txt)}
button{background:var(--card);border:1px solid var(--line);color:var(--txt);padding:8px 13px;border-radius:8px;cursor:pointer;font-size:13px}
button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
button.good{background:var(--ok);border-color:var(--ok);color:#062}
.filters{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}
.filters .f{padding:6px 12px;font-size:12.5px}
.filters .f.active{background:var(--accent);border-color:var(--accent);color:#fff}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;padding:18px 22px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.tile.done{border-color:var(--ok);box-shadow:0 0 0 1px var(--ok) inset}
.tile img{width:100%;height:150px;object-fit:contain;background:#000;border-bottom:1px solid var(--line);cursor:zoom-in}
.body{padding:11px 13px}
.tag{display:inline-block;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;margin-bottom:6px}
.tag.clean{background:var(--ok);color:#062}.tag.messy{background:var(--warn);color:#111}
.kv{display:flex;justify-content:space-between;gap:8px;font-size:12px;padding:1px 0}
.kv .k{color:var(--mut)}.kv .v{font-family:ui-monospace,Menlo,monospace}
.kv .v.err{color:var(--err)}
.lbl{display:block;color:var(--mut);font-size:11px;margin:9px 0 3px}
.row{display:flex;gap:7px}
.row input{flex:1;min-width:0;background:#0b0d11;border:1px solid var(--line);color:var(--txt);border-radius:7px;padding:8px 9px;font:13px ui-monospace,Menlo,monospace;text-transform:uppercase}
.row input:focus{outline:none;border-color:var(--accent)}
.confirm{white-space:nowrap;padding:8px 12px}
.confirm.on{background:var(--ok);border-color:var(--ok);color:#062;font-weight:700}
.more{display:block;margin:8px auto 40px;padding:11px 26px}
.hint{color:var(--mut);font-size:12px;padding:0 22px}
#lightbox{position:fixed;inset:0;background:rgba(0,0,0,.9);display:none;align-items:center;justify-content:center;z-index:50;cursor:zoom-out}
#lightbox img{max-width:94%;max-height:94%}
</style></head><body>
<div class="bar">
  <h1>N&rarr;W Review &mdash; correct the serial, then Confirm</h1>
  <div class="track">
    <div class="pbar"><i id="pfill"></i></div>
    <div class="stat">Confirmed <b id="ndone">0</b> / __TOTAL__ &nbsp;(<span id="ppct">0</span>%) &middot; remaining <b id="nrem">__TOTAL__</b></div>
    <button class="good" onclick="confirmVisible()">✓ Confirm all visible</button>
    <button class="primary" onclick="exportCSV()">⭳ Export confirmed CSV</button>
    <button onclick="if(confirm('Clear all your edits & confirmations?')){localStorage.removeItem(KEY);location.reload();}">Reset</button>
  </div>
  <div class="filters">
    <button class="f active" data-f="all" onclick="setF('all',this)">All __TOTAL__</button>
    <button class="f" data-f="clean" onclick="setF('clean',this)">Clean N&rarr;W __CLEAN__</button>
    <button class="f" data-f="messy" onclick="setF('messy',this)">Messy __MESSY__</button>
    <button class="f" data-f="todo" onclick="setF('todo',this)">Unconfirmed</button>
    <button class="f" data-f="done" onclick="setF('done',this)">Confirmed</button>
  </div>
</div>
<div class="hint">Edit the <b>Corrected serial</b> box (prefilled with the FP32 read). Click <b>Confirm</b> when right. Click an image to zoom. Progress auto-saves in this browser; use <b>Export</b> to send me the file.</div>
<div id="grid" class="grid"></div>
<button id="more" class="more" onclick="render()">Load more</button>
<div id="lightbox" onclick="this.style.display='none'"><img id="lbimg" src=""></div>
<script>
const DATA = __DATA__;
const KEY = "ntow_review_v1";
let state = JSON.parse(localStorage.getItem(KEY) || "{}");   // {filename:{v,c}}
let filt='all', shown=0, PAGE=48, view=[];

function save(){ localStorage.setItem(KEY, JSON.stringify(state)); updateTrack(); }
function st(f){ return state[f] || {v:null, c:false}; }
function valOf(r){ const s=st(r.f); return s.v!==null? s.v : r.t; }

function updateTrack(){
  let done=DATA.filter(r=>st(r.f).c).length;
  document.getElementById('ndone').textContent=done;
  document.getElementById('nrem').textContent=DATA.length-done;
  let pc=Math.round(done/DATA.length*100);
  document.getElementById('ppct').textContent=pc;
  document.getElementById('pfill').style.width=pc+'%';
}
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function matchF(r){
  const c=st(r.f).c;
  if(filt==='all')return true;
  if(filt==='clean')return r.clean==='yes';
  if(filt==='messy')return r.clean!=='yes';
  if(filt==='todo')return !c;
  if(filt==='done')return c;
}
function setF(f,el){
  filt=f; shown=0;
  document.querySelectorAll('.filters .f').forEach(b=>b.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('grid').innerHTML='';
  view=DATA.filter(matchF); render();
}
function onEdit(f,inp){ const s=st(f); s.v=inp.value.toUpperCase(); state[f]=s; save(); }
function toggle(f,btn){
  const s=st(f); s.c=!s.c; state[f]=s; save();
  const tile=btn.closest('.tile');
  btn.classList.toggle('on',s.c); btn.textContent=s.c?'✓ Confirmed':'Confirm';
  tile.classList.toggle('done',s.c);
}
function confirmVisible(){
  view.forEach(r=>{const s=st(r.f); s.c=true; if(s.v===null)s.v=r.t; state[r.f]=s;});
  save();
  document.getElementById('grid').innerHTML=''; shown=0; view=DATA.filter(matchF); render();
}
function zoom(src){ document.getElementById('lbimg').src=src; document.getElementById('lightbox').style.display='flex'; }
function render(){
  const g=document.getElementById('grid');
  for(const r of view.slice(shown,shown+PAGE)){
    const s=st(r.f);
    const src='images/'+encodeURIComponent(r.f);
    const div=document.createElement('div'); div.className='tile'+(s.c?' done':'');
    div.innerHTML=
      '<img loading="lazy" src="'+src+'" onclick="zoom(this.src)">'+
      '<div class="body">'+
        '<span class="tag '+(r.clean==='yes'?'clean':'messy')+'">'+(r.clean==='yes'?'CLEAN N→W':'MESSY')+'</span>'+
        '<div class="kv"><span class="k">FP32 (truth)</span><span class="v">'+esc(r.t)+'</span></div>'+
        '<div class="kv"><span class="k">New PTQ</span><span class="v err">'+esc(r.p)+'</span></div>'+
        '<div class="kv"><span class="k">Old / QAT</span><span class="v">'+esc(r.o)+' / '+esc(r.q)+'</span></div>'+
        '<span class="lbl">Corrected serial</span>'+
        '<div class="row">'+
          '<input value="'+esc(valOf(r))+'" oninput="onEdit(\\''+r.f.replace(/'/g,"&#39;")+'\\',this)">'+
          '<button class="confirm'+(s.c?' on':'')+'" onclick="toggle(\\''+r.f.replace(/'/g,"&#39;")+'\\',this)">'+(s.c?'✓ Confirmed':'Confirm')+'</button>'+
        '</div>'+
      '</div>';
    g.appendChild(div);
  }
  shown+=Math.min(PAGE, view.length-shown);
  document.getElementById('more').style.display= shown<view.length?'block':'none';
}
function exportCSV(){
  let out=[['filename','corrected_serial','new_ptq_pred','confirmed']];
  DATA.forEach(r=>{const s=st(r.f); out.push([r.f, valOf(r), r.p, s.c?'yes':'no']);});
  let csv=out.map(row=>row.map(c=>'"'+String(c).replace(/"/g,'""')+'"').join(',')).join('\\n');
  let blob=new Blob([csv],{type:'text/csv'});
  let a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download='ntow_corrected_labels.csv'; a.click();
}
setF('all', document.querySelector('.filters .f'));
updateTrack();
</script>
</body></html>"""

html = (html
        .replace("__DATA__", json.dumps(rows, separators=(",", ":")))
        .replace("__TOTAL__", str(total))
        .replace("__CLEAN__", str(sum(1 for r in rows if r["clean"] == "yes")))
        .replace("__MESSY__", str(sum(1 for r in rows if r["clean"] != "yes"))))

with open(OUT, "w") as f:
    f.write(html)
print("Wrote:", OUT)
print(f"total={total} clean={sum(1 for r in rows if r['clean']=='yes')} messy={sum(1 for r in rows if r['clean']!='yes')}")
