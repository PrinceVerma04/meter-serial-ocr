#!/usr/bin/env python3
"""
Head-to-head review page for the 3 models (FP32 original / QAT INT8 / PTQ INT8).
Shows each crop with all three readings, a Correct/Incorrect toggle per model,
a live monitor, and a CSV export.

Input : size_reduction/models_compare.csv
Output: size_reduction/models_compare.html
"""
import csv, os, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="size_reduction/models_compare.csv")
ap.add_argument("--data", default="new dataset for test/tpcodl_tpsodl_roi_images")
ap.add_argument("--out", default="size_reduction/models_compare.html")
args = ap.parse_args()

IMG_REL = os.path.relpath(args.data, os.path.dirname(args.out))
rows = list(csv.DictReader(open(args.csv)))
cards = []
for r in rows:
    img = r["image"]
    cards.append({
        "img": IMG_REL + "/" + img, "name": img,
        "path": r.get("image_path") or os.path.abspath(os.path.join(args.data, img)),
        "fp32": (r.get("fp32_reading") or "").strip(), "fps": r.get("fp32_score", ""),
        "qat": (r.get("qat_reading") or "").strip(),  "qs": r.get("qat_score", ""),
        "ptq": (r.get("ptq_reading") or "").strip(),  "ps": r.get("ptq_score", ""),
        "search": (img + " " + (r.get("fp32_reading") or "") + " " + (r.get("qat_reading") or "") + " " + (r.get("ptq_reading") or "")).lower(),
    })
data_json = json.dumps(cards)

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3-Model Comparison — FP32 vs QAT vs PTQ</title><style>
  :root{--bg:#0f1115;--card:#1a1d24;--line:#2a2f3a;--text:#e6e8ec;--muted:#9aa3b2;
        --good:#2ecc71;--bad:#e74c3c;--fp:#4a9eff;--qat:#2ecc71;--ptq:#b07cff;}
  *{box-sizing:border-box}body{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--text)}
  header{position:sticky;top:0;z-index:20;background:#12151b;border-bottom:1px solid var(--line);padding:14px 20px}
  h1{margin:0 0 8px;font-size:17px}
  .monitor{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--muted);align-items:center}
  .monitor b{color:var(--text)}.pill{padding:3px 10px;border-radius:20px;border:1px solid var(--line);background:var(--card)}
  .pill.fp{color:var(--fp)}.pill.qat{color:var(--qat)}.pill.ptq{color:var(--ptq)}.pill.left{color:#e67e22}
  .bar{height:6px;border-radius:4px;background:#222833;overflow:hidden;flex:1;min-width:120px}.bar>i{display:block;height:100%;width:0;background:var(--good);transition:width .2s}
  .controls{margin-top:10px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  input[type=search]{flex:1;min-width:200px;padding:9px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--text);font-size:14px}
  .filters button{padding:7px 12px;border-radius:20px;border:1px solid var(--line);background:var(--card);color:var(--text);cursor:pointer;font-size:13px}
  .filters button.active{background:#2b3444;border-color:#3d4a60}
  .dl{padding:8px 14px;border-radius:8px;border:1px solid var(--good);background:var(--good);color:#052;font-weight:600;cursor:pointer;font-size:13px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px;padding:20px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
  .card.done{outline:2px solid #2b3444}
  .imgwrap{background:#000;display:flex;align-items:center;justify-content:center;min-height:110px;padding:8px;cursor:zoom-in}
  .imgwrap img{max-width:100%;height:auto}.fname{padding:6px 12px 2px;font-size:11px;color:var(--muted);word-break:break-all}
  .mrow{display:flex;align-items:center;gap:8px;padding:5px 12px}
  .tag{font-size:9px;font-weight:700;padding:2px 6px;border-radius:5px;flex:none;min-width:74px;text-align:center;white-space:nowrap}
  .tag.fp{background:rgba(74,158,255,.18);color:var(--fp)}.tag.qat{background:rgba(46,204,113,.18);color:var(--qat)}.tag.ptq{background:rgba(176,124,255,.18);color:var(--ptq)}
  .rd{flex:1;font-family:ui-monospace,monospace;font-size:16px;font-weight:600;letter-spacing:.5px;word-break:break-all}
  .rd.empty{color:var(--muted);font-style:italic;font-size:12px;font-weight:400}
  .sc{font-size:10px;color:var(--muted);flex:none}.judge{display:flex;gap:5px;flex:none}
  .jb{width:28px;height:26px;border-radius:6px;border:1px solid var(--line);background:#20293a;color:var(--text);cursor:pointer;font-size:12px}
  .jb.ok.on{background:var(--good);color:#052;border-color:var(--good)}.jb.no.on{background:var(--bad);color:#310;border-color:var(--bad)}
  .cardbtns{padding:6px 12px 12px}.copybtn{font-size:11px;padding:5px 10px;border-radius:6px;border:1px solid var(--line);background:#20293a;color:var(--text);cursor:pointer}
  .copybtn.done{background:var(--good);color:#052}
  .lightbox{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.92);display:none;flex-direction:column;align-items:center;justify-content:center;gap:14px}
  .lightbox.open{display:flex}.lightbox img{max-width:94vw;max-height:80vh;transform:scale(var(--z,1));transform-origin:center}
  .lb-bar{display:flex;gap:10px;align-items:center;color:var(--muted);font-size:13px;flex-wrap:wrap;justify-content:center;font-family:ui-monospace,monospace;padding:0 16px}
  .lb-bar button{padding:7px 13px;border-radius:8px;border:1px solid #3d4a60;background:#1a1d24;color:var(--text);cursor:pointer}
  .lb-path{max-width:70vw;word-break:break-all;color:#cbd3e1}.empty-msg{padding:40px;text-align:center;color:var(--muted)}
</style></head><body>
<header><h1>3-Model Comparison — FP32 (81MB) · QAT INT8 (22MB) · PTQ INT8 (21MB) — 500 images</h1>
  <div class="monitor">
    <span>Judged <b id="m-done">0</b>/<b id="m-total">0</b></span>
    <span class="pill left">Left <b id="m-left">0</b></span>
    <span class="pill fp">FP32 ✓<b id="m-fp">0</b></span>
    <span class="pill qat">QAT ✓<b id="m-qat">0</b></span>
    <span class="pill ptq">PTQ ✓<b id="m-ptq">0</b></span>
    <div class="bar"><i id="m-fill"></i></div>
  </div>
  <div class="controls">
    <input type="search" id="q" placeholder="Search serial or filename…">
    <div class="filters">
      <button data-f="all" class="active">All</button>
      <button data-f="unjudged">◻ Not judged</button>
      <button data-f="differ">Models differ</button>
    </div>
    <button class="dl" id="dl">⬇ Download results.csv</button>
  </div></header>
<div class="grid" id="grid"></div><div class="empty-msg" id="empty" style="display:none">No matches.</div>
<div class="lightbox" id="lb"><img id="lbimg" alt="">
  <div class="lb-bar"><button id="zout">– zoom</button><button id="zin">+ zoom</button><button id="zreset">reset</button>
  <span class="lb-path" id="lbpath"></span><button id="lbcopy">Copy path</button><button id="lbclose">✕ close</button></div></div>
<script>
const DATA=__DATA__, KEY="models3_compare_v1";
const FB=JSON.parse(localStorage.getItem(KEY)||"{}");   // name -> {fp,qat,ptq: 'ok'|'no'}
const grid=document.getElementById('grid'), q=document.getElementById('q'); let filter='all';
function save(){localStorage.setItem(KEY,JSON.stringify(FB));}
function judged(c){const f=FB[c.name];return f&&f.fp&&f.qat&&f.ptq;}
function updateMonitor(){
  let done=0,fp=0,qat=0,ptq=0;
  for(const c of DATA){const f=FB[c.name];if(!f)continue;
    if(f.fp&&f.qat&&f.ptq)done++; if(f.fp==='ok')fp++; if(f.qat==='ok')qat++; if(f.ptq==='ok')ptq++;}
  document.getElementById('m-total').textContent=DATA.length;
  document.getElementById('m-done').textContent=done;
  document.getElementById('m-left').textContent=DATA.length-done;
  document.getElementById('m-fp').textContent=fp;document.getElementById('m-qat').textContent=qat;document.getElementById('m-ptq').textContent=ptq;
  document.getElementById('m-fill').style.width=(done/DATA.length*100)+'%';
}
function copyText(t,b,l){const d=()=>{const o=b.textContent;b.textContent=l;b.classList.add('done');setTimeout(()=>{b.textContent=o;b.classList.remove('done')},1200)};
  if(navigator.clipboard)navigator.clipboard.writeText(t).then(d).catch(fb);else fb();function fb(){const x=document.createElement('textarea');x.value=t;document.body.appendChild(x);x.select();try{document.execCommand('copy');d()}catch(e){}document.body.removeChild(x)}}
const lb=document.getElementById('lb'),lbimg=document.getElementById('lbimg'),lbpath=document.getElementById('lbpath');let zoom=3,cur='';
function setZoom(z){zoom=Math.min(12,Math.max(1,z));lbimg.style.setProperty('--z',zoom)}
function openLB(c){cur=c.path;lbimg.src=c.img;lbpath.textContent=c.path;setZoom(3);lb.classList.add('open')}
document.getElementById('zin').onclick=()=>setZoom(zoom+1);document.getElementById('zout').onclick=()=>setZoom(zoom-1);
document.getElementById('zreset').onclick=()=>setZoom(3);document.getElementById('lbclose').onclick=()=>lb.classList.remove('open');
document.getElementById('lbcopy').onclick=e=>copyText(cur,e.target,'Copied ✓');
lb.addEventListener('click',e=>{if(e.target===lb)lb.classList.remove('open')});
lb.addEventListener('wheel',e=>{e.preventDefault();setZoom(zoom+(e.deltaY<0?1:-1))},{passive:false});
document.addEventListener('keydown',e=>{if(e.key==='Escape')lb.classList.remove('open')});
function passes(c){const f=FB[c.name];
  if(filter==='unjudged')return !judged(c);
  if(filter==='differ'){const u=new Set([c.fp,c.qat,c.ptq].map(x=>(x||'').toUpperCase()));return u.size>1;}
  return true;}
function mrow(tag,cls,reading,score,which,f){
  const rd=reading?'<span class="rd">'+reading+'</span>':'<span class="rd empty">(blank)</span>';
  const cur=f?f[which]:'';
  return '<div class="mrow" data-w="'+which+'"><span class="tag '+cls+'">'+tag+'</span>'+rd+
    '<span class="sc">'+(score||'')+'</span><span class="judge">'+
    '<button class="jb ok'+(cur==='ok'?' on':'')+'" title="correct">✓</button>'+
    '<button class="jb no'+(cur==='no'?' on':'')+'" title="incorrect">✗</button></span></div>';
}
function render(){const term=q.value.trim().toLowerCase();grid.innerHTML='';let shown=0;
  for(const c of DATA){if(!passes(c))continue;if(term&&!c.search.includes(term))continue;shown++;
    const f=FB[c.name];const el=document.createElement('div');el.className='card'+(judged(c)?' done':'');
    el.innerHTML='<div class="imgwrap" title="click to zoom"><img loading="lazy" src="'+c.img+'"></div>'+
      '<div class="fname">'+c.name+'</div>'+
      mrow('FP32 81M','fp',c.fp32,c.fps,'fp',f)+
      mrow('QAT 22M','qat',c.qat,c.qs,'qat',f)+
      mrow('PTQ 21M','ptq',c.ptq,c.ps,'ptq',f)+
      '<div class="cardbtns"><button class="copybtn">Copy path</button></div>';
    el.querySelector('.imgwrap').onclick=()=>openLB(c);
    el.querySelector('.copybtn').onclick=ev=>copyText(c.path,ev.target,'Copied ✓');
    el.querySelectorAll('.mrow').forEach(row=>{const w=row.dataset.w;
      row.querySelector('.ok').onclick=()=>setJudge(c,w,'ok',el);
      row.querySelector('.no').onclick=()=>setJudge(c,w,'no',el);});
    grid.appendChild(el);}
  document.getElementById('empty').style.display=shown?'none':'block';}
function setJudge(c,which,val,el){const f=FB[c.name]||{};f[which]=val;FB[c.name]=f;save();
  el.querySelectorAll('.mrow').forEach(row=>{if(row.dataset.w!==which)return;
    row.querySelector('.ok').classList.toggle('on',val==='ok');row.querySelector('.no').classList.toggle('on',val==='no');});
  el.classList.toggle('done',judged(c));updateMonitor();}
q.addEventListener('input',render);
document.querySelectorAll('.filters button').forEach(b=>{b.onclick=()=>{document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('active'));b.classList.add('active');filter=b.dataset.f;render();};});
document.getElementById('dl').onclick=()=>{const esc=s=>'"'+(s||"").replace(/"/g,'""')+'"';
  let csv="image,image_path,fp32_reading,qat_reading,ptq_reading,fp32_verdict,qat_verdict,ptq_verdict\n";
  for(const c of DATA){const f=FB[c.name]||{};
    const v=x=>x==='ok'?'correct':(x==='no'?'incorrect':'');
    csv+=[esc(c.name),esc(c.path),esc(c.fp32),esc(c.qat),esc(c.ptq),esc(v(f.fp)),esc(v(f.qat)),esc(v(f.ptq))].join(",")+"\n";}
  const b=new Blob([csv],{type:"text/csv"});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download="models3_results.csv";a.click();};
updateMonitor();render();
</script></body></html>"""
out = HTML.replace("__DATA__", data_json)
with open(args.out, "w") as fh:
    fh.write(out)
print(f"Wrote {args.out} ({len(cards)} images)")
