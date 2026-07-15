#!/usr/bin/env python3
"""
Build a head-to-head review page for two models (old vs new/retrained).

For each image it shows the crop and BOTH models' readings side by side, with a
Correct / Incorrect toggle for EACH model independently. A live monitor tracks
how many you've judged, and "Download results.csv" exports every judgment.

Input : results/model_compare.csv
        (image, image_path, old_reading, old_score, new_reading, new_score)
Output: results/model_compare.html

Usage:
  python3 scripts/build_compare_page.py \
      --csv results/model_compare.csv \
      --data "new dataset for test/tpcodl_tpsodl_roi_images" \
      --out results/model_compare.html
"""
import csv, os, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/model_compare.csv")
ap.add_argument("--data", default="new dataset for test/tpcodl_tpsodl_roi_images")
ap.add_argument("--out", default="results/model_compare.html")
ap.add_argument("--old-label", default="OLD", help="display label for the old_reading model")
ap.add_argument("--new-label", default="NEW", help="display label for the new_reading model")
ap.add_argument("--title", default="Old vs New Model — Head-to-Head")
ap.add_argument("--storage-key", default="model_compare_v1",
                help="localStorage key — use a unique one to avoid inheriting old marks")
args = ap.parse_args()

IMG_REL = os.path.relpath(args.data, os.path.dirname(args.out))

rows = list(csv.DictReader(open(args.csv)))
cards = []
for r in rows:
    img = r["image"]
    cards.append({
        "img": IMG_REL + "/" + img,
        "path": r.get("image_path") or os.path.abspath(os.path.join(args.data, img)),
        "name": img,
        "old": (r.get("old_reading") or "").strip(),
        "olds": r.get("old_score", ""),
        "new": (r.get("new_reading") or "").strip(),
        "news": r.get("new_score", ""),
        "search": (img + " " + (r.get("old_reading") or "") + " " + (r.get("new_reading") or "")).lower(),
    })
data_json = json.dumps(cards)

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Old vs New Model — Head-to-Head</title>
<style>
  :root{--bg:#0f1115;--card:#1a1d24;--line:#2a2f3a;--text:#e6e8ec;--muted:#9aa3b2;
        --good:#2ecc71;--bad:#e74c3c;--old:#4a9eff;--new:#b07cff;--focus:#ffcc00;}
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--text)}
  header{position:sticky;top:0;z-index:20;background:#12151b;border-bottom:1px solid var(--line);padding:14px 20px}
  h1{margin:0 0 8px;font-size:17px}
  .monitor{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--muted);align-items:center}
  .monitor b{color:var(--text)} .pill{padding:3px 10px;border-radius:20px;border:1px solid var(--line);background:var(--card)}
  .pill.old{color:var(--old)} .pill.new{color:var(--new)} .pill.left{color:#e67e22}
  .bar{height:6px;border-radius:4px;background:#222833;overflow:hidden;flex:1;min-width:140px}
  .bar>i{display:block;height:100%;width:0;background:var(--good);transition:width .2s}
  .controls{margin-top:10px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  input[type=search]{flex:1;min-width:200px;padding:9px 12px;border-radius:8px;border:1px solid var(--line);
        background:var(--card);color:var(--text);font-size:14px}
  .filters button{padding:7px 12px;border-radius:20px;border:1px solid var(--line);background:var(--card);
        color:var(--text);cursor:pointer;font-size:13px}
  .filters button.active{background:#2b3444;border-color:#3d4a60}
  .dl{padding:8px 14px;border-radius:8px;border:1px solid var(--good);background:var(--good);color:#052;font-weight:600;cursor:pointer;font-size:13px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;padding:20px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
  .card.done{outline:2px solid #2b3444}
  .card.focused{box-shadow:0 0 0 3px var(--focus),0 0 16px rgba(255,204,0,.35);border-color:var(--focus)}
  .kbdhint{margin-top:8px;font-size:12px;color:var(--muted);display:flex;gap:6px;flex-wrap:wrap;align-items:center}
  .kbdhint b{color:var(--text);background:#20293a;padding:1px 6px;border-radius:4px;border:1px solid var(--line);font-family:ui-monospace,monospace;font-weight:600}
  .imgwrap{background:#000;display:flex;align-items:center;justify-content:center;min-height:110px;padding:8px;cursor:zoom-in}
  .imgwrap img{max-width:100%;height:auto}
  .fname{padding:6px 12px 2px;font-size:11px;color:var(--muted);word-break:break-all}
  .mrow{display:flex;align-items:center;gap:8px;padding:6px 12px}
  .tag{font-size:10px;font-weight:700;padding:2px 7px;border-radius:5px;flex:none;min-width:58px;text-align:center;white-space:nowrap}
  .tag.old{background:rgba(74,158,255,.18);color:var(--old)} .tag.new{background:rgba(176,124,255,.18);color:var(--new)}
  .rd{flex:1;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:17px;font-weight:600;letter-spacing:.5px;word-break:break-all}
  .rd.empty{color:var(--muted);font-style:italic;font-size:13px;font-weight:400}
  .sc{font-size:10px;color:var(--muted);flex:none}
  .judge{display:flex;gap:6px;flex:none}
  .jb{width:30px;height:28px;border-radius:6px;border:1px solid var(--line);background:#20293a;color:var(--text);cursor:pointer;font-size:13px}
  .jb.ok.on{background:var(--good);color:#052;border-color:var(--good)}
  .jb.no.on{background:var(--bad);color:#310;border-color:var(--bad)}
  .diff{font-size:10px;color:#e67e22;padding:0 12px 8px}
  .cardbtns{padding:4px 12px 12px}
  .copybtn{font-size:11px;padding:5px 10px;border-radius:6px;border:1px solid var(--line);background:#20293a;color:var(--text);cursor:pointer}
  .copybtn.done{background:var(--good);color:#052;border-color:var(--good)}
  .lightbox{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.92);display:none;flex-direction:column;align-items:center;justify-content:center;gap:14px}
  .lightbox.open{display:flex}
  .lightbox img{max-width:94vw;max-height:80vh;transform:scale(var(--z,1));transform-origin:center;transition:transform .1s}
  .lb-bar{display:flex;gap:10px;align-items:center;color:var(--muted);font-size:13px;flex-wrap:wrap;justify-content:center;font-family:ui-monospace,monospace;padding:0 16px}
  .lb-bar button{padding:7px 13px;border-radius:8px;border:1px solid #3d4a60;background:#1a1d24;color:var(--text);cursor:pointer}
  .lb-path{max-width:70vw;word-break:break-all;color:#cbd3e1}
  .empty-msg{padding:40px;text-align:center;color:var(--muted)}
</style></head><body>
<header>
  <h1>__TITLE__</h1>
  <div class="monitor">
    <span>Judged <b id="m-done">0</b> / <b id="m-total">0</b></span>
    <span class="pill left">Left <b id="m-left">0</b></span>
    <span class="pill old">__OLDLBL__ correct <b id="m-old">0</b></span>
    <span class="pill new">__NEWLBL__ correct <b id="m-new">0</b></span>
    <div class="bar"><i id="m-fill"></i></div>
  </div>
  <div class="controls">
    <input type="search" id="q" placeholder="Search serial or filename…">
    <div class="filters">
      <button data-f="all" class="active">All</button>
      <button data-f="unjudged">◻ Not judged</button>
      <button data-f="differ">Models differ</button>
      <button data-f="agree">Models agree</button>
    </div>
    <button class="dl" id="dl">⬇ Download results.csv</button>
  </div>
  <div class="kbdhint">⌨
    <span><b>←</b>/<b>→</b> (or <b>↑</b>/<b>↓</b>) move</span>·
    <span><b>1</b>/<b>2</b> __OLDLBL__ ✓/✗</span>·
    <span><b>3</b>/<b>4</b> __NEWLBL__ ✓/✗</span>·
    <span><b>5</b> both ✓</span><span><b>6</b> both ✗</span>·
    <span><b>Enter</b> zoom</span><span><b>Esc</b> close zoom</span>
  </div>
</header>
<div class="grid" id="grid"></div>
<div class="empty-msg" id="empty" style="display:none">No matches.</div>

<div class="lightbox" id="lb"><img id="lbimg" alt="">
  <div class="lb-bar"><button id="zout">– zoom</button><button id="zin">+ zoom</button><button id="zreset">reset</button>
  <span class="lb-path" id="lbpath"></span><button id="lbcopy">Copy path</button><button id="lbclose">✕ close</button></div>
</div>

<script>
const DATA = __DATA__;
const KEY = "__STORAGEKEY__";                 // name -> {o:'ok'|'no', n:'ok'|'no'}
const FB = JSON.parse(localStorage.getItem(KEY) || "{}");
const grid=document.getElementById('grid'), q=document.getElementById('q');
let filter='all';
let SHOWN=[], focusIdx=-1;      // {c,el} currently rendered, and the focused one
function save(){ localStorage.setItem(KEY, JSON.stringify(FB)); }
function judged(c){ const f=FB[c.name]; return f && f.o && f.n; }   // both models judged

function updateMonitor(){
  const total=DATA.length; let done=0, oldok=0, newok=0;
  for(const c of DATA){ const f=FB[c.name]; if(!f) continue;
    if(f.o&&f.n) done++;
    if(f.o==='ok') oldok++; if(f.n==='ok') newok++; }
  document.getElementById('m-total').textContent=total;
  document.getElementById('m-done').textContent=done;
  document.getElementById('m-left').textContent=total-done;
  document.getElementById('m-old').textContent=oldok;
  document.getElementById('m-new').textContent=newok;
  document.getElementById('m-fill').style.width=total?(done/total*100)+'%':'0';
}
function copyText(txt,btn,label){const done=()=>{const t=btn.textContent;btn.textContent=label;btn.classList.add('done');
  setTimeout(()=>{btn.textContent=t;btn.classList.remove('done');},1200);};
  if(navigator.clipboard)navigator.clipboard.writeText(txt).then(done).catch(fb);else fb();
  function fb(){const t=document.createElement('textarea');t.value=txt;document.body.appendChild(t);t.select();try{document.execCommand('copy');done();}catch(e){}document.body.removeChild(t);}}

const lb=document.getElementById('lb'),lbimg=document.getElementById('lbimg'),lbpath=document.getElementById('lbpath');
let zoom=3,curPath='';
function setZoom(z){zoom=Math.min(12,Math.max(1,z));lbimg.style.setProperty('--z',zoom);}
function openLB(c){curPath=c.path;lbimg.src=c.img;lbpath.textContent=c.path;setZoom(3);lb.classList.add('open');}
document.getElementById('zin').onclick=()=>setZoom(zoom+1);document.getElementById('zout').onclick=()=>setZoom(zoom-1);
document.getElementById('zreset').onclick=()=>setZoom(3);document.getElementById('lbclose').onclick=()=>lb.classList.remove('open');
document.getElementById('lbcopy').onclick=e=>copyText(curPath,e.target,'Copied ✓');
lb.addEventListener('click',e=>{if(e.target===lb)lb.classList.remove('open');});
lb.addEventListener('wheel',e=>{e.preventDefault();setZoom(zoom+(e.deltaY<0?1:-1));},{passive:false});
document.addEventListener('keydown',e=>{if(e.key==='Escape')lb.classList.remove('open');});

function passes(c){
  const f=FB[c.name];
  if(filter==='unjudged') return !judged(c);
  if(filter==='differ') return (c.old||'').toUpperCase()!==(c.new||'').toUpperCase();
  if(filter==='agree') return (c.old||'').toUpperCase()===(c.new||'').toUpperCase();
  return true;
}
function mrow(tag,cls,reading,score,which,f){
  const rd = reading ? '<span class="rd">'+reading+'</span>'
                     : '<span class="rd empty">(blank)</span>';
  const cur = f ? f[which] : '';
  return '<div class="mrow" data-w="'+which+'">'+
    '<span class="tag '+cls+'">'+tag+'</span>'+rd+
    '<span class="sc">'+(score||'')+'</span>'+
    '<span class="judge">'+
      '<button class="jb ok'+(cur==='ok'?' on':'')+'" title="correct">✓</button>'+
      '<button class="jb no'+(cur==='no'?' on':'')+'" title="incorrect">✗</button>'+
    '</span></div>';
}
function render(){
  const term=q.value.trim().toLowerCase(); grid.innerHTML=''; SHOWN=[];
  for(const c of DATA){
    if(!passes(c)) continue;
    if(term && !c.search.includes(term)) continue;
    const f=FB[c.name];
    const differ=(c.old||'').toUpperCase()!==(c.new||'').toUpperCase();
    const el=document.createElement('div');
    el.className='card'+(judged(c)?' done':'');
    el.innerHTML=
      '<div class="imgwrap" title="click to zoom"><img loading="lazy" src="'+c.img+'"></div>'+
      '<div class="fname">'+c.name+'</div>'+
      mrow('__OLDLBL__','old',c.old,c.olds,'o',f)+
      mrow('__NEWLBL__','new',c.new,c.news,'n',f)+
      (differ?'<div class="diff">models differ</div>':'')+
      '<div class="cardbtns"><button class="copybtn">Copy path</button></div>';
    el.querySelector('.imgwrap').onclick=()=>openLB(c);
    el.querySelector('.copybtn').onclick=ev=>copyText(c.path,ev.target,'Copied ✓');
    el.querySelectorAll('.mrow').forEach(row=>{
      const which=row.dataset.w;
      row.querySelector('.ok').onclick=()=>setJudge(c,which,'ok',el);
      row.querySelector('.no').onclick=()=>setJudge(c,which,'no',el);
    });
    const idx=SHOWN.length;
    el.addEventListener('click',()=>{focusIdx=idx;applyFocus(false);});
    grid.appendChild(el);
    SHOWN.push({c,el});
  }
  document.getElementById('empty').style.display=SHOWN.length?'none':'block';
  if(focusIdx>=SHOWN.length) focusIdx=SHOWN.length-1;
  if(focusIdx<0 && SHOWN.length) focusIdx=0;
  applyFocus(false);
}
// ---- keyboard focus + judging ----
function applyFocus(scroll){
  SHOWN.forEach((s,i)=>s.el.classList.toggle('focused', i===focusIdx));
  if(scroll!==false && focusIdx>=0 && SHOWN[focusIdx])
    SHOWN[focusIdx].el.scrollIntoView({block:'nearest',behavior:'smooth'});
}
function moveFocus(d){
  if(!SHOWN.length) return;
  focusIdx = focusIdx<0 ? 0 : Math.min(SHOWN.length-1, Math.max(0, focusIdx+d));
  applyFocus(true);
}
function cols(){                                   // cards per row in the responsive grid
  if(SHOWN.length<2) return 1;
  const y0=SHOWN[0].el.offsetTop; let n=1;
  for(let i=1;i<SHOWN.length;i++){ if(SHOWN[i].el.offsetTop===y0) n++; else break; }
  return n;
}
function judgeFocused(which,val){
  if(focusIdx<0||!SHOWN[focusIdx]) return;
  const {c,el}=SHOWN[focusIdx];
  setJudge(c,which,val,el);
  const f=FB[c.name];
  if(f&&f.o&&f.n) setTimeout(()=>moveFocus(1),140);   // auto-advance once fully judged
}
document.addEventListener('keydown',e=>{
  if(lb.classList.contains('open')) return;             // lightbox owns keys while open
  const t=e.target;
  if(t && (t.tagName==='INPUT'||t.tagName==='TEXTAREA')) return;   // don't hijack the search box
  let handled=true;
  switch(e.key){
    case 'ArrowRight': case 'l': moveFocus(1); break;
    case 'ArrowLeft':  case 'h': moveFocus(-1); break;
    case 'ArrowDown':  case 'j': moveFocus(cols()); break;
    case 'ArrowUp':    case 'k': moveFocus(-cols()); break;
    case '1': judgeFocused('o','ok'); break;
    case '2': judgeFocused('o','no'); break;
    case '3': judgeFocused('n','ok'); break;
    case '4': judgeFocused('n','no'); break;
    case '5': judgeFocused('o','ok'); judgeFocused('n','ok'); break;
    case '6': judgeFocused('o','no'); judgeFocused('n','no'); break;
    case 'Enter': if(SHOWN[focusIdx]) openLB(SHOWN[focusIdx].c); break;
    default: handled=false;
  }
  if(handled) e.preventDefault();
});
function setJudge(c,which,val,el){
  const f=FB[c.name]||{}; f[which]=val; FB[c.name]=f; save();
  // update button states within this row
  el.querySelectorAll('.mrow').forEach(row=>{
    if(row.dataset.w!==which) return;
    row.querySelector('.ok').classList.toggle('on',val==='ok');
    row.querySelector('.no').classList.toggle('on',val==='no');
  });
  el.classList.toggle('done',judged(c));
  updateMonitor();
}
q.addEventListener('input',render);
document.querySelectorAll('.filters button').forEach(b=>{b.onclick=()=>{
  document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');filter=b.dataset.f;render();};});

document.getElementById('dl').onclick=()=>{
  const esc=s=>'"'+(s||"").replace(/"/g,'""')+'"';
  let csv="image,image_path,old_reading,old_score,new_reading,new_score,old_verdict,new_verdict\n";
  for(const c of DATA){
    const f=FB[c.name]||{};
    const ov=f.o==='ok'?'correct':(f.o==='no'?'incorrect':'');
    const nv=f.n==='ok'?'correct':(f.n==='no'?'incorrect':'');
    csv+=[esc(c.name),esc(c.path),esc(c.old),c.olds,esc(c.new),c.news,esc(ov),esc(nv)].join(",")+"\n";
  }
  const b=new Blob([csv],{type:"text/csv"});const a=document.createElement('a');
  a.href=URL.createObjectURL(b);a.download="model_compare_results.csv";a.click();
};
updateMonitor(); render();
</script></body></html>
"""
out = (HTML.replace("__DATA__", data_json)
           .replace("__OLDLBL__", args.old_label)
           .replace("__NEWLBL__", args.new_label)
           .replace("__TITLE__", args.title)
           .replace("__STORAGEKEY__", args.storage_key))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as fh:
    fh.write(out)
print(f"Wrote {args.out}  ({len(cards)} images)")
