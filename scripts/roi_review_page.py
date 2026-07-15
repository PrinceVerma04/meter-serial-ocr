#!/usr/bin/env python3
"""
Build a self-contained review/verification page for an OCR run.

Turns an extract_serials.py output CSV into results/roi_review.html — a local
web page where a human checks every prediction:

  * shows each crop (click to zoom) with the OCR model's reading
  * copyable FULL image path (one click)
  * three verdict buttons per image:  ✓ Correct   ✗ Wrong   🚫 Not visible
  * when Wrong, type the true serial in the correction box (or mark unreadable)
  * "Not visible" flags a poor-quality crop to be moved to a separate folder
    (run scripts/move_not_visible.py on the exported CSV to actually move them)
  * live monitor: how many checked / not-checked / correct / wrong / not-visible
  * "Download results.csv" exports every verdict + your corrections (full audit)
  * "Download training-labels.csv" exports your corrections in the SAME format
    as results/ground_truth.csv (image,true_label,verdict) so they merge in and
    feed scripts/prepare_finetune_data.py for the next fine-tune / retrain

Progress is saved in the browser (localStorage), so you can close and resume.

Usage:
  python3 scripts/roi_review_page.py \
      --csv results/roi_serials.csv \
      --data "dataset for test/ROI" \
      --out results/roi_review.html
"""
import csv, os, html, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/roi_serials.csv",
                help="extract_serials.py output CSV")
ap.add_argument("--data", default="dataset for test/ROI",
                help="image folder the CSV refers to")
ap.add_argument("--out", default="results/roi_review.html")
args = ap.parse_args()

# image src in the HTML is relative to the HTML file (which lives in results/)
IMG_REL = os.path.relpath(args.data, os.path.dirname(args.out))

rows = list(csv.DictReader(open(args.csv)))

cards = []
for r in rows:
    img = r["image"]
    reading = (r.get("display_text") or "").strip()
    status = r.get("status", "")
    score = float(r["score"]) if r.get("score") else 0.0
    cards.append({
        "img": IMG_REL + "/" + img,
        "path": os.path.abspath(os.path.join(args.data, img)),  # full copyable path
        "name": img,
        "reading": reading,          # what the OCR model produced ("" = blank)
        "status": status,            # accepted / review / blank (from the gate)
        "score": round(score, 3),
        "search": (reading + " " + img + " " + status).lower(),
    })

data_json = json.dumps(cards)

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ROI OCR Verification</title>
<style>
  :root{--bg:#0f1115;--card:#1a1d24;--line:#2a2f3a;--text:#e6e8ec;--muted:#9aa3b2;
        --good:#2ecc71;--bad:#e74c3c;--warn:#e67e22;--accent:#4a9eff;--blank:#7a8494;}
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       background:var(--bg);color:var(--text)}
  header{position:sticky;top:0;z-index:20;background:#12151b;border-bottom:1px solid var(--line);
         padding:14px 20px}
  h1{margin:0 0 8px;font-size:17px}
  /* live monitor bar */
  .monitor{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--muted);align-items:center}
  .monitor b{color:var(--text)}
  .pill{padding:3px 10px;border-radius:20px;border:1px solid var(--line);background:var(--card)}
  .pill.good{color:var(--good)} .pill.bad{color:var(--bad)} .pill.left{color:var(--warn)}
  .bar{height:6px;border-radius:4px;background:#222833;overflow:hidden;flex:1;min-width:160px}
  .bar > i{display:block;height:100%;width:0;background:var(--good);transition:width .2s}
  .controls{margin-top:10px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  input[type=search]{flex:1;min-width:220px;padding:9px 12px;border-radius:8px;
        border:1px solid var(--line);background:var(--card);color:var(--text);font-size:14px}
  .filters button{padding:7px 12px;border-radius:20px;border:1px solid var(--line);
        background:var(--card);color:var(--text);cursor:pointer;font-size:13px}
  .filters button.active{background:#2b3444;border-color:#3d4a60}
  .dl{padding:8px 14px;border-radius:8px;border:1px solid var(--good);background:var(--good);
      color:#052;font-weight:600;cursor:pointer;font-size:13px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;padding:20px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;
        display:flex;flex-direction:column}
  .card.correct{outline:2px solid var(--good)}
  .card.wrong{outline:2px solid var(--bad)}
  .card.novis{outline:2px solid var(--warn)}
  .imgwrap{background:#000;display:flex;align-items:center;justify-content:center;min-height:100px;
           padding:8px;cursor:zoom-in}
  .imgwrap img{max-width:100%;height:auto}
  .reading{padding:10px 12px 4px;display:flex;align-items:baseline;justify-content:space-between;gap:8px}
  .serial{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:21px;font-weight:600;
          letter-spacing:1px;word-break:break-all}
  .serial.empty{color:var(--blank);font-style:italic;font-size:15px}
  .score{flex:none;font-size:11px;color:var(--muted)}
  .fname{padding:0 12px 8px;font-size:11px;color:var(--muted);word-break:break-all}
  .verdict{display:flex;gap:8px;padding:0 12px 8px}
  .vbtn{flex:1;padding:8px;border-radius:8px;border:1px solid var(--line);background:#20293a;
        color:var(--text);cursor:pointer;font-size:14px;font-weight:600}
  .vbtn.ok:hover,.vbtn.ok.on{background:var(--good);color:#052;border-color:var(--good)}
  .vbtn.no:hover,.vbtn.no.on{background:var(--bad);color:#310;border-color:var(--bad)}
  .vbtn.nv:hover,.vbtn.nv.on{background:var(--warn);color:#210;border-color:var(--warn)}
  .fixrow{display:flex;gap:6px;padding:0 12px 10px;align-items:center}
  .fixlbl{font-size:11px;color:var(--muted);white-space:nowrap}
  .fixinp{flex:1;min-width:80px;font-family:ui-monospace,monospace;font-size:15px;padding:7px 9px;
          border-radius:7px;border:1px solid var(--warn);background:#0f1218;color:var(--text);
          text-transform:uppercase}
  .unread{padding:7px 9px;border-radius:7px;border:1px solid var(--line);background:#20293a;
          color:var(--muted);cursor:pointer;font-size:12px;white-space:nowrap}
  .cardbtns{padding:0 12px 12px}
  .copybtn{font-size:11px;padding:5px 10px;border-radius:6px;border:1px solid var(--line);
           background:#20293a;color:var(--text);cursor:pointer}
  .copybtn.done{background:var(--good);color:#052;border-color:var(--good)}
  .empty-msg{padding:40px;text-align:center;color:var(--muted)}
  /* zoom lightbox */
  .lightbox{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.92);display:none;
            flex-direction:column;align-items:center;justify-content:center;gap:14px}
  .lightbox.open{display:flex}
  .lightbox img{max-width:94vw;max-height:80vh;transform:scale(var(--z,1));transform-origin:center;
                transition:transform .1s}
  .lb-bar{display:flex;gap:10px;align-items:center;color:var(--muted);font-size:13px;flex-wrap:wrap;
          justify-content:center;font-family:ui-monospace,monospace;padding:0 16px}
  .lb-bar button{padding:7px 13px;border-radius:8px;border:1px solid #3d4a60;background:#1a1d24;
                 color:var(--text);cursor:pointer}
  .lb-path{max-width:70vw;word-break:break-all;color:#cbd3e1}
</style>
</head>
<body>
<header>
  <h1>ROI OCR Verification</h1>
  <div class="monitor">
    <span>Checked <b id="m-done">0</b> / <b id="m-total">0</b></span>
    <span class="pill left">Not checked <b id="m-left">0</b></span>
    <span class="pill good">✓ Correct <b id="m-ok">0</b></span>
    <span class="pill bad">✗ Wrong <b id="m-no">0</b></span>
    <span class="pill left">🚫 Not visible <b id="m-nv">0</b></span>
    <div class="bar"><i id="m-fill"></i></div>
  </div>
  <div class="controls">
    <input type="search" id="q" placeholder="Search serial or filename…">
    <div class="filters">
      <button data-f="all" class="active">All</button>
      <button data-f="unchecked">◻ Not checked</button>
      <button data-f="correct">✓ Correct</button>
      <button data-f="wrong">✗ Wrong</button>
      <button data-f="not_visible">🚫 Not visible</button>
      <button data-f="accepted">Model: accepted</button>
      <button data-f="review">Model: review</button>
      <button data-f="blank">Model: blank</button>
    </div>
    <button class="dl" id="dl">⬇ Download results.csv</button>
    <button class="dl" id="dl-train" title="corrections in ground_truth.csv format, for retraining"
            style="background:var(--accent);border-color:var(--accent);color:#04121f">⬇ Download training-labels.csv</button>
  </div>
</header>

<div class="grid" id="grid"></div>
<div class="empty-msg" id="empty" style="display:none">No matches.</div>

<div class="lightbox" id="lb">
  <img id="lbimg" alt="">
  <div class="lb-bar">
    <button id="zout">– zoom</button><button id="zin">+ zoom</button><button id="zreset">reset</button>
    <span class="lb-path" id="lbpath"></span>
    <button id="lbcopy">Copy path</button><button id="lbclose">✕ close</button>
  </div>
</div>

<script>
const DATA = __DATA__;
const KEY = "roi_review_v1";                 // browser store: name -> {v, s}
const FB = JSON.parse(localStorage.getItem(KEY) || "{}");
const grid = document.getElementById('grid'), q = document.getElementById('q');
let filter = 'all';

function save(){ localStorage.setItem(KEY, JSON.stringify(FB)); }
function setVerdict(c, verdict, value){
  FB[c.name] = {v:verdict, s:(value||"")};
  save(); updateMonitor();
}
function clearVerdict(c){ delete FB[c.name]; save(); updateMonitor(); }

// ---- live monitor ----
function updateMonitor(){
  const total = DATA.length;
  let ok=0,no=0,nv=0;
  for(const c of DATA){ const f=FB[c.name]; if(!f)continue;
    if(f.v==='correct')ok++; else if(f.v==='not_visible')nv++; else no++; }
  const done = ok+no+nv, left = total-done;
  document.getElementById('m-total').textContent = total;
  document.getElementById('m-done').textContent  = done;
  document.getElementById('m-left').textContent  = left;
  document.getElementById('m-ok').textContent    = ok;
  document.getElementById('m-no').textContent    = no;
  document.getElementById('m-nv').textContent    = nv;
  document.getElementById('m-fill').style.width  = total? (done/total*100)+'%' : '0';
}

// ---- copy helper ----
function copyText(txt, btn, label){
  const done=()=>{const t=btn.textContent;btn.textContent=label;btn.classList.add('done');
    setTimeout(()=>{btn.textContent=t;btn.classList.remove('done');},1200);};
  if(navigator.clipboard) navigator.clipboard.writeText(txt).then(done).catch(fb); else fb();
  function fb(){const t=document.createElement('textarea');t.value=txt;document.body.appendChild(t);
    t.select();try{document.execCommand('copy');done();}catch(e){}document.body.removeChild(t);}
}

// ---- zoom lightbox ----
const lb=document.getElementById('lb'),lbimg=document.getElementById('lbimg'),
      lbpath=document.getElementById('lbpath'); let zoom=3,curPath='';
function setZoom(z){zoom=Math.min(12,Math.max(1,z));lbimg.style.setProperty('--z',zoom);}
function openLB(c){curPath=c.path;lbimg.src=c.img;lbpath.textContent=c.path;setZoom(3);lb.classList.add('open');}
function closeLB(){lb.classList.remove('open');}
document.getElementById('zin').onclick=()=>setZoom(zoom+1);
document.getElementById('zout').onclick=()=>setZoom(zoom-1);
document.getElementById('zreset').onclick=()=>setZoom(3);
document.getElementById('lbclose').onclick=closeLB;
document.getElementById('lbcopy').onclick=e=>copyText(curPath,e.target,'Copied ✓');
lb.addEventListener('click',e=>{if(e.target===lb)closeLB();});
lb.addEventListener('wheel',e=>{e.preventDefault();setZoom(zoom+(e.deltaY<0?1:-1));},{passive:false});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeLB();});

// ---- render ----
function passesFilter(c){
  const f = FB[c.name];
  if(filter==='all') return true;
  if(filter==='unchecked')   return !f;
  if(filter==='correct')     return f && f.v==='correct';
  if(filter==='not_visible') return f && f.v==='not_visible';
  if(filter==='wrong')       return f && f.v!=='correct' && f.v!=='not_visible';
  return c.status===filter;              // accepted / review / blank
}
function render(){
  const term=q.value.trim().toLowerCase();
  grid.innerHTML=''; let shown=0;
  for(const c of DATA){
    if(!passesFilter(c)) continue;
    if(term && !c.search.includes(term)) continue;
    shown++;
    const f=FB[c.name];
    const clsOf = v => v==='correct'?'card correct':(v==='not_visible'?'card novis':'card wrong');
    const el=document.createElement('div');
    el.className = f ? clsOf(f.v) : 'card';
    const readingHtml = c.reading
      ? '<span class="serial">'+c.reading+'</span>'
      : '<span class="serial empty">(blank / no read)</span>';
    const isWrong = f && f.v!=='correct' && f.v!=='not_visible';   // wrong or unreadable
    // the correction box is ALWAYS visible, prefilled with the model reading so
    // you can just fix the wrong characters in place (TWPD1234 -> TPDL1234)
    const startVal = (f && (f.v==='wrong'||f.v==='fixed')) ? f.s
                   : ((f && (f.v==='unreadable'||f.v==='not_visible')) ? '' : c.reading);
    el.innerHTML =
      '<div class="imgwrap" title="click to zoom"><img loading="lazy" src="'+c.img+'"></div>'+
      '<div class="reading">'+readingHtml+'<span class="score">'+
         (c.reading?('conf '+c.score+' · '+c.status):c.status)+'</span></div>'+
      '<div class="fname">'+c.name+'</div>'+
      '<div class="fixrow">'+
        '<span class="fixlbl">Correct&nbsp;value:</span>'+
        '<input class="fixinp" placeholder="type the correct serial" value="'+
           startVal.replace(/"/g,'&quot;')+'">'+
      '</div>'+
      '<div class="verdict">'+
        '<button class="vbtn ok'+(f&&f.v==='correct'?' on':'')+'">✓ Correct</button>'+
        '<button class="vbtn no'+(isWrong?' on':'')+'">✗ Wrong</button>'+
        '<button class="vbtn nv'+(f&&f.v==='not_visible'?' on':'')+'">🚫 Not visible</button>'+
        '<button class="unread">Unreadable</button>'+
      '</div>'+
      '<div class="cardbtns"><button class="copybtn">Copy path</button></div>';

    const okBtn=el.querySelector('.ok'), noBtn=el.querySelector('.no'), nvBtn=el.querySelector('.nv'),
          fixInp=el.querySelector('.fixinp');
    const setBtns=(on)=>{ okBtn.classList.toggle('on',on==='ok'); noBtn.classList.toggle('on',on==='no');
                          nvBtn.classList.toggle('on',on==='nv'); };

    el.querySelector('.imgwrap').onclick=()=>openLB(c);
    el.querySelector('.copybtn').onclick=ev=>copyText(c.path,ev.target,'Copied ✓');

    okBtn.onclick=()=>{ setVerdict(c,'correct',c.reading);
      fixInp.value=c.reading; el.className='card correct'; setBtns('ok'); };
    // ✗ Wrong: record the current box value as the correction and put the cursor
    // in the box so you can edit the wrong characters straight away
    noBtn.onclick=()=>{ setVerdict(c,'wrong', fixInp.value.trim().toUpperCase()||c.reading);
      el.className='card wrong'; setBtns('no'); fixInp.focus(); fixInp.select(); };
    nvBtn.onclick=()=>{ setVerdict(c,'not_visible','');
      el.className='card novis'; setBtns('nv'); };
    // typing your correction directly also marks the card wrong (with your value)
    const commitFix=()=>{ setVerdict(c,'wrong', fixInp.value.trim().toUpperCase());
      el.className='card wrong'; setBtns('no'); };
    fixInp.oninput=commitFix;
    fixInp.onkeydown=e=>{ if(e.key==='Enter'){commitFix(); fixInp.blur();} };
    el.querySelector('.unread').onclick=()=>{ setVerdict(c,'unreadable','');
      fixInp.value=''; el.className='card wrong'; setBtns('no'); };

    grid.appendChild(el);
  }
  document.getElementById('empty').style.display = shown? 'none':'block';
}
q.addEventListener('input',render);
document.querySelectorAll('.filters button').forEach(b=>{
  b.onclick=()=>{document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); filter=b.dataset.f; render();};
});

// ---- export ----
document.getElementById('dl').onclick=()=>{
  const esc=s=>'"'+(s||"").replace(/"/g,'""')+'"';
  let csv="image,image_path,model_reading,score,model_status,verdict,correct_serial\n";
  for(const c of DATA){
    const f=FB[c.name];
    const verdict = f ? f.v : "unchecked";
    // correct_serial = model reading if confirmed correct; the typed fix if wrong; blank if unreadable/unchecked
    let truth="";
    if(f){ if(f.v==='correct') truth=c.reading; else if(f.v==='unreadable') truth=""; else truth=f.s; }
    csv += [esc(c.name),esc(c.path),esc(c.reading),c.score,esc(c.status),
            esc(verdict),esc(truth)].join(",")+"\n";
  }
  const b=new Blob([csv],{type:"text/csv"});const a=document.createElement('a');
  a.href=URL.createObjectURL(b); a.download="roi_results.csv"; a.click();
};

// ---- export: retraining labels (same schema as results/ground_truth.csv) ----
// Only reviewed images become labels:
//   correct  -> true_label = the model reading   (verdict "correct")
//   wrong    -> true_label = your typed serial   (verdict "fixed")
//   unreadable / not_visible -> blank label      (kept, but excluded from training
//                                                 by prepare_finetune_data.py)
//   unchecked -> skipped entirely
document.getElementById('dl-train').onclick=()=>{
  const esc=s=>'"'+(s||"").replace(/"/g,'""')+'"';
  let csv="image,true_label,verdict\n", n=0, corr=0;
  for(const c of DATA){
    const f=FB[c.name];
    if(!f) continue;                                  // skip unchecked
    let label="", verdict=f.v;
    if(f.v==='correct'){ label=c.reading; }
    else if(f.v==='not_visible'||f.v==='unreadable'){ label=""; }
    else { label=(f.s||"").toUpperCase(); verdict="fixed"; corr++; }   // wrong -> your correction
    csv += [esc(c.name), esc(label), esc(verdict)].join(",")+"\n";
    n++;
  }
  if(!n){ alert("No reviewed images yet — mark some Correct/Wrong first."); return; }
  const b=new Blob([csv],{type:"text/csv"});const a=document.createElement('a');
  a.href=URL.createObjectURL(b); a.download="roi_training_labels.csv"; a.click();
  console.log(`training-labels: ${n} labeled, ${corr} human corrections`);
};

updateMonitor(); render();
</script>
</body>
</html>
"""

out = HTML.replace("__DATA__", data_json)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as fh:
    fh.write(out)
print(f"Wrote {args.out}  ({len(cards)} images)")
