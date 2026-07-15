#!/usr/bin/env python3
"""Generate results/review.html: each meter crop next to its extracted serial.

Usage: python3 scripts/build_review_page.py [--csv results/serials.csv]
                                            [--data data/serialno_dev]
"""
import csv, os, html, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/serials.csv")
ap.add_argument("--data", default="data/serialno_dev", help="image folder the CSV refers to")
ap.add_argument("--out", default="results/review.html")
args = ap.parse_args()

# image src is relative to the HTML file (which lives in results/)
IMG_REL = os.path.relpath(args.data, os.path.dirname(args.out))

rows = list(csv.DictReader(open(args.csv)))

cards = []
found = 0
for r in rows:
    img = r["image"]
    serial = (r.get("final_serial") or r.get("display_text") or r.get("predicted_serial") or "").strip()
    score = float(r["score"]) if r.get("score") else 0.0
    tier = r.get("tier", "")
    status = r.get("status", "")
    alt = (r.get("trocr") or "").strip()    # TrOCR second opinion (for 'check' tier)
    if serial:
        found += 1
    # colour by trust tier (final.csv) > gate status (test_serials) > raw score
    if tier:
        cls = {"accept": "good", "check": "check", "review": "low", "blank": "blank"}.get(tier, "low")
        badge = {"accept": f"accept {score:.2f}", "check": "check · models differ",
                 "review": f"review {score:.2f}", "blank": "blank"}[tier]
    elif status == "accepted":
        cls, badge = "good", f"accepted {score:.2f}"
    elif status == "review":
        cls, badge = "low", f"review {score:.2f}"
    elif status == "blank":
        cls, badge = "blank", "blank"
    elif not serial:
        cls, badge = "blank", "no read"
    elif score >= 0.90:
        cls, badge = "good", f"{score:.2f}"
    else:
        cls, badge = "low", f"{score:.2f}"
    cards.append({
        "img": IMG_REL + "/" + img,
        "path": os.path.abspath(os.path.join(args.data, img)),   # full path to copy
        "serial": serial or "—",
        "alt": alt if (cls == "check" and alt and alt.upper() != serial.upper()) else "",
        "badge": badge,
        "cls": cls,
        "search": (serial + " " + alt + " " + img).lower(),
    })

total = len(rows)
data_json = json.dumps(cards)

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meter Serial Number Review</title>
<style>
  :root {
    --bg:#0f1115; --card:#1a1d24; --line:#2a2f3a; --text:#e6e8ec; --muted:#9aa3b2;
    --good:#2ecc71; --mid:#f1c40f; --low:#e67e22; --blank:#e74c3c;
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--text); }
  header { position:sticky; top:0; z-index:10; background:#12151b; border-bottom:1px solid var(--line);
           padding:16px 20px; }
  h1 { margin:0 0 4px; font-size:18px; }
  .stats { color:var(--muted); font-size:13px; }
  .controls { margin-top:12px; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  input[type=search] { flex:1; min-width:220px; padding:9px 12px; border-radius:8px;
        border:1px solid var(--line); background:var(--card); color:var(--text); font-size:14px; }
  .filters button { padding:7px 12px; border-radius:20px; border:1px solid var(--line);
        background:var(--card); color:var(--text); cursor:pointer; font-size:13px; }
  .filters button.active { background:#2b3444; border-color:#3d4a60; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
          gap:14px; padding:20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden;
          display:flex; flex-direction:column; }
  .imgwrap { background:#000; display:flex; align-items:center; justify-content:center;
             min-height:90px; padding:8px; cursor:zoom-in; }
  .imgwrap img { max-width:100%; height:auto; image-rendering:auto; }
  /* zoom lightbox */
  .lightbox { position:fixed; inset:0; z-index:100; background:rgba(0,0,0,.9);
              display:none; flex-direction:column; align-items:center; justify-content:center; gap:14px; }
  .lightbox.open { display:flex; }
  .lightbox img { max-width:94vw; max-height:80vh; image-rendering:auto;
                  transition:transform .1s; transform:scale(var(--z,1)); transform-origin:center; }
  .lb-bar { display:flex; gap:10px; align-items:center; color:var(--muted); font-size:13px; flex-wrap:wrap;
            justify-content:center; font-family:ui-monospace,monospace; padding:0 16px; }
  .lb-bar button { padding:7px 13px; border-radius:8px; border:1px solid #3d4a60;
                   background:#1a1d24; color:var(--text); cursor:pointer; font-size:14px; }
  .lb-bar button:hover { background:#2b3444; }
  .lb-path { max-width:70vw; word-break:break-all; color:#cbd3e1; }
  .meta { padding:10px 12px; display:flex; align-items:center; justify-content:space-between; gap:8px; }
  .serial { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:20px; font-weight:600;
            letter-spacing:1px; word-break:break-all; }
  .badge { flex:none; font-size:12px; padding:3px 9px; border-radius:20px; font-weight:600; }
  .good .badge { background:rgba(46,204,113,.15); color:var(--good); }
  .mid  .badge { background:rgba(241,196,15,.15); color:var(--mid); }
  .low  .badge { background:rgba(230,126,34,.15); color:var(--low); }
  .check .badge { background:rgba(241,196,15,.18); color:var(--mid); }
  .check { outline:1px solid rgba(241,196,15,.35); }
  .alt { padding:0 12px 8px; font-size:12px; color:var(--mid);
         font-family:ui-monospace,Menlo,monospace; }
  .alt b { color:var(--muted); font-family:system-ui,sans-serif; font-weight:500; }
  .blank .serial { color:var(--blank); }
  .blank .badge { background:rgba(231,76,60,.15); color:var(--blank); }
  .fname { padding:0 12px 6px; font-size:11px; color:var(--muted); word-break:break-all; }
  .cardbtns { padding:0 12px 10px; }
  .copybtn { font-size:11px; padding:4px 10px; border-radius:6px; border:1px solid var(--line);
             background:#20293a; color:var(--text); cursor:pointer; }
  .copybtn:hover { background:#2b3444; }
  .copybtn.done { background:var(--good); color:#062; border-color:var(--good); }
  /* live feedback controls */
  .fb { display:flex; gap:6px; padding:2px 12px 10px; flex-wrap:wrap; align-items:center; }
  .fbinp { flex:1; min-width:110px; font-family:ui-monospace,monospace; font-size:14px;
           padding:6px 8px; border-radius:6px; border:1px solid var(--line);
           background:#0f1218; color:var(--text); text-transform:uppercase; }
  .fbbtn { padding:6px 9px; border-radius:6px; border:1px solid var(--line);
           background:#20293a; color:var(--text); cursor:pointer; font-size:12px; }
  .fbbtn:hover { background:#2b3444; }
  .fbbtn.ok:hover { background:var(--good); color:#062; }
  .fbbtn.bad:hover { background:var(--blank); color:#310; }
  .usealt { cursor:pointer; color:var(--mid); text-decoration:underline; }
  .card.reviewed { outline:2px solid var(--good); }
  .card.reviewed .badge::after { content:" · reviewed"; color:var(--good); }
  .empty { padding:40px; text-align:center; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Meter Serial Number Review</h1>
  <div class="stats" id="stats"></div>
  <div class="controls">
    <input type="search" id="q" placeholder="Search serial or filename…">
    <div class="filters">
      <button data-f="all" class="active">All</button>
      <button data-f="good">✓ Accept</button>
      <button data-f="low">Review</button>
      <button data-f="blank">Blank</button>
      <button data-f="unreviewed">◻ Unreviewed</button>
    </div>
    <button class="copybtn" id="dlfb" style="padding:8px 12px;font-size:13px">⬇ Download feedback.csv</button>
  </div>
</header>
<div class="grid" id="grid"></div>
<div class="empty" id="empty" style="display:none">No matches.</div>

<!-- zoom lightbox -->
<div class="lightbox" id="lb">
  <img id="lbimg" alt="">
  <div class="lb-bar">
    <button id="zout">– zoom</button><button id="zin">+ zoom</button>
    <button id="zreset">reset</button>
    <span class="lb-path" id="lbpath"></span>
    <button id="lbcopy">Copy path</button>
    <button id="lbclose">✕ close</button>
  </div>
</div>

<script>
const DATA = __DATA__;
const grid = document.getElementById('grid');
const q = document.getElementById('q');
const stats = document.getElementById('stats');
let filter = 'all';

// --- live feedback store (persists in the browser) ---
const FBKEY = "serial_feedback_v1__DSKEY__";   // per-dataset: keeps each dataset's feedback separate
const FB = JSON.parse(localStorage.getItem(FBKEY) || "{}");   // name -> {v:verdict, s:value}
function fbName(c){ return c.path.split('/').pop(); }
function saveFB(c, verdict, value){
  FB[fbName(c)] = {v:verdict, s:value};
  localStorage.setItem(FBKEY, JSON.stringify(FB));
}
function exportFB(){
  let csv = "image,tier,model_read,trocr,verdict,true_label\\n";
  const esc = s => '"' + (s||"").replace(/"/g,'""') + '"';
  for (const c of DATA){
    const f = FB[fbName(c)];
    if (!f) continue;                       // only export images you reviewed
    const truth = f.v==='unreadable' ? "" : (f.v==='fixed' ? f.s : (c.serial==='—'?'':c.serial));
    csv += [esc(fbName(c)), esc(c.cls), esc(c.serial==='—'?'':c.serial), esc(c.alt),
            esc(f.v), esc(truth)].join(",") + "\\n";
  }
  const b = new Blob([csv], {type:"text/csv"});
  const a = document.createElement('a'); a.href = URL.createObjectURL(b);
  a.download = "feedback.csv"; a.click();
}
document.getElementById('dlfb').addEventListener('click', exportFB);

function copyText(txt, btn, label) {
  const done = () => { btn.textContent = label; btn.classList.add('done');
                       setTimeout(()=>{btn.textContent='Copy path'; btn.classList.remove('done');}, 1200); };
  if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done).catch(fb); else fb();
  function fb(){ const t=document.createElement('textarea'); t.value=txt; document.body.appendChild(t);
                 t.select(); try{document.execCommand('copy'); done();}catch(e){} document.body.removeChild(t); }
}

// --- zoom lightbox ---
const lb=document.getElementById('lb'), lbimg=document.getElementById('lbimg'),
      lbpath=document.getElementById('lbpath'); let zoom=3, curPath='';
function setZoom(z){ zoom=Math.min(12, Math.max(1, z)); lbimg.style.setProperty('--z', zoom); }
function openLB(c){ curPath=c.path; lbimg.src=c.img; lbpath.textContent=c.path; setZoom(3); lb.classList.add('open'); }
function closeLB(){ lb.classList.remove('open'); }
document.getElementById('zin').onclick=()=>setZoom(zoom+1);
document.getElementById('zout').onclick=()=>setZoom(zoom-1);
document.getElementById('zreset').onclick=()=>setZoom(3);
document.getElementById('lbclose').onclick=closeLB;
document.getElementById('lbcopy').onclick=(e)=>copyText(curPath, e.target, 'Copied ✓');
lb.addEventListener('click', e=>{ if(e.target===lb) closeLB(); });
lb.addEventListener('wheel', e=>{ e.preventDefault(); setZoom(zoom + (e.deltaY<0?1:-1)); }, {passive:false});
document.addEventListener('keydown', e=>{ if(e.key==='Escape') closeLB();
  if(lb.classList.contains('open')){ if(e.key==='+'||e.key==='=')setZoom(zoom+1); if(e.key==='-')setZoom(zoom-1);} });

function render() {
  const term = q.value.trim().toLowerCase();
  let shown = 0;
  grid.innerHTML = '';
  for (const c of DATA) {
    const reviewed = !!FB[fbName(c)];
    if (filter === 'unreviewed') { if (reviewed) continue; }
    else if (filter !== 'all' && c.cls !== filter) continue;
    if (term && !c.search.includes(term)) continue;
    shown++;
    const el = document.createElement('div');
    el.className = 'card ' + c.cls + (reviewed ? ' reviewed' : '');
    const fname = c.path.split('/').pop();
    const fb = FB[fbName(c)];
    const startVal = fb ? (fb.v==='unreadable' ? '' : fb.s) : (c.serial==='—'?'':c.serial);
    const altHtml = c.alt
      ? '<div class="alt"><b>TrOCR:</b> <span class="usealt">' + c.alt + '</span></div>' : '';
    el.innerHTML =
      '<div class="imgwrap" title="click to zoom"><img loading="lazy" src="' + c.img + '"></div>' +
      '<div class="meta"><span class="serial">' + c.serial + '</span>' +
      '<span class="badge">' + c.badge + '</span></div>' +
      '<div class="fname">' + fname + '</div>' + altHtml +
      '<div class="fb">' +
        '<input class="fbinp" value="' + startVal.replace(/"/g,'&quot;') + '">' +
        '<button class="fbbtn ok" title="the reading is correct">✓</button>' +
        '<button class="fbbtn" title="save the typed correction">fix</button>' +
        '<button class="fbbtn bad" title="unreadable / blank">✗</button>' +
      '</div>' +
      '<div class="cardbtns"><button class="copybtn">Copy path</button></div>';
    const inp = el.querySelector('.fbinp');
    const mark = () => { el.classList.add('reviewed'); updateCount(); };
    el.querySelector('.imgwrap').addEventListener('click', ()=>openLB(c));
    el.querySelector('.copybtn').addEventListener('click', ev=>copyText(c.path, ev.target, 'Copied ✓'));
    el.querySelector('.ok').addEventListener('click', ()=>{ saveFB(c,'correct',inp.value.trim().toUpperCase()); mark(); });
    el.querySelector('.fbbtn:not(.ok):not(.bad)').addEventListener('click', ()=>{ saveFB(c,'fixed',inp.value.trim().toUpperCase()); mark(); });
    el.querySelector('.bad').addEventListener('click', ()=>{ saveFB(c,'unreadable',''); inp.value=''; mark(); });
    inp.addEventListener('keydown', e=>{ if(e.key==='Enter'){ saveFB(c,'fixed',inp.value.trim().toUpperCase()); mark(); }});
    const ua = el.querySelector('.usealt');
    if (ua) ua.addEventListener('click', ()=>{ inp.value = c.alt.toUpperCase(); inp.focus(); });
    grid.appendChild(el);
  }
  document.getElementById('empty').style.display = shown ? 'none' : 'block';
  updateCount(shown);
}
function updateCount(shown) {
  const g = DATA.filter(c=>c.cls==='good').length;
  const done = Object.keys(FB).length;
  const s = (shown===undefined) ? '' : ` · showing ${shown}`;
  stats.textContent = `${DATA.length} images · ${g} accepted · reviewed ${done}${s}`;
}
q.addEventListener('input', render);
document.querySelectorAll('.filters button').forEach(b => {
  b.addEventListener('click', () => {
    document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); filter = b.dataset.f; render();
  });
});
render();
</script>
</body>
</html>
"""

# per-dataset feedback key, so each dataset's review progress/export is isolated
dskey = os.path.basename(args.data.rstrip("/")) or "default"
out = HTML.replace("__DATA__", data_json).replace("__DSKEY__", dskey)
with open(args.out, "w") as fh:
    fh.write(out)
print(f"Wrote {args.out}  ({total} images, {found} with a serial)")
