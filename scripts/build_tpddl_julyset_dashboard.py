#!/usr/bin/env python3
"""
Build a local HTML dashboard for the TPDDL 1st-4th July set 3-model-vs-bot merge.

Input:  Some_analysis_part/merged_3models_vs_bot_tpddl.csv
        (image_name, fp32, ptq_v3, ptq_v2, all_agree, bot_serial, bigImg_url,
         fp32_eq_bot, ptq_v3_eq_bot, ptq_v2_eq_bot)
Images: Some_analysis_part/1stjuly_4thjuly_2026_tpddl/  (relative to the html) —
        this is the exact crop that was fed into the model for extraction, shown
        unmodified (no thumbnail re-crop). NOT the full uncropped meter photo.

Output: Some_analysis_part/tpddl_julyset_dashboard.html   (open in a browser)

Any image already present in Some_analysis_part/all3_agree_eq_bot_tpddl/ground_truth.csv
is EXCLUDED from this dashboard (that folder is the running "taken out / resolved"
set — built by export_all3_agree_eq_bot_tpddl.py and grown over time by merge
scripts like merge_fp32ptqv3_reviewed_into_all3.py). Only the remaining,
still-unresolved rows are shown/filtered here.

Filters (with counts):
    all                    every remaining row
    all3_agree             fp32 == ptq_v3 == ptq_v2 (but != bot, since agree+=bot is excluded)
    all3_neq_bot           none of the 3 models match bot_serial
    fp32_eq_bot / _neq_bot fp32 alone == / != bot_serial
    ptq_v3_eq_bot / _neq_bot
    ptq_v2_eq_bot / _neq_bot

Usage:
    python3 scripts/build_tpddl_julyset_dashboard.py
"""
import os
import csv
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "Some_analysis_part")
SRC = os.path.join(BASE, "merged_3models_vs_bot_tpddl.csv")
IMGDIR = "1stjuly_4thjuly_2026_tpddl"  # relative to BASE — the crop actually fed to the model
OUT = os.path.join(BASE, "tpddl_julyset_dashboard.html")

with open(SRC, newline="", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))

# Any image already taken out into all3_agree_eq_bot_tpddl/ground_truth.csv, OR
# explicitly discarded (reviewed and dropped, no label) via discarded_tpddl.csv,
# is excluded here so the dashboard only shows the remaining, still-unresolved images.
TAKEN_OUT_CSV = os.path.join(BASE, "all3_agree_eq_bot_tpddl", "ground_truth.csv")
with open(TAKEN_OUT_CSV, newline="", encoding="utf-8") as f:
    taken_out = {r["filename"] for r in csv.DictReader(f)}

DISCARDED_CSV = os.path.join(BASE, "discarded_tpddl.csv")
discarded = set()
if os.path.exists(DISCARDED_CSV):
    with open(DISCARDED_CSV, newline="", encoding="utf-8") as f:
        discarded = {r["filename"] for r in csv.DictReader(f)}

EXCLUDED_N = 0
rows = []
for r in all_rows:
    if r["image_name"] in taken_out or r["image_name"] in discarded:
        EXCLUDED_N += 1
        continue
    rows.append(r)

data = []
counts = {
    "all": 0,
    "all3_agree": 0,
    "fp32_eq_bot": 0,
    "ptq_v3_eq_bot": 0,
    "ptq_v2_eq_bot": 0,
    "all3_neq_bot": 0,
    "fp32_neq_bot": 0,
    "ptq_v3_neq_bot": 0,
    "ptq_v2_neq_bot": 0,
    "fp32_and_ptqv3_eq_bot": 0,
}
for r in rows:
    all3 = r["all_agree"] == "yes"
    f_ok = r["fp32_eq_bot"] == "1"
    p3_ok = r["ptq_v3_eq_bot"] == "1"
    p2_ok = r["ptq_v2_eq_bot"] == "1"
    all3_neq_bot = (not f_ok) and (not p3_ok) and (not p2_ok)  # none of the 3 match bot
    fp32_and_ptqv3_ok = f_ok and p3_ok  # both fp32 and ptq_v3 match bot (ptq_v2 unconstrained)

    counts["all"] += 1
    if all3:
        counts["all3_agree"] += 1
    if fp32_and_ptqv3_ok:
        counts["fp32_and_ptqv3_eq_bot"] += 1
    if f_ok:
        counts["fp32_eq_bot"] += 1
    if p3_ok:
        counts["ptq_v3_eq_bot"] += 1
    if p2_ok:
        counts["ptq_v2_eq_bot"] += 1
    if all3_neq_bot:
        counts["all3_neq_bot"] += 1
    if not f_ok:
        counts["fp32_neq_bot"] += 1
    if not p3_ok:
        counts["ptq_v3_neq_bot"] += 1
    if not p2_ok:
        counts["ptq_v2_neq_bot"] += 1

    data.append({
        "img": r["image_name"],
        "url": r.get("bigImg_url", ""),
        "fp32": r["fp32"], "ptq_v3": r["ptq_v3"], "ptq_v2": r["ptq_v2"],
        "bot": r["bot_serial"],
        "all3": all3, "all3_neq_bot": all3_neq_bot, "fp32_and_ptqv3_ok": fp32_and_ptqv3_ok,
        "f_ok": f_ok, "p3_ok": p3_ok, "p2_ok": p2_ok,
    })

total = counts["all"]
pct = lambda n: round(100.0 * n / total, 2) if total else 0
DATA = json.dumps(data)

FILTERS = [
    ("all", "All"),
    ("all3_agree", "All 3 agree"),
    ("all3_neq_bot", "All 3 ≠ bot (none match)"),
    ("fp32_and_ptqv3_eq_bot", "FP32 & PTQ v3 both = bot"),
    ("fp32_eq_bot", "FP32 = bot"),
    ("fp32_neq_bot", "FP32 ≠ bot"),
    ("ptq_v3_eq_bot", "PTQ v3 = bot"),
    ("ptq_v3_neq_bot", "PTQ v3 ≠ bot"),
    ("ptq_v2_eq_bot", "PTQ v2 = bot"),
    ("ptq_v2_neq_bot", "PTQ v2 ≠ bot"),
]

kpi_html = "".join(
    f'<div class="kpi"><div class=n>{counts[k]:,}</div>'
    f'<div class=l>{label}<br>{pct(counts[k])}%</div></div>'
    for k, label in FILTERS
)
btn_html = "".join(
    f'<span class=fgroup>'
    f'<button data-f="{k}" class="{"on" if k == "all" else ""}">{label} ({counts[k]:,})</button>'
    f'<a class=review href="tpddl_review_{k}.html" target=_blank title="Keep/Discard/Write review for this filter">review &rarr;</a>'
    f'</span>'
    for k, label in FILTERS
)

tpl = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>TPDDL July set — 3-model vs bot</title>
<style>
:root{color-scheme:light dark;
 --bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--mut:#52514e;--line:#e6e5e2;
 --good:#0ca30c;--goodink:#006300;--bad:#d03b3b;--warn:#fab219;--warnink:#7a5200;--accent:#2f6fed;}
@media(prefers-color-scheme:dark){:root{
 --bg:#141413;--card:#1f1f1e;--ink:#f5f5f2;--mut:#b4b3ab;--line:#33322f;
 --good:#3ec13e;--goodink:#8ff08f;--bad:#ef6a6a;--warn:#fab219;--warnink:#ffd98a;--accent:#6c9bff;}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
 background:var(--bg);color:var(--ink);padding:28px clamp(14px,4vw,48px)}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 22px;font-size:13.5px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.kpi .n{font-size:26px;font-weight:650;letter-spacing:-.5px}
.kpi .l{color:var(--mut);font-size:12px;margin-top:3px}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);
 border-radius:10px;padding:12px 16px;margin-bottom:24px;font-size:13.5px;color:var(--mut)}
.note b{color:var(--ink)}
.gbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
.fgroup{display:inline-flex;align-items:center;gap:6px}
.fgroup a.review{font-size:12px;color:var(--accent);text-decoration:none;white-space:nowrap}
.fgroup a.review:hover{text-decoration:underline}
button{font:inherit;background:var(--card);color:var(--ink);
 border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
button.on{background:var(--ink);color:var(--bg)}
.reviewbar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;background:var(--card);
 border:1px solid var(--line);border-radius:14px;padding:14px 18px;margin-bottom:18px}
.reviewbar .stats{display:flex;gap:16px;font-size:13px;color:var(--mut);flex-wrap:wrap}
.reviewbar .stats b{color:var(--ink)}
.reviewbar label{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--mut);cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px}
.card{background:var(--card);border:2px solid var(--line);border-radius:12px;overflow:hidden;transition:border-color .15s}
.card.st-keep{border-color:var(--good)}
.card.st-discard{border-color:var(--bad);opacity:.55}
.card.st-written{border-color:var(--accent)}
.card img{width:100%;height:160px;object-fit:contain;background:#000;display:block;cursor:zoom-in}
.card .b{padding:10px 12px;font-size:12.5px}
.namerow{display:flex;align-items:center;gap:6px;margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.namerow code{flex:1;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px;color:var(--mut);
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cp{flex:none;font-size:11px;padding:3px 9px;border-radius:6px;border:1px solid var(--line);
 background:var(--bg);color:var(--ink);cursor:pointer}
.cp.done{background:var(--good);color:#fff;border-color:var(--good)}
.row{display:flex;justify-content:space-between;gap:8px;margin:2px 0}
.row .k{color:var(--mut)}.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.ok{color:var(--goodink)}.no{color:var(--bad)}
.more{display:block;margin:22px auto;padding:9px 22px}
.revrow{display:flex;gap:6px;margin-top:9px;padding-top:9px;border-top:1px solid var(--line)}
.revrow button{flex:1;font-size:11px;padding:5px 6px}
.revrow button.keep{border-color:var(--good);color:var(--goodink)}
.revrow button.keep.active{background:var(--good);color:#fff}
.revrow button.discard{border-color:var(--bad);color:var(--bad)}
.revrow button.discard.active{background:var(--bad);color:#fff}
.writerow{display:flex;gap:6px;margin-top:6px}
.writerow input{flex:1;font:inherit;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;
 padding:5px 8px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.writerow button{font-size:11px;padding:5px 9px}
.statustag{font-size:10.5px;margin-top:6px;color:var(--mut)}
.statustag.keep{color:var(--goodink)}.statustag.discard{color:var(--bad)}.statustag.written{color:var(--accent)}
.lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:99;
 align-items:center;justify-content:center;cursor:zoom-out;padding:30px}
.lightbox.on{display:flex}
.lightbox img{max-width:100%;max-height:100%;border-radius:8px;box-shadow:0 10px 40px rgba(0,0,0,.5)}
</style></head><body>
<div class=lightbox id=lightbox><img id=lightboximg src="" alt=""></div>
<h1>TPDDL 1st&ndash;4th July set &mdash; 3-model results vs bot annotation</h1>
<p class=sub>__TOTAL__ images (of __RAWTOTAL__ originally scored) &middot; models: fp32 / ptq_v3 (shipped) / ptq_v2 (old) &middot; normalized compare (uppercase, spaces ignored)</p>

<div class=kpis>__KPIS__</div>

<div class=note><b>Read with care.</b> <span class=mono>bot_serial</span> is the utility bot's own reading, not verified ground truth &mdash; overall agreement with it is only ~50&ndash;54%, so many "wrong" rows may actually be the bot being wrong, not the model.</div>

<div class=note><b>Excluded.</b> __EXPORTN__ images already taken out to <span class=mono>Some_analysis_part/all3_agree_eq_bot_tpddl/</span> (initial "all 3 agree &amp; = bot" export, plus hand-reviewed merges since) are <b>removed from the view below</b> &mdash; only the remaining, still-unresolved images are shown.</div>

<div class=reviewbar>
 <div class=stats>
  <span><b id=stReviewed>0</b> reviewed</span>
  <span><b id=stKeep>0</b> kept</span>
  <span><b id=stDiscard>0</b> discarded</span>
  <span><b id=stWritten>0</b> written</span>
 </div>
 <label><input type=checkbox id=hideReviewed> hide already-reviewed</label>
 <button class=primary id=downloadBtn>Download CSV (<span id=dlCount>0</span>)</button>
 <button class=bad id=downloadDiscardBtn>Download Discarded (<span id=dlDiscardCount>0</span>)</button>
</div>

<div class=gbar>__BUTTONS__</div>
<div class=grid id=grid></div>
<button class=more id=more>Show more</button>

<script>
const DATA=__DATA__, DIR="__IMGDIR__/";
let filt="all", shown=0, PAGE=60;
const grid=document.getElementById('grid'), more=document.getElementById('more');
const lightbox=document.getElementById('lightbox'), lightboximg=document.getElementById('lightboximg');
function openZoom(src){lightboximg.src=src;lightbox.classList.add('on');}
lightbox.onclick=()=>{lightbox.classList.remove('on');lightboximg.src="";};

// ---- review state: one shared store keyed by image name, works across every filter ----
const STORE_KEY = "tpddl_julyset_dashboard_review_v1";
let revState = JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); // { imgName: {status, label} }
const hideReviewed = document.getElementById('hideReviewed');
const revEls = {
  stReviewed: document.getElementById('stReviewed'), stKeep: document.getElementById('stKeep'),
  stDiscard: document.getElementById('stDiscard'), stWritten: document.getElementById('stWritten'),
  dlCount: document.getElementById('dlCount'),
  dlDiscardCount: document.getElementById('dlDiscardCount'),
};
function saveRev(){ localStorage.setItem(STORE_KEY, JSON.stringify(revState)); updateRevStats(); }
function updateRevStats(){
  const entries = Object.values(revState);
  const keep = entries.filter(e=>e.status==='keep').length;
  const discard = entries.filter(e=>e.status==='discard').length;
  const written = entries.filter(e=>e.status==='written').length;
  revEls.stReviewed.textContent = entries.length;
  revEls.stKeep.textContent = keep;
  revEls.stDiscard.textContent = discard;
  revEls.stWritten.textContent = written;
  revEls.dlCount.textContent = keep + written;
  revEls.dlDiscardCount.textContent = discard;
}
document.getElementById('downloadBtn').onclick = ()=>{
  const csvRows = [['filename','ground truth']];
  for(const d of DATA){
    const s = revState[d.img];
    if(!s || s.status==='discard') continue;
    const label = s.status==='keep' ? d.bot : s.label;
    if(!label) continue;
    csvRows.push([d.img, label]);
  }
  const csv = csvRows.map(r=>r.map(c=>`"${String(c).replace(/"/g,'""')}"`).join(',')).join('\\n');
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'tpddl_julyset_dashboard_reviewed.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
};
document.getElementById('downloadDiscardBtn').onclick = ()=>{
  const csvRows = [['filename']];
  for(const d of DATA){
    const s = revState[d.img];
    if(!s || s.status!=='discard') continue;
    csvRows.push([d.img]);
  }
  const csv = csvRows.map(r=>r.map(c=>`"${String(c).replace(/"/g,'""')}"`).join(',')).join('\\n');
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'tpddl_julyset_dashboard_discarded.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
};
hideReviewed.onchange = ()=>render(true);
const PRED = {
  all:      d=>true,
  all3_agree: d=>d.all3,
  all3_neq_bot: d=>d.all3_neq_bot,
  fp32_and_ptqv3_eq_bot: d=>d.fp32_and_ptqv3_ok,
  fp32_eq_bot: d=>d.f_ok,
  fp32_neq_bot: d=>!d.f_ok,
  ptq_v3_eq_bot: d=>d.p3_ok,
  ptq_v3_neq_bot: d=>!d.p3_ok,
  ptq_v2_eq_bot: d=>d.p2_ok,
  ptq_v2_neq_bot: d=>!d.p2_ok,
};
const view=()=>{
 let v = DATA.filter(PRED[filt]);
 if(hideReviewed.checked) v = v.filter(d=>!revState[d.img]);
 return v;
};
function esc(s){return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))||"<em>&empty;</em>";}
function tag(ok){return ok?'<span class=ok>&#10003;</span>':'<span class=no>&#10007;</span>';}
function statusLabel(s){
 if(!s) return '';
 if(s.status==='keep') return `<div class="statustag keep">&#10003; kept &mdash; label: ${esc(s.label)}</div>`;
 if(s.status==='written') return `<div class="statustag written">&#9998; written &mdash; label: ${esc(s.label)}</div>`;
 return `<div class="statustag discard">&#10007; discarded</div>`;
}
function render(reset){
 if(reset){grid.innerHTML="";shown=0;}
 const v=view();
 for(let i=shown;i<Math.min(shown+PAGE,v.length);i++){const d=v[i];
  const src=DIR+encodeURIComponent(d.img);
  const s=revState[d.img];
  const c=document.createElement('div');c.className='card'+(s?' st-'+s.status:'');
  c.innerHTML=`<img loading=lazy src="${src}" alt="" title="click to zoom">
   <div class=b>
    <div class=namerow><code title="${d.img}">${d.img}</code><button class=cp>copy</button></div>
    <div class=row><span class=k>bot</span><span class=mono>${esc(d.bot)}</span></div>
    <div class=row><span class=k>fp32</span><span class=mono>${esc(d.fp32)} ${tag(d.f_ok)}</span></div>
    <div class=row><span class=k>ptq_v3</span><span class=mono>${esc(d.ptq_v3)} ${tag(d.p3_ok)}</span></div>
    <div class=row><span class=k>ptq_v2</span><span class=mono>${esc(d.ptq_v2)} ${tag(d.p2_ok)}</span></div>
    <div class=revrow>
     <button class="keep${s&&s.status==='keep'?' active':''}">&#10003; Keep</button>
     <button class="discard${s&&s.status==='discard'?' active':''}">&#10007; Discard</button>
    </div>
    <div class=writerow>
     <input type=text placeholder="write true label" value="${s&&s.status==='written'?esc(s.label):''}">
     <button class=save>Save</button>
    </div>
    <div class=statuswrap>${statusLabel(s)}</div>
   </div>`;
  c.querySelector('img').onclick=()=>openZoom(src);
  const cpBtn=c.querySelector('.cp');
  cpBtn.onclick=()=>{navigator.clipboard.writeText(d.img).then(()=>{cpBtn.textContent='✓';cpBtn.classList.add('done');
    setTimeout(()=>{cpBtn.textContent='copy';cpBtn.classList.remove('done');},1200);});};

  const keepBtn=c.querySelector('.keep'), discardBtn=c.querySelector('.discard');
  const input=c.querySelector('.writerow input'), saveBtn=c.querySelector('.writerow .save');
  const statusWrap=c.querySelector('.statuswrap');
  function applyStatus(newState){
   revState[d.img]=newState; saveRev();
   c.className='card st-'+newState.status;
   keepBtn.classList.toggle('active', newState.status==='keep');
   discardBtn.classList.toggle('active', newState.status==='discard');
   statusWrap.innerHTML=statusLabel(newState);
   if(hideReviewed.checked){ c.remove(); }
  }
  keepBtn.onclick=()=>applyStatus({status:'keep', label:d.bot});
  discardBtn.onclick=()=>applyStatus({status:'discard', label:''});
  saveBtn.onclick=()=>{const val=input.value.trim(); if(!val) return; applyStatus({status:'written', label:val});};
  input.addEventListener('keydown', e=>{ if(e.key==='Enter'){ e.preventDefault(); saveBtn.click(); } });

  grid.appendChild(c);}
 shown=Math.min(shown+PAGE,v.length);
 more.style.display=shown>=v.length?'none':'block';
}
document.querySelectorAll('[data-f]').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('[data-f]').forEach(x=>x.classList.remove('on'));
 b.classList.add('on');filt=b.dataset.f;render(true);});
more.onclick=()=>render(false);
updateRevStats();
render(true);
</script>
</body></html>"""

subst = {
    "__TOTAL__": f"{total:,}",
    "__RAWTOTAL__": f"{len(all_rows):,}",
    "__KPIS__": kpi_html,
    "__BUTTONS__": btn_html,
    "__DATA__": DATA,
    "__IMGDIR__": IMGDIR,
    "__EXPORTN__": f"{EXCLUDED_N:,}",
}
out = tpl
for k, v in subst.items():
    out = out.replace(k, v)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(out)

print("Counts:")
for k, label in FILTERS:
    print(f"  {label:22s} {counts[k]:5,} / {total:,}  ({pct(counts[k])}%)")
print("\nwrote", OUT)
