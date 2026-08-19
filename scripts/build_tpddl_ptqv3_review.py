#!/usr/bin/env python3
"""
Build a single-image, keyboard-driven review tool for the TPDDL July set images
where PTQ v3 (the shipped model) disagrees with the bot (ptq_v3 != bot),
regardless of what fp32/ptq_v2 said -> 3,441 images.

Input:  Some_analysis_part/merged_3models_vs_bot_tpddl.csv
Images: Some_analysis_part/1stjuly_4thjuly_2026_tpddl/  (the crop fed to the model)

Output: Some_analysis_part/tpddl_ptqv3_review.html   (open in a browser)

How it works (all client-side, no server needed):
  - One image at a time, large, click-to-zoom.
  - Keyboard: C = correct (accept bot's value as the true label), I = incorrect
    (focuses the correction box so you type the true serial), Enter = save
    correction & advance, LeftArrow/RightArrow = navigate without judging,
    N/P = next/prev, S = skip.
  - Progress auto-saves to the browser's localStorage after every action, so a
    refresh / accidental tab close does not lose your work.
  - "Download CSV" (button, always available) exports every REVIEWED row so
    far as `filename, ground truth` (same 2-column format used elsewhere in
    this project for retraining, e.g. `retraining dataset with label/`).

Usage:
    python3 scripts/build_tpddl_ptqv3_review.py
"""
import os
import csv
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "Some_analysis_part")
SRC = os.path.join(BASE, "merged_3models_vs_bot_tpddl.csv")
IMGDIR = "1stjuly_4thjuly_2026_tpddl"  # relative to BASE
OUT = os.path.join(BASE, "tpddl_ptqv3_review.html")

with open(SRC, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

data = []
for r in rows:
    if r["ptq_v3_eq_bot"] != "0":
        continue
    data.append({
        "img": r["image_name"],
        "fp32": r["fp32"], "ptq_v3": r["ptq_v3"], "ptq_v2": r["ptq_v2"],
        "bot": r["bot_serial"],
    })

print(f"images to review: {len(data)}")
DATA = json.dumps(data)

tpl = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>TPDDL — review: PTQ v3 disagrees with bot</title>
<style>
:root{color-scheme:light dark;
 --bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--mut:#52514e;--line:#e6e5e2;
 --good:#0ca30c;--goodink:#006300;--bad:#d03b3b;--badink:#8a1f1f;--warn:#fab219;--warnink:#7a5200;
 --accent:#2f6fed;}
@media(prefers-color-scheme:dark){:root{
 --bg:#141413;--card:#1f1f1e;--ink:#f5f5f2;--mut:#b4b3ab;--line:#33322f;
 --good:#3ec13e;--goodink:#8ff08f;--bad:#ef6a6a;--badink:#ffb0b0;--warn:#fab219;--warnink:#ffd98a;
 --accent:#6c9bff;}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
 background:var(--bg);color:var(--ink);padding:20px clamp(14px,4vw,48px) 60px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 18px;font-size:13.5px}

.topbar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:18px}
.pbar{flex:1;min-width:220px;height:10px;border-radius:6px;background:var(--line);overflow:hidden}
.pbar i{display:block;height:100%;background:var(--accent)}
.stats{display:flex;gap:16px;font-size:13px;color:var(--mut);flex-wrap:wrap}
.stats b{color:var(--ink)}
button{font:inherit;background:var(--card);color:var(--ink);
 border:1px solid var(--line);border-radius:8px;padding:8px 16px;cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
button.good{background:var(--good);color:#fff;border-color:var(--good)}
button.bad{background:var(--bad);color:#fff;border-color:var(--bad)}
button:disabled{opacity:.4;cursor:default}

.wrap{display:grid;grid-template-columns:minmax(280px,460px) 1fr;gap:26px;align-items:start}
@media(max-width:820px){.wrap{grid-template-columns:1fr}}
.imgcard{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;text-align:center}
.imgcard img{max-width:100%;max-height:320px;background:#000;border-radius:8px;cursor:zoom-in;object-fit:contain}
.namerow{display:flex;align-items:center;gap:8px;margin-top:12px}
.namerow code{flex:1;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--mut);
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:left}
.cp{flex:none;font-size:11px;padding:4px 10px;border-radius:6px;border:1px solid var(--line);
 background:var(--bg);color:var(--ink);cursor:pointer}
.cp.done{background:var(--good);color:#fff;border-color:var(--good)}

.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
.kv{display:flex;justify-content:space-between;gap:10px;padding:9px 0;border-bottom:1px solid var(--line)}
.kv:last-of-type{border-bottom:none}
.kv .k{color:var(--mut);font-size:13px}
.kv .v{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:15px;font-weight:600}
.kv.bot .v{color:var(--warnink)}

.actions{display:flex;gap:10px;margin:18px 0 12px;flex-wrap:wrap}
.correction{display:flex;gap:8px;margin-bottom:14px}
.correction input{flex:1;font:inherit;font-family:ui-monospace,Menlo,Consolas,monospace;
 padding:10px 12px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.status{font-size:13px;padding:8px 12px;border-radius:8px;margin-bottom:14px;display:none}
.status.correct{display:block;background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--goodink)}
.status.incorrect{display:block;background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--badink)}
.nav{display:flex;gap:10px;justify-content:space-between;margin-top:16px}
.keys{margin-top:20px;font-size:12px;color:var(--mut);line-height:1.9}
.keys kbd{background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:1px 7px;
 font-family:ui-monospace,Menlo,Consolas,monospace}

.lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:99;
 align-items:center;justify-content:center;cursor:zoom-out;padding:30px}
.lightbox.on{display:flex}
.lightbox img{max-width:100%;max-height:100%;border-radius:8px}
</style></head><body>
<div class=lightbox id=lightbox><img id=lightboximg src="" alt=""></div>

<h1>TPDDL July set &mdash; review: PTQ v3 (shipped) &ne; bot</h1>
<p class=sub>__TOTAL__ images where the shipped PTQ v3 model disagrees with the bot reading (fp32/ptq_v2 may or may not agree with it). Judge whether the bot value is actually right, or type the true serial.</p>

<div class=topbar>
 <div class=pbar><i id=pbarfill style="width:0%"></i></div>
 <div class=stats>
  <span><b id=stReviewed>0</b>/__TOTAL__ reviewed</span>
  <span class=good><b id=stCorrect>0</b> correct</span>
  <span class=bad><b id=stIncorrect>0</b> incorrect</span>
 </div>
 <button class=primary id=downloadBtn>Download CSV (<span id=dlCount>0</span>)</button>
</div>

<div class=wrap>
 <div class=imgcard>
  <img id=img src="" alt="" title="click to zoom">
  <div class=namerow><code id=fname></code><button class=cp id=cpBtn>copy</button></div>
 </div>

 <div class=panel>
  <div class="kv bot"><span class=k>bot reading</span><span class=v id=vBot></span></div>
  <div class=kv><span class=k>fp32</span><span class=v id=vFp32></span></div>
  <div class=kv><span class=k>ptq_v3</span><span class=v id=vPtq3></span></div>
  <div class=kv><span class=k>ptq_v2</span><span class=v id=vPtq2></span></div>

  <div class=status id=statusBox></div>

  <div class=actions>
   <button class=good id=btnCorrect>&#10003; Correct <kbd>C</kbd></button>
   <button class=bad id=btnIncorrect>&#10007; Incorrect <kbd>I</kbd></button>
  </div>
  <div class=correction>
   <input id=correctionInput type=text placeholder="type the true serial, then Enter" autocomplete=off>
   <button id=saveCorrection>Save <kbd>&crarr;</kbd></button>
  </div>

  <div class=nav>
   <button id=prevBtn>&larr; Prev <kbd>P</kbd></button>
   <button id=skipBtn>Skip <kbd>S</kbd></button>
   <button id=nextBtn>Next &rarr; <kbd>N</kbd></button>
  </div>

  <div class=keys>
   <div><kbd>C</kbd> mark correct (bot value is right) &amp; advance</div>
   <div><kbd>I</kbd> mark incorrect &amp; focus the correction box</div>
   <div><kbd>Enter</kbd> (in box) save correction &amp; advance</div>
   <div><kbd>&larr;</kbd>/<kbd>&rarr;</kbd> or <kbd>P</kbd>/<kbd>N</kbd> navigate</div>
   <div><kbd>S</kbd> skip without judging</div>
  </div>
 </div>
</div>

<script>
const DATA=__DATA__, DIR="__IMGDIR__/";
const STORE_KEY = "tpddl_ptqv3_review_v1";
let state = JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); // { imgName: {status, label} }
let idx = 0;

const els = {
  img: document.getElementById('img'), fname: document.getElementById('fname'),
  vBot: document.getElementById('vBot'), vFp32: document.getElementById('vFp32'),
  vPtq3: document.getElementById('vPtq3'), vPtq2: document.getElementById('vPtq2'),
  statusBox: document.getElementById('statusBox'),
  correctionInput: document.getElementById('correctionInput'),
  pbarfill: document.getElementById('pbarfill'),
  stReviewed: document.getElementById('stReviewed'), stCorrect: document.getElementById('stCorrect'),
  stIncorrect: document.getElementById('stIncorrect'), dlCount: document.getElementById('dlCount'),
  lightbox: document.getElementById('lightbox'), lightboximg: document.getElementById('lightboximg'),
};

function save(){ localStorage.setItem(STORE_KEY, JSON.stringify(state)); updateStats(); }

function updateStats(){
  const entries = Object.values(state);
  const reviewed = entries.length;
  const correct = entries.filter(e=>e.status==='correct').length;
  const incorrect = entries.filter(e=>e.status==='incorrect').length;
  els.stReviewed.textContent = reviewed;
  els.stCorrect.textContent = correct;
  els.stIncorrect.textContent = incorrect;
  els.dlCount.textContent = reviewed;
  els.pbarfill.style.width = (100*reviewed/DATA.length)+'%';
}

function render(){
  const d = DATA[idx];
  els.img.src = DIR+encodeURIComponent(d.img);
  els.fname.textContent = d.img; els.fname.title = d.img;
  els.vBot.textContent = d.bot || '∅';
  els.vFp32.textContent = d.fp32 || '∅';
  els.vPtq3.textContent = d.ptq_v3 || '∅';
  els.vPtq2.textContent = d.ptq_v2 || '∅';
  const s = state[d.img];
  els.correctionInput.value = (s && s.status==='incorrect') ? s.label : '';
  els.statusBox.className = 'status' + (s ? ' '+s.status : '');
  els.statusBox.textContent = s ? (s.status==='correct' ? '✓ marked correct (label = bot value)' : `✗ marked incorrect — label: ${s.label||'(empty)'}`) : '';
  document.title = `(${idx+1}/${DATA.length}) TPDDL review`;
}

function goto(i){ idx = Math.max(0, Math.min(DATA.length-1, i)); render(); }
function next(){ goto(idx+1); }
function prev(){ goto(idx-1); }

function markCorrect(){
  const d = DATA[idx];
  state[d.img] = {status:'correct', label:d.bot};
  save(); next();
}
function markIncorrectFocus(){
  const d = DATA[idx];
  if(!state[d.img] || state[d.img].status!=='incorrect') state[d.img] = {status:'incorrect', label:''};
  save(); render();
  els.correctionInput.focus();
}
function saveCorrection(){
  const d = DATA[idx];
  const val = els.correctionInput.value.trim();
  if(!val) return;
  state[d.img] = {status:'incorrect', label:val};
  save(); next();
}

document.getElementById('btnCorrect').onclick = markCorrect;
document.getElementById('btnIncorrect').onclick = markIncorrectFocus;
document.getElementById('saveCorrection').onclick = saveCorrection;
document.getElementById('prevBtn').onclick = prev;
document.getElementById('nextBtn').onclick = next;
document.getElementById('skipBtn').onclick = next;
document.getElementById('cpBtn').onclick = ()=>{
  const btn = document.getElementById('cpBtn');
  navigator.clipboard.writeText(DATA[idx].img).then(()=>{
    btn.textContent='✓'; btn.classList.add('done');
    setTimeout(()=>{btn.textContent='copy'; btn.classList.remove('done');},1200);
  });
};
els.img.onclick = ()=>{ els.lightboximg.src = els.img.src; els.lightbox.classList.add('on'); };
els.lightbox.onclick = ()=>{ els.lightbox.classList.remove('on'); els.lightboximg.src=''; };

els.correctionInput.addEventListener('keydown', e=>{
  if(e.key==='Enter'){ e.preventDefault(); saveCorrection(); }
});

document.addEventListener('keydown', e=>{
  if(document.activeElement === els.correctionInput) return; // let the input handle its own keys
  switch(e.key){
    case 'c': case 'C': markCorrect(); break;
    case 'i': case 'I': markIncorrectFocus(); break;
    case 'ArrowRight': case 'n': case 'N': next(); break;
    case 'ArrowLeft': case 'p': case 'P': prev(); break;
    case 's': case 'S': next(); break;
  }
});

document.getElementById('downloadBtn').onclick = ()=>{
  const rows = [['filename','ground truth']];
  for(const d of DATA){
    const s = state[d.img];
    if(!s) continue;
    const label = s.status==='correct' ? d.bot : s.label;
    if(!label) continue;
    rows.push([d.img, label]);
  }
  const csv = rows.map(r=>r.map(c=>`"${String(c).replace(/"/g,'""')}"`).join(',')).join('\\n');
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'tpddl_ptqv3_reviewed.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

updateStats();
render();
</script>
</body></html>"""

subst = {
    "__TOTAL__": f"{len(data):,}",
    "__DATA__": DATA,
    "__IMGDIR__": IMGDIR,
}
out = tpl
for k, v in subst.items():
    out = out.replace(k, v)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(out)

print("wrote", OUT)
