#!/usr/bin/env python3
"""
Generate a Keep/Discard/Write review page for EVERY filter on the main TPDDL
July-set dashboard (tpddl_julyset_dashboard.html), so each filter has its own
dedicated hand-labeling tool instead of just a read-only grid.

Input:  Some_analysis_part/merged_3models_vs_bot_tpddl.csv
Images already taken out to Some_analysis_part/all3_agree_eq_bot_tpddl/ are
excluded from every filter here (same rule as the main dashboard).

Output: Some_analysis_part/tpddl_review_<filter_key>.html   (one per filter)

Each page: one image at a time, click-to-zoom, K=Keep (label=bot value),
D=Discard (excluded from export), W=focus a box to type a different true
label (Enter to save). Progress -> that page's own localStorage key. A
"Download CSV" button exports Keep/Written rows as `filename, ground truth`.

Usage:
    python3 scripts/build_tpddl_filter_reviews.py
"""
import os
import csv
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "Some_analysis_part")
SRC = os.path.join(BASE, "merged_3models_vs_bot_tpddl.csv")
IMGDIR = "1stjuly_4thjuly_2026_tpddl"
TAKEN_OUT_CSV = os.path.join(BASE, "all3_agree_eq_bot_tpddl", "ground_truth.csv")
DISCARDED_CSV = os.path.join(BASE, "discarded_tpddl.csv")

with open(SRC, newline="", encoding="utf-8") as f:
    all_rows = list(csv.DictReader(f))
with open(TAKEN_OUT_CSV, newline="", encoding="utf-8") as f:
    taken_out = {r["filename"] for r in csv.DictReader(f)}
discarded = set()
if os.path.exists(DISCARDED_CSV):
    with open(DISCARDED_CSV, newline="", encoding="utf-8") as f:
        discarded = {r["filename"] for r in csv.DictReader(f)}

rows = [r for r in all_rows if r["image_name"] not in taken_out and r["image_name"] not in discarded]

# ---- filter predicates, identical semantics to build_tpddl_julyset_dashboard.py ----
def flags(r):
    all3 = r["all_agree"] == "yes"
    f_ok = r["fp32_eq_bot"] == "1"
    p3_ok = r["ptq_v3_eq_bot"] == "1"
    p2_ok = r["ptq_v2_eq_bot"] == "1"
    all3_neq_bot = (not f_ok) and (not p3_ok) and (not p2_ok)
    return all3, f_ok, p3_ok, p2_ok, all3_neq_bot


FILTERS = [
    ("all", "All", lambda a, f, p3, p2, n: True),
    ("all3_agree", "All 3 agree", lambda a, f, p3, p2, n: a),
    ("all3_neq_bot", "All 3 ≠ bot (none match)", lambda a, f, p3, p2, n: n),
    ("fp32_and_ptqv3_eq_bot", "FP32 & PTQ v3 both = bot", lambda a, f, p3, p2, n: f and p3),
    ("fp32_eq_bot", "FP32 = bot", lambda a, f, p3, p2, n: f),
    ("fp32_neq_bot", "FP32 ≠ bot", lambda a, f, p3, p2, n: not f),
    ("ptq_v3_eq_bot", "PTQ v3 = bot", lambda a, f, p3, p2, n: p3),
    ("ptq_v3_neq_bot", "PTQ v3 ≠ bot", lambda a, f, p3, p2, n: not p3),
    ("ptq_v2_eq_bot", "PTQ v2 = bot", lambda a, f, p3, p2, n: p2),
    ("ptq_v2_neq_bot", "PTQ v2 ≠ bot", lambda a, f, p3, p2, n: not p2),
]

TPL = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>TPDDL — review: __LABEL__</title>
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
a.back{color:var(--accent);text-decoration:none;font-size:13px}

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
.status.keep{display:block;background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--goodink)}
.status.discard{display:block;background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--badink)}
.status.written{display:block;background:color-mix(in srgb,var(--accent) 16%,transparent);color:var(--accent)}
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

<p><a class=back href="tpddl_julyset_dashboard.html">&larr; back to dashboard</a></p>
<h1>TPDDL July set &mdash; review: __LABEL__</h1>
<p class=sub>__TOTAL__ images matching this filter. Keep to accept the bot value, Discard to drop, or write a different true label.</p>

<div class=topbar>
 <div class=pbar><i id=pbarfill style="width:0%"></i></div>
 <div class=stats>
  <span><b id=stReviewed>0</b>/__TOTAL__ reviewed</span>
  <span class=good><b id=stKeep>0</b> kept</span>
  <span class=bad><b id=stDiscard>0</b> discarded</span>
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
   <button class=good id=btnKeep>&#10003; Keep <kbd>K</kbd></button>
   <button class=bad id=btnDiscard>&#10007; Discard <kbd>D</kbd></button>
  </div>
  <div class=correction>
   <input id=writeInput type=text placeholder="write a different true label, then Enter" autocomplete=off>
   <button id=saveWrite>Save <kbd>&crarr;</kbd></button>
  </div>

  <div class=nav>
   <button id=prevBtn>&larr; Prev <kbd>P</kbd></button>
   <button id=skipBtn>Skip <kbd>S</kbd></button>
   <button id=nextBtn>Next &rarr; <kbd>N</kbd></button>
  </div>

  <div class=keys>
   <div><kbd>K</kbd> keep (label = bot value) &amp; advance</div>
   <div><kbd>D</kbd> discard (exclude from export) &amp; advance</div>
   <div><kbd>W</kbd> focus the write box to type a different label</div>
   <div><kbd>Enter</kbd> (in box) save written label &amp; advance</div>
   <div><kbd>&larr;</kbd>/<kbd>&rarr;</kbd> or <kbd>P</kbd>/<kbd>N</kbd> navigate</div>
   <div><kbd>S</kbd> skip without judging</div>
  </div>
 </div>
</div>

<script>
const DATA=__DATA__, DIR="__IMGDIR__/";
const STORE_KEY = "__STORE_KEY__";
let state = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
let idx = 0;

const els = {
  img: document.getElementById('img'), fname: document.getElementById('fname'),
  vBot: document.getElementById('vBot'), vFp32: document.getElementById('vFp32'),
  vPtq3: document.getElementById('vPtq3'), vPtq2: document.getElementById('vPtq2'),
  statusBox: document.getElementById('statusBox'),
  writeInput: document.getElementById('writeInput'),
  pbarfill: document.getElementById('pbarfill'),
  stReviewed: document.getElementById('stReviewed'), stKeep: document.getElementById('stKeep'),
  stDiscard: document.getElementById('stDiscard'), dlCount: document.getElementById('dlCount'),
  lightbox: document.getElementById('lightbox'), lightboximg: document.getElementById('lightboximg'),
};

function save(){ localStorage.setItem(STORE_KEY, JSON.stringify(state)); updateStats(); }

function updateStats(){
  const entries = Object.values(state);
  const reviewed = entries.length;
  const keep = entries.filter(e=>e.status==='keep' || e.status==='written').length;
  const discard = entries.filter(e=>e.status==='discard').length;
  els.stReviewed.textContent = reviewed;
  els.stKeep.textContent = keep;
  els.stDiscard.textContent = discard;
  els.dlCount.textContent = keep;
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
  els.writeInput.value = (s && s.status==='written') ? s.label : '';
  els.statusBox.className = 'status' + (s ? ' '+s.status : '');
  els.statusBox.textContent = s ? (
    s.status==='keep' ? '✓ kept (label = bot value)' :
    s.status==='written' ? `✎ written — label: ${s.label||'(empty)'}` :
    '✗ discarded — excluded from export'
  ) : '';
  document.title = `(${idx+1}/${DATA.length}) TPDDL review`;
}

function goto(i){ idx = Math.max(0, Math.min(DATA.length-1, i)); render(); }
function next(){ goto(idx+1); }
function prev(){ goto(idx-1); }

function markKeep(){
  const d = DATA[idx];
  state[d.img] = {status:'keep', label:d.bot};
  save(); next();
}
function markDiscard(){
  const d = DATA[idx];
  state[d.img] = {status:'discard', label:''};
  save(); next();
}
function focusWrite(){ els.writeInput.focus(); }
function saveWrite(){
  const d = DATA[idx];
  const val = els.writeInput.value.trim();
  if(!val) return;
  state[d.img] = {status:'written', label:val};
  save(); next();
}

document.getElementById('btnKeep').onclick = markKeep;
document.getElementById('btnDiscard').onclick = markDiscard;
document.getElementById('saveWrite').onclick = saveWrite;
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

els.writeInput.addEventListener('keydown', e=>{
  if(e.key==='Enter'){ e.preventDefault(); saveWrite(); }
});

document.addEventListener('keydown', e=>{
  if(document.activeElement === els.writeInput) return;
  switch(e.key){
    case 'k': case 'K': markKeep(); break;
    case 'd': case 'D': markDiscard(); break;
    case 'w': case 'W': focusWrite(); break;
    case 'ArrowRight': case 'n': case 'N': next(); break;
    case 'ArrowLeft': case 'p': case 'P': prev(); break;
    case 's': case 'S': next(); break;
  }
});

document.getElementById('downloadBtn').onclick = ()=>{
  const rows = [['filename','ground truth']];
  for(const d of DATA){
    const s = state[d.img];
    if(!s || s.status==='discard') continue;
    const label = s.label;
    if(!label) continue;
    rows.push([d.img, label]);
  }
  const csv = rows.map(r=>r.map(c=>`"${String(c).replace(/"/g,'""')}"`).join(',')).join('\\n');
  const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = '__DLNAME__';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

updateStats();
render();
</script>
</body></html>"""


def build_one(key, label, pred):
    data = []
    for r in rows:
        a, f, p3, p2, n = flags(r)
        if not pred(a, f, p3, p2, n):
            continue
        data.append({
            "img": r["image_name"],
            "fp32": r["fp32"], "ptq_v3": r["ptq_v3"], "ptq_v2": r["ptq_v2"],
            "bot": r["bot_serial"],
        })

    out_path = os.path.join(BASE, f"tpddl_review_{key}.html")
    subst = {
        "__LABEL__": label,
        "__TOTAL__": f"{len(data):,}",
        "__DATA__": json.dumps(data),
        "__IMGDIR__": IMGDIR,
        "__STORE_KEY__": f"tpddl_review_{key}_v1",
        "__DLNAME__": f"tpddl_review_{key}_reviewed.csv",
    }
    out = TPL
    for k, v in subst.items():
        out = out.replace(k, v)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    return len(data), out_path


if __name__ == "__main__":
    print(f"remaining pool (already-taken-out excluded): {len(rows):,}")
    for key, label, pred in FILTERS:
        n, path = build_one(key, label, pred)
        print(f"  {label:28s} {n:5,}  -> {os.path.basename(path)}")
