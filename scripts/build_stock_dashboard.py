#!/usr/bin/env python3
"""TEMPORARY dashboard for the STOCK PP-OCRv5 det+rec results on the 18,544 set.

Categorizes each image by edit distance (stock prediction vs corrected GT):
exact / 1-char / 2-char / wrong(3+). Shows the real meter image + GT + stock read.
Output: 26788 images for test/agreed_fp32_oldptq_qat/TEMP_stock_dashboard.html
"""
import csv, json, os
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "26788 images for test")
CDIR = os.path.join(BASE, "agreed_fp32_oldptq_qat")
SRC = os.path.join(CDIR, "images")
SCORED = os.path.join(CDIR, "stock_fullpipe_scored.csv")
OUT = os.path.join(CDIR, "TEMP_stock_dashboard.html")

rows = []
counts = {0: 0, 1: 0, 2: 0, 3: 0}
for r in csv.DictReader(open(SCORED)):
    d = int(r["edit_distance"]); k = d if d < 3 else 3
    counts[k] += 1
    rows.append({"f": r["filename"], "gt": r["ground_truth"], "p": r["stock_pred"],
                 "s": r["score"], "d": d, "k": k})
rows.sort(key=lambda x: (x["k"], x["d"], x["f"]))
total = len(rows)
def pct(n): return f"{100.0*n/total:.2f}%"
BASE_URL = "file://" + quote(SRC) + "/"

html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TEMP — Stock PP-OCRv5 det+rec vs Consensus (18,544)</title>
<style>
:root{--bg:#0f1115;--card:#1a1d24;--line:#2a2e37;--txt:#e6e8ec;--mut:#9aa2af;
--exact:#2ecc71;--one:#f1c40f;--two:#e67e22;--wrong:#e74c3c;--accent:#4aa3ff;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--txt)}
header{padding:20px 26px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:19px}
.warn{background:#3a2a00;border:1px solid #7a5b00;color:#ffd479;padding:8px 12px;border-radius:8px;margin-top:8px;font-size:12.5px}
.sub{color:var(--mut);font-size:13px}
.cards{display:flex;flex-wrap:wrap;gap:12px;padding:18px 26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px 17px;min-width:150px;flex:1}
.card .n{font-size:25px;font-weight:700}
.card .l{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.card.exact{border-left:4px solid var(--exact)}.card.one{border-left:4px solid var(--one)}
.card.two{border-left:4px solid var(--two)}.card.wrong{border-left:4px solid var(--wrong)}
.tabs{display:flex;gap:8px;padding:0 26px;flex-wrap:wrap}
.tab{background:var(--card);border:1px solid var(--line);color:var(--txt);padding:9px 16px;border-radius:9px 9px 0 0;cursor:pointer;font-size:13px}
.tab.active{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;padding:18px 26px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.tile img{width:100%;height:120px;object-fit:contain;background:#000;border-bottom:1px solid var(--line);cursor:zoom-in}
.body{padding:10px 12px;font-size:12.5px}
.badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;color:#111}
.b0{background:var(--exact)}.b1{background:var(--one)}.b2{background:var(--two)}.b3{background:var(--wrong);color:#fff}
.kv{display:flex;justify-content:space-between;gap:8px;padding:2px 0;border-top:1px dashed var(--line)}
.kv .k{color:var(--mut)}.kv .v{font-family:ui-monospace,Menlo,monospace;font-weight:600}
.gt .v{color:var(--exact)}.match .v{color:var(--exact)}.nomatch .v{color:var(--wrong)}
.link{display:flex;gap:6px;align-items:center;margin-top:8px;padding-top:8px;border-top:1px dashed var(--line)}
.link input{flex:1;min-width:0;background:#0b0d11;border:1px solid var(--line);color:var(--mut);border-radius:6px;padding:5px 7px;font:11px ui-monospace,Menlo,monospace}
.link a{color:var(--accent);text-decoration:none;font-size:11px;white-space:nowrap}
.copy{background:var(--accent);border:none;color:#fff;border-radius:6px;padding:5px 10px;font-size:11px;font-weight:600;cursor:pointer}
.copy.ok{background:var(--exact)}
.more{display:block;margin:8px auto 40px;padding:11px 26px;background:var(--card);border:1px solid var(--line);color:var(--txt);border-radius:9px;cursor:pointer}
.note{color:var(--mut);font-size:12px;padding:0 26px 6px}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.9);display:none;align-items:center;justify-content:center;z-index:50;cursor:zoom-out}#lb img{max-width:94%;max-height:94%}
</style></head><body>
<header>
<h1>Stock PP-OCRv5 (server_det + server_rec, un-fine-tuned) — vs Consensus GT</h1>
<div class="sub">Full detector+recognizer pipeline, ~167 MB, NOT trained on meter data. Scored vs corrected consensus GT (space-insensitive). 18,544 images.</div>
<div class="warn">⚠ TEMPORARY dashboard. This is the baseline stock model — its 71.85% exact shows accuracy <b>before</b> fine-tuning.</div>
</header>
<div class="cards">
  <div class="card exact"><div class="n">__EXACT__</div><div class="l">Exact match</div><div class="sub">__EXACT_P__</div></div>
  <div class="card one"><div class="n">__ONE__</div><div class="l">1 char off</div><div class="sub">__ONE_P__ · cum __CUM1__</div></div>
  <div class="card two"><div class="n">__TWO__</div><div class="l">2 char off</div><div class="sub">__TWO_P__ · cum __CUM2__</div></div>
  <div class="card wrong"><div class="n">__WRONG__</div><div class="l">Wrong 3+</div><div class="sub">__WRONG_P__</div></div>
  <div class="card"><div class="n">__TOTAL__</div><div class="l">Total imgs</div><div class="sub">18,544 eval set</div></div>
</div>
<div class="tabs">
  <div class="tab active" data-k="0" onclick="pick(0,this)">Exact · __EXACT__</div>
  <div class="tab" data-k="1" onclick="pick(1,this)">1-char · __ONE__</div>
  <div class="tab" data-k="2" onclick="pick(2,this)">2-char · __TWO__</div>
  <div class="tab" data-k="3" onclick="pick(3,this)">Wrong 3+ · __WRONG__</div>
</div>
<div class="note">GT = consensus ground truth (green). Stock read is green if it matches, red otherwise. Click image to zoom. Copyable file:// link per tile.</div>
<div id="grid" class="grid"></div>
<button id="more" class="more" onclick="render()">Load more</button>
<div id="lb" onclick="this.style.display='none'"><img id="lbi" src=""></div>
<script>
const DATA = __DATA__;
const BASE = "__BASE__";
let curK=0, shown=0, PAGE=60, filtered=[];
function pick(k, el){
  curK=k; shown=0;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active')); el.classList.add('active');
  document.getElementById('grid').innerHTML=''; filtered=DATA.filter(r=>r.k===k); render();
}
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function copyLink(btn){
  const url=btn.parentElement.querySelector('input').value;
  const done=()=>{const o=btn.textContent;btn.textContent='Copied';btn.classList.add('ok');setTimeout(()=>{btn.textContent=o;btn.classList.remove('ok');},1200);};
  if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(url).then(done).catch(()=>fb(btn,done));}else fb(btn,done);
}
function fb(btn,done){const i=btn.parentElement.querySelector('input');i.focus();i.select();try{document.execCommand('copy');done();}catch(e){btn.textContent='Ctrl+C';}}
function zoom(s){document.getElementById('lbi').src=s;document.getElementById('lb').style.display='flex';}
function render(){
  const g=document.getElementById('grid');
  for(const r of filtered.slice(shown,shown+PAGE)){
    const label=r.d===0?'EXACT':(r.d+' off');
    const cls=r.d===0?'match':'nomatch';
    const url=BASE+encodeURIComponent(r.f);
    const div=document.createElement('div');div.className='tile';
    div.innerHTML=
      '<img loading="lazy" src="images/'+encodeURIComponent(r.f)+'" onclick="zoom(this.src)">'+
      '<div class="body">'+
        '<span class="badge b'+r.k+'">'+label+'</span>'+
        '<div class="kv gt"><span class="k">GT (consensus)</span><span class="v">'+esc(r.gt)+'</span></div>'+
        '<div class="kv '+cls+'"><span class="k">Stock read</span><span class="v">'+esc(r.p)+'</span></div>'+
        '<div class="kv"><span class="k">conf</span><span class="v">'+esc(r.s)+'</span></div>'+
        '<div class="link"><input readonly value="'+url+'" onclick="this.select()">'+
        '<button class="copy" onclick="copyLink(this)">Copy</button>'+
        '<a href="'+url+'" target="_blank" rel="noopener">Open ↗</a></div>'+
      '</div>';
    g.appendChild(div);
  }
  shown+=Math.min(PAGE,filtered.length-shown);
  document.getElementById('more').style.display= shown<filtered.length?'block':'none';
}
pick(0, document.querySelector('.tab'));
</script>
</body></html>"""

html = (html
        .replace("__BASE__", BASE_URL)
        .replace("__DATA__", json.dumps(rows, separators=(",", ":")))
        .replace("__TOTAL__", str(total))
        .replace("__EXACT_P__", pct(counts[0])).replace("__EXACT__", str(counts[0]))
        .replace("__ONE_P__", pct(counts[1])).replace("__ONE__", str(counts[1]))
        .replace("__TWO_P__", pct(counts[2])).replace("__TWO__", str(counts[2]))
        .replace("__WRONG_P__", pct(counts[3])).replace("__WRONG__", str(counts[3]))
        .replace("__CUM1__", pct(counts[0]+counts[1]))
        .replace("__CUM2__", pct(counts[0]+counts[1]+counts[2])))

with open(OUT, "w") as f:
    f.write(html)
print("Wrote:", OUT)
print(f"total={total} exact={counts[0]} one={counts[1]} two={counts[2]} wrong={counts[3]}")
