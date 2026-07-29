#!/usr/bin/env python3
"""TEMPORARY verification dashboard for the CORRECTED new-PTQ vs consensus results.

Reads the corrected CSVs, shows every image with all model outputs + corrected GT,
and adds a "Fixed" tab isolating the rows that changed from the old (buggy) scoring
so the user can spot-check the leading-zero / space fixes before the full re-run.

Images are served from the full source folder  26788 images for test/images/
(so the 282 newly-added consensus images display too).
Output: 26788 images for test/agreed_fp32_oldptq_qat/TEMP_verify_dashboard.html
"""
import csv, json, os
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "26788 images for test")
CDIR = os.path.join(BASE, "agreed_fp32_oldptq_qat")
SRC_IMGS = os.path.join(BASE, "images")                  # all 26,787 source images
FOUR = os.path.join(BASE, "four_model_results.csv")
CORR = os.path.join(CDIR, "new_ptq_vs_groundtruth_corrected.csv")
OLD = os.path.join(CDIR, "new_ptq_vs_groundtruth.csv")    # previous buggy scoring
OUT = os.path.join(CDIR, "TEMP_verify_dashboard.html")

four = {}
with open(FOUR) as f:
    for r in csv.DictReader(f):
        four[r["filename"]] = r

# old scoring: filename -> old match (to detect what got fixed)
old_match = {}
with open(OLD) as f:
    for r in csv.DictReader(f):
        old_match[r["filename"]] = r["match"]

rows = []
counts = {0: 0, 1: 0, 2: 0, 3: 0}
fixed_n = new_n = 0
with open(CORR) as f:
    for r in csv.DictReader(f):
        fn = r["filename"]; d = int(r["edit_distance"]); k = d if d < 3 else 3
        counts[k] += 1
        fm = four.get(fn, {})
        om = old_match.get(fn)
        if om is None:
            status = "new"; new_n += 1          # newly-added consensus image (space-only agree)
        elif om == "0" and r["match"] == "1":
            status = "fixed"; fixed_n += 1        # was wrongly flagged, now correct
        else:
            status = "same"
        rows.append({
            "f": fn, "gt": r["ground_truth"], "p": r["new_ptq_pred"],
            "d": d, "k": k, "st": status,
            "fp32": fm.get("retrained_fp32", ""), "old": fm.get("ptq_int8_old_v2", ""),
            "qat": fm.get("qat_int8", ""),
        })

rows.sort(key=lambda x: (x["k"], x["d"], x["f"]))
total = len(rows)
def pct(n): return f"{100.0*n/total:.2f}%"

BASE_URL = "file://" + quote(SRC_IMGS) + "/"

html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TEMP — Corrected New PTQ vs Consensus (verify before re-run)</title>
<style>
:root{--bg:#0f1115;--card:#1a1d24;--line:#2a2e37;--txt:#e6e8ec;--mut:#9aa2af;
--exact:#2ecc71;--one:#f1c40f;--two:#e67e22;--wrong:#e74c3c;--accent:#4aa3ff;--fix:#a66bff;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--txt)}
header{padding:20px 26px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:19px}
.warn{background:#3a2a00;border:1px solid #7a5b00;color:#ffd479;padding:8px 12px;border-radius:8px;margin-top:8px;font-size:12.5px}
.sub{color:var(--mut);font-size:13px}
.cards{display:flex;flex-wrap:wrap;gap:12px;padding:18px 26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 16px;min-width:140px;flex:1}
.card .n{font-size:24px;font-weight:700}
.card .l{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.card.exact{border-left:4px solid var(--exact)}.card.one{border-left:4px solid var(--one)}
.card.two{border-left:4px solid var(--two)}.card.wrong{border-left:4px solid var(--wrong)}
.card.fix{border-left:4px solid var(--fix)}
.tabs{display:flex;gap:8px;padding:0 26px;flex-wrap:wrap}
.tab{background:var(--card);border:1px solid var(--line);color:var(--txt);padding:9px 16px;border-radius:9px 9px 0 0;cursor:pointer;font-size:13px}
.tab.active{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.tab.fixtab.active{background:var(--fix);border-color:var(--fix)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;padding:18px 26px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.tile img{width:100%;height:120px;object-fit:contain;background:#000;border-bottom:1px solid var(--line)}
.tile .body{padding:10px 12px;font-size:12.5px}
.badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;color:#111;margin-right:5px}
.b0{background:var(--exact)}.b1{background:var(--one)}.b2{background:var(--two)}.b3{background:var(--wrong);color:#fff}
.tag{display:inline-block;padding:2px 7px;border-radius:20px;font-size:10px;font-weight:700}
.tag.fixed{background:var(--fix);color:#fff}.tag.new{background:var(--accent);color:#fff}
.kv{display:flex;justify-content:space-between;gap:8px;padding:2px 0;border-top:1px dashed var(--line)}
.kv .k{color:var(--mut)}.kv .v{font-family:ui-monospace,Menlo,monospace;font-weight:600}
.gt .v{color:var(--exact)}.match .v{color:var(--exact)}.nomatch .v{color:var(--wrong)}
.link{display:flex;gap:6px;align-items:center;margin-top:8px;padding-top:8px;border-top:1px dashed var(--line)}
.link input{flex:1;min-width:0;background:#0b0d11;border:1px solid var(--line);color:var(--mut);border-radius:6px;padding:5px 7px;font:11px ui-monospace,Menlo,monospace}
.link a{color:var(--accent);text-decoration:none;font-size:11px;white-space:nowrap}
.copy{background:var(--accent);border:none;color:#fff;border-radius:6px;padding:5px 10px;font-size:11px;font-weight:600;cursor:pointer;white-space:nowrap}
.copy.ok{background:var(--exact)}
.more{display:block;margin:8px auto 34px;padding:11px 26px;background:var(--card);border:1px solid var(--line);color:var(--txt);border-radius:9px;cursor:pointer;font-size:14px}
.note{color:var(--mut);font-size:12px;padding:0 26px 6px}
</style></head><body>
<header>
<h1>TEMPORARY — Corrected New PTQ (v3) vs 3-Model Consensus</h1>
<div class="sub">Fixes: (1) GT keeps <b>leading zeros</b> the models actually read; (2) comparison ignores <b>spaces</b>.
GT = FP32 raw reference. Consensus = FP32 + Old PTQ + QAT agree (space-insensitive).</div>
<div class="warn">⚠ This is a VERIFY-FIRST dashboard from the corrected scoring of existing predictions.
Spot-check the <b>Fixed</b> tab, then tell me to launch the full foreground re-run.</div>
</header>
<div class="cards">
  <div class="card exact"><div class="n">__EXACT__</div><div class="l">Exact match</div><div class="sub">__EXACT_P__</div></div>
  <div class="card one"><div class="n">__ONE__</div><div class="l">1 char off</div><div class="sub">__ONE_P__ · cum __CUM1__</div></div>
  <div class="card two"><div class="n">__TWO__</div><div class="l">2 char off</div><div class="sub">__TWO_P__ · cum __CUM2__</div></div>
  <div class="card wrong"><div class="n">__WRONG__</div><div class="l">Wrong 3+</div><div class="sub">__WRONG_P__</div></div>
  <div class="card fix"><div class="n">__FIXED__</div><div class="l">Fixed vs old</div><div class="sub">+__NEW__ new consensus imgs</div></div>
  <div class="card"><div class="n">__TOTAL__</div><div class="l">Total imgs</div><div class="sub">corrected consensus</div></div>
</div>
<div class="tabs">
  <div class="tab active" data-mode="k0" onclick="pick('k0',this)">Exact · __EXACT__</div>
  <div class="tab" data-mode="k1" onclick="pick('k1',this)">1-char · __ONE__</div>
  <div class="tab" data-mode="k2" onclick="pick('k2',this)">2-char · __TWO__</div>
  <div class="tab" data-mode="k3" onclick="pick('k3',this)">Wrong 3+ · __WRONG__</div>
  <div class="tab fixtab" data-mode="fixed" onclick="pick('fixed',this)">★ Fixed · __FIXED__</div>
</div>
<div class="note">GT = corrected consensus. Purple “FIXED” = was wrongly flagged before, now correct. Blue “NEW” = image added to consensus by the space-insensitive rule. Copyable file:// link per tile.</div>
<div id="grid" class="grid"></div>
<button id="more" class="more" onclick="render()">Load more</button>
<script>
const DATA = __DATA__;
const BASE = "__BASE__";
let mode='k0', shown=0, PAGE=60, filtered=[];
function pick(m, el){
  mode=m; shown=0;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('grid').innerHTML='';
  if(m==='fixed') filtered=DATA.filter(r=>r.st==='fixed'||r.st==='new');
  else filtered=DATA.filter(r=>r.k===parseInt(m.slice(1)));
  render();
}
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function copyLink(btn){
  const url=btn.parentElement.querySelector('input').value;
  const done=()=>{const o=btn.textContent;btn.textContent='Copied';btn.classList.add('ok');setTimeout(()=>{btn.textContent=o;btn.classList.remove('ok');},1200);};
  if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(url).then(done).catch(()=>fb(btn,done));}else fb(btn,done);
}
function fb(btn,done){const i=btn.parentElement.querySelector('input');i.focus();i.select();try{document.execCommand('copy');done();}catch(e){btn.textContent='Ctrl+C';}}
function render(){
  const g=document.getElementById('grid');
  for(const r of filtered.slice(shown,shown+PAGE)){
    const label=r.d===0?'EXACT':(r.d+' off');
    const ptqCls=r.d===0?'match':'nomatch';
    const url=BASE+encodeURIComponent(r.f);
    let tag='';
    if(r.st==='fixed') tag='<span class="tag fixed">FIXED</span>';
    else if(r.st==='new') tag='<span class="tag new">NEW</span>';
    const div=document.createElement('div');div.className='tile';
    div.innerHTML=
      '<a href="'+url+'" target="_blank" rel="noopener"><img loading="lazy" src="../images/'+encodeURIComponent(r.f)+'" alt=""></a>'+
      '<div class="body">'+
        '<span class="badge b'+r.k+'">'+label+'</span>'+tag+
        '<div class="kv gt"><span class="k">GT (corrected)</span><span class="v">'+esc(r.gt)+'</span></div>'+
        '<div class="kv '+ptqCls+'"><span class="k">New PTQ v3</span><span class="v">'+esc(r.p)+'</span></div>'+
        '<div class="kv"><span class="k">FP32</span><span class="v">'+esc(r.fp32)+'</span></div>'+
        '<div class="kv"><span class="k">Old PTQ</span><span class="v">'+esc(r.old)+'</span></div>'+
        '<div class="kv"><span class="k">QAT</span><span class="v">'+esc(r.qat)+'</span></div>'+
        '<div class="link"><input readonly value="'+url+'" onclick="this.select()">'+
        '<button class="copy" onclick="copyLink(this)">Copy</button>'+
        '<a href="'+url+'" target="_blank" rel="noopener">Open ↗</a></div>'+
      '</div>';
    g.appendChild(div);
  }
  shown+=Math.min(PAGE,filtered.length-shown);
  document.getElementById('more').style.display= shown<filtered.length?'block':'none';
}
pick('k0', document.querySelector('.tab'));
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
        .replace("__CUM2__", pct(counts[0]+counts[1]+counts[2]))
        .replace("__FIXED__", str(fixed_n)).replace("__NEW__", str(new_n)))

with open(OUT, "w") as f:
    f.write(html)
print("Wrote:", OUT)
print(f"total={total} exact={counts[0]} fixed={fixed_n} new_consensus={new_n}")
