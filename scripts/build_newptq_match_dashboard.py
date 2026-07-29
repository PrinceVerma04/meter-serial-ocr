#!/usr/bin/env python3
"""Build a local HTML dashboard categorizing the NEW PTQ (v3) reads against the
3-model consensus ground truth (FP32 + old PTQ + QAT all agreed).

Categories by Levenshtein edit distance between new-PTQ prediction and the
consensus serial: exact (0), 1-char, 2-char, wrong (3+).

Shows the real meter image + all four model outputs per row.
Output: 26788 images for test/agreed_fp32_oldptq_qat/newptq_match_dashboard.html
"""
import csv, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "26788 images for test")
CDIR = os.path.join(BASE, "agreed_fp32_oldptq_qat")
SCORED = os.path.join(CDIR, "new_ptq_vs_groundtruth_corrected.csv")  # corrected scoring
FOURMODEL = os.path.join(BASE, "four_model_results.csv")
OUT = os.path.join(CDIR, "newptq_match_dashboard.html")


def lev(a, b):
    a, b = str(a), str(b)
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[n]


# per-model outputs keyed by filename
four = {}
with open(FOURMODEL) as f:
    for r in csv.DictReader(f):
        four[r["filename"]] = r

rows = []
counts = {0: 0, 1: 0, 2: 0, 3: 0}
with open(SCORED) as f:
    for r in csv.DictReader(f):
        gt = r["ground_truth"]
        pred = r["new_ptq_pred"]
        # corrected CSV already carries the space-insensitive edit distance
        d = int(r["edit_distance"]) if r.get("edit_distance") not in (None, "") else lev(gt, pred)
        k = d if d < 3 else 3
        counts[k] += 1
        fm = four.get(r["filename"], {})
        rows.append({
            "f": r["filename"],
            "gt": gt,
            "p": pred,
            "d": d,
            "k": k,
            "fp32": fm.get("retrained_fp32", ""),
            "old": fm.get("ptq_int8_old_v2", ""),
            "qat": fm.get("qat_int8", ""),
        })

# order: exact, 1-char, 2-char, wrong — and within a bucket, shortest edits first
rows.sort(key=lambda x: (x["k"], x["d"], x["f"]))
total = len(rows)


def pct(n): return f"{100.0 * n / total:.2f}%"


summary = {
    "total": total,
    "exact": counts[0], "one": counts[1], "two": counts[2], "wrong": counts[3],
    "exact_p": pct(counts[0]), "one_p": pct(counts[1]),
    "two_p": pct(counts[2]), "wrong_p": pct(counts[3]),
    "cum1_p": pct(counts[0] + counts[1]),
    "cum2_p": pct(counts[0] + counts[1] + counts[2]),
}

html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>New PTQ vs 3-Model Consensus — Match Dashboard</title>
<style>
:root{--bg:#0f1115;--card:#1a1d24;--line:#2a2e37;--txt:#e6e8ec;--mut:#9aa2af;
--exact:#2ecc71;--one:#f1c40f;--two:#e67e22;--wrong:#e74c3c;--accent:#4aa3ff;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--txt)}
header{padding:22px 26px;border-bottom:1px solid var(--line)}
h1{margin:0 0 4px;font-size:20px}
.sub{color:var(--mut);font-size:13px}
.cards{display:flex;flex-wrap:wrap;gap:12px;padding:20px 26px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px;min-width:150px;flex:1}
.card .n{font-size:26px;font-weight:700}
.card .l{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.card.exact{border-left:4px solid var(--exact)}
.card.one{border-left:4px solid var(--one)}
.card.two{border-left:4px solid var(--two)}
.card.wrong{border-left:4px solid var(--wrong)}
.tabs{display:flex;gap:8px;padding:0 26px;flex-wrap:wrap}
.tab{background:var(--card);border:1px solid var(--line);color:var(--txt);padding:9px 16px;border-radius:9px 9px 0 0;cursor:pointer;font-size:13px}
.tab.active{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;padding:18px 26px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.tile img{width:100%;height:120px;object-fit:contain;background:#000;border-bottom:1px solid var(--line)}
.tile .body{padding:10px 12px;font-size:12.5px}
.badge{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;color:#111}
.b0{background:var(--exact)}.b1{background:var(--one)}.b2{background:var(--two)}.b3{background:var(--wrong);color:#fff}
.kv{display:flex;justify-content:space-between;gap:8px;padding:2px 0;border-top:1px dashed var(--line)}
.kv .k{color:var(--mut)}
.kv .v{font-family:ui-monospace,Menlo,monospace;font-weight:600}
.gt .v{color:var(--exact)}
.match .v{color:var(--exact)} .nomatch .v{color:var(--wrong)}
.more{display:block;margin:8px auto 34px;padding:11px 26px;background:var(--card);border:1px solid var(--line);color:var(--txt);border-radius:9px;cursor:pointer;font-size:14px}
.note{color:var(--mut);font-size:12px;padding:0 26px 6px}
.link{display:flex;gap:6px;align-items:center;margin-top:8px;padding-top:8px;border-top:1px dashed var(--line)}
.link input{flex:1;min-width:0;background:#0b0d11;border:1px solid var(--line);color:var(--mut);border-radius:6px;padding:5px 7px;font:11px ui-monospace,Menlo,monospace}
.link a{color:var(--accent);text-decoration:none;font-size:11px;white-space:nowrap}
.copy{background:var(--accent);border:none;color:#fff;border-radius:6px;padding:5px 10px;font-size:11px;font-weight:600;cursor:pointer;white-space:nowrap}
.copy.ok{background:var(--exact)}
</style></head><body>
<header>
<h1>New PTQ (v3) vs 3-Model Consensus Ground Truth</h1>
<div class="sub">Ground truth = images where <b>Retrained&nbsp;FP32 + Old&nbsp;PTQ + QAT</b> agree (space-insensitive, leading zeros kept).
The <b>New PTQ INT8 (v3)</b> deploy model was excluded from the vote, then scored against it.
Categories = character edit distance (space-insensitive). Corrected consensus set of 18,544 images.</div>
</header>
<div class="cards">
  <div class="card exact"><div class="n">__EXACT__</div><div class="l">Exact match</div><div class="sub">__EXACT_P__</div></div>
  <div class="card one"><div class="n">__ONE__</div><div class="l">1 character off</div><div class="sub">__ONE_P__ · cumulative __CUM1__</div></div>
  <div class="card two"><div class="n">__TWO__</div><div class="l">2 characters off</div><div class="sub">__TWO_P__ · cumulative __CUM2__</div></div>
  <div class="card wrong"><div class="n">__WRONG__</div><div class="l">Wrong (3+ chars)</div><div class="sub">__WRONG_P__</div></div>
  <div class="card"><div class="n">__TOTAL__</div><div class="l">Total images</div><div class="sub">3-model consensus set</div></div>
</div>
<div class="tabs">
  <div class="tab active" data-k="0" onclick="pick(0,this)">Exact · __EXACT__</div>
  <div class="tab" data-k="1" onclick="pick(1,this)">1-char · __ONE__</div>
  <div class="tab" data-k="2" onclick="pick(2,this)">2-char · __TWO__</div>
  <div class="tab" data-k="3" onclick="pick(3,this)">Wrong 3+ · __WRONG__</div>
</div>
<div class="note">GT = 3-model consensus (ground truth). Green PTQ row = matches consensus. Each tile has a copyable direct <b>file://</b> link to the image. Images load lazily; use “Load more”.</div>
<div id="grid" class="grid"></div>
<button id="more" class="more" onclick="render()">Load more</button>
<script>
const DATA = __DATA__;
const BASE = "__BASE__";   // absolute file:// URL of the images/ folder
let curK = 0, shown = 0, PAGE = 60, filtered = [];
function pick(k, el){
  curK = k; shown = 0;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('grid').innerHTML='';
  filtered = DATA.filter(r=>r.k===k);
  render();
}
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function copyLink(btn){
  const url = btn.parentElement.querySelector('input').value;
  const done = ()=>{ const old=btn.textContent; btn.textContent='Copied'; btn.classList.add('ok');
    setTimeout(()=>{btn.textContent=old; btn.classList.remove('ok');},1200); };
  if(navigator.clipboard && navigator.clipboard.writeText){
    navigator.clipboard.writeText(url).then(done).catch(()=>fallback(btn,done));
  } else { fallback(btn,done); }
}
function fallback(btn,done){
  const inp = btn.parentElement.querySelector('input');
  inp.focus(); inp.select();
  try{ document.execCommand('copy'); done(); }catch(e){ btn.textContent='Ctrl+C'; }
}
function render(){
  const g = document.getElementById('grid');
  const slice = filtered.slice(shown, shown+PAGE);
  for(const r of slice){
    const bcls = 'b'+r.k;
    const label = r.d===0?'EXACT':(r.d+' off');
    const ptqCls = r.d===0?'match':'nomatch';
    const url = BASE + encodeURIComponent(r.f);
    const div = document.createElement('div'); div.className='tile';
    div.innerHTML =
      '<a href="'+url+'" target="_blank" rel="noopener"><img loading="lazy" src="images/'+encodeURIComponent(r.f)+'" alt=""></a>'+
      '<div class="body">'+
        '<span class="badge '+bcls+'">'+label+'</span>'+
        '<div class="kv gt"><span class="k">GT (consensus)</span><span class="v">'+esc(r.gt)+'</span></div>'+
        '<div class="kv '+ptqCls+'"><span class="k">New PTQ v3</span><span class="v">'+esc(r.p)+'</span></div>'+
        '<div class="kv"><span class="k">FP32</span><span class="v">'+esc(r.fp32)+'</span></div>'+
        '<div class="kv"><span class="k">Old PTQ</span><span class="v">'+esc(r.old)+'</span></div>'+
        '<div class="kv"><span class="k">QAT</span><span class="v">'+esc(r.qat)+'</span></div>'+
        '<div class="link">'+
          '<input readonly value="'+url+'" onclick="this.select()">'+
          '<button class="copy" onclick="copyLink(this)">Copy</button>'+
          '<a href="'+url+'" target="_blank" rel="noopener">Open ↗</a>'+
        '</div>'+
      '</div>';
    g.appendChild(div);
  }
  shown += slice.length;
  document.getElementById('more').style.display = shown<filtered.length ? 'block':'none';
}
pick(0, document.querySelector('.tab'));
</script>
</body></html>"""

# absolute file:// URL of the images/ folder (percent-encode path segments, keep slashes)
from urllib.parse import quote
IMG_DIR = os.path.join(CDIR, "images")
BASE_URL = "file://" + quote(IMG_DIR) + "/"

html = (html
        .replace("__BASE__", BASE_URL)
        .replace("__DATA__", json.dumps(rows, separators=(",", ":")))
        .replace("__TOTAL__", str(summary["total"]))
        .replace("__EXACT_P__", summary["exact_p"]).replace("__EXACT__", str(summary["exact"]))
        .replace("__ONE_P__", summary["one_p"]).replace("__ONE__", str(summary["one"]))
        .replace("__TWO_P__", summary["two_p"]).replace("__TWO__", str(summary["two"]))
        .replace("__WRONG_P__", summary["wrong_p"]).replace("__WRONG__", str(summary["wrong"]))
        .replace("__CUM1__", summary["cum1_p"]).replace("__CUM2__", summary["cum2_p"]))

with open(OUT, "w") as f:
    f.write(html)

print("Wrote:", OUT)
print("Summary:", json.dumps(summary, indent=2))
