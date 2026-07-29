#!/usr/bin/env python3
"""
Build a single self-contained HTML dashboard comparing the THREE meter-OCR models
on a dataset, from the CSV that test_three_models.py produced.

    python3 scripts/build_comparison_dashboard.py \
        --csv    "datasets/comparison_dev/three_model_results.csv" \
        --images "datasets/comparison_dev/serialno_dev" \
        --out    "datasets/comparison_dev/comparison_dashboard.html"

Every crop is embedded as a base64 data-URI, so the .html is fully portable — open
it anywhere, no server, no image folder needed. There is NO ground truth for this
dataset, so the page reports model READINGS and cross-model AGREEMENT (a strong
proxy: when all three independent models agree, the read is almost certainly right),
plus a human REVIEW TRACKER (mark each crop correct/wrong per model, saved in the
browser) and a downloadable summary of everything.
"""
import os, csv, json, base64, argparse, mimetypes, html
import cv2

# Crops with a variance-of-Laplacian below this are flagged "blurry". 100 is the
# usual OpenCV rule-of-thumb for small text crops; tune with --blur-threshold.
BLUR_THRESHOLD = 100.0

MODELS = [
    ("fp32", "FP32", "finetune/inference_v2 · Paddle · 81 MB"),
    ("qat",  "QAT",  "deploy/qat_int8 · Paddle INT8 · 22 MB"),
    ("ptq",  "PTQ",  "deploy/ptq_int8 · ONNX INT8 · 21 MB"),
]


def data_uri(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as fh:
        return f"data:{mime};base64," + base64.b64encode(fh.read()).decode("ascii")


def blur_score(path):
    """Sharpness = variance of the Laplacian. Low = blurry. None if unreadable."""
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return round(float(cv2.Laplacian(im, cv2.CV_64F).var()), 1)


def load_rows(csv_path, img_dir, blur_threshold):
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            fn = r["filename"]
            p = os.path.join(img_dir, fn)
            if not os.path.exists(p):
                continue
            bs = blur_score(p)
            rows.append({
                "file": fn,
                "img": data_uri(p),
                "fp32": (r.get("fp32_result") or "").strip(),
                "qat":  (r.get("qat_result")  or "").strip(),
                "ptq":  (r.get("ptq_result")  or "").strip(),
                "blur": bs,
                "blurry": (bs is not None and bs < blur_threshold),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Meter-OCR — 3-Model Comparison")
    ap.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD,
                    help="variance-of-Laplacian below this = blurry (default 100)")
    args = ap.parse_args()

    rows = load_rows(args.csv, args.images, args.blur_threshold)
    payload = json.dumps(rows)
    n = len(rows)

    # agreement summary (server-side, for the header)
    agree3 = sum(1 for r in rows if r["fp32"] and r["fp32"] == r["qat"] == r["ptq"])
    blanks = {k: sum(1 for r in rows if not r[k]) for k, _, _ in MODELS}
    nblur = sum(1 for r in rows if r["blurry"])

    tpl = TEMPLATE
    tpl = tpl.replace("__TITLE__", html.escape(args.title))
    tpl = tpl.replace("__N__", str(n))
    tpl = tpl.replace("__AGREE3__", str(agree3))
    tpl = tpl.replace("__AGREE3PCT__", f"{(agree3/n*100 if n else 0):.1f}")
    tpl = tpl.replace("__BLANKS__", json.dumps(blanks))
    tpl = tpl.replace("__NBLUR__", str(nblur))
    tpl = tpl.replace("__BLURTH__", str(args.blur_threshold))
    tpl = tpl.replace("__DATA__", payload)
    tpl = tpl.replace("__STORE_KEY__", "meterocr_compare_review_v1")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(tpl)
    print(f"wrote {args.out}  ({n} rows, {os.path.getsize(args.out)/1e6:.1f} MB)")
    print(f"  all-3-agree: {agree3}/{n} ({agree3/n*100 if n else 0:.1f}%)   blanks: {blanks}")
    print(f"  blurry (var-Laplacian < {args.blur_threshold:g}): {nblur}/{n} ({nblur/n*100 if n else 0:.1f}%)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#0f1115; --panel:#171a21; --panel2:#1e222b; --line:#2a2f3a; --txt:#e7ebf0;
  --muted:#9aa4b2; --accent:#4f9cff; --ok:#2ec77a; --warn:#f5a623; --bad:#ff5d5d;
  --chip:#232834;
}
@media (prefers-color-scheme:light){:root{
  --bg:#f4f6fa; --panel:#fff; --panel2:#f0f3f8; --line:#dde3ec; --txt:#131722;
  --muted:#5c6b7f; --chip:#eef2f8;
}}
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--txt)}
header{padding:18px 22px;border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:20}
h1{margin:0 0 4px;font-size:19px}
.sub{color:var(--muted);font-size:12.5px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
.stat{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:8px 14px;min-width:96px}
.stat b{display:block;font-size:20px;line-height:1.1}
.stat span{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.tabs{display:flex;gap:4px;padding:10px 22px 0;background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:15;flex-wrap:wrap}
.tab{padding:9px 16px;border:1px solid transparent;border-bottom:none;border-radius:9px 9px 0 0;cursor:pointer;color:var(--muted);font-weight:600}
.tab.active{background:var(--bg);color:var(--txt);border-color:var(--line)}
.tab .cnt{font-size:11px;color:var(--muted);font-weight:500}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px 22px;background:var(--panel);border-bottom:1px solid var(--line);position:sticky;z-index:14}
input[type=search],select{background:var(--panel2);border:1px solid var(--line);color:var(--txt);border-radius:8px;padding:7px 10px;font-size:13px}
input[type=search]{min-width:220px}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:8px 13px;font-weight:600;cursor:pointer;font-size:13px}
button.ghost{background:var(--chip);color:var(--txt);border:1px solid var(--line)}
.wrap{padding:18px 22px 60px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
.card img{width:100%;background:#000;object-fit:contain;max-height:120px;image-rendering:auto}
.card .body{padding:9px 11px}
.read{font:15px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600;word-break:break-all}
.read.blank{color:var(--muted);font-style:italic;font-weight:400}
.fname{color:var(--muted);font-size:10.5px;margin-top:5px;word-break:break-all}
.rowreads{display:grid;grid-template-columns:auto 1fr;gap:2px 8px;font:12.5px/1.35 ui-monospace,monospace;margin-top:4px}
.rowreads .k{color:var(--muted)}
.badge{display:inline-block;font-size:10px;font-weight:700;padding:2px 7px;border-radius:20px;margin-top:6px}
.b-ok{background:rgba(46,199,122,.16);color:var(--ok)}
.b-warn{background:rgba(245,166,35,.16);color:var(--warn)}
.b-bad{background:rgba(255,93,93,.16);color:var(--bad)}
.b-blur{background:rgba(159,122,234,.18);color:#a78bfa}
.card.is-blurry{outline:1px dashed #a78bfa}
.rev{display:flex;gap:6px;margin-top:8px}
.rev button{flex:1;padding:5px;font-size:12px}
.rev .yes{background:var(--chip);color:var(--ok);border:1px solid var(--line)}
.rev .no{background:var(--chip);color:var(--bad);border:1px solid var(--line)}
.rev .blur{background:var(--chip);color:#a78bfa;border:1px solid var(--line)}
.rev .yes.on{background:var(--ok);color:#04210f}
.rev .no.on{background:var(--bad);color:#2a0606}
.rev .blur.on{background:#a78bfa;color:#1e1233}
.card.marked-yes{outline:2px solid var(--ok)}
.card.marked-no{outline:2px solid var(--bad)}
.card.marked-blur{outline:2px solid #a78bfa}
.diff{background:rgba(245,166,35,.12)}
.track{padding:14px 22px}
.bar{height:12px;border-radius:8px;background:var(--panel2);overflow:hidden;border:1px solid var(--line);display:flex}
.bar i{display:block;height:100%}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;color:var(--muted);font-size:12px}
.legend b{color:var(--txt)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:middle}
table.summary{border-collapse:collapse;width:100%;margin-top:8px}
table.summary th,table.summary td{border:1px solid var(--line);padding:8px 10px;text-align:left;font-size:13px}
table.summary th{background:var(--panel2)}
.hide{display:none}
.hint{color:var(--muted);font-size:12px;margin:4px 0 0}
</style></head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">FP32 (Paddle 81 MB) · QAT (Paddle INT8 22 MB) · PTQ (ONNX INT8 21 MB) — no ground truth; agreement + human review.</div>
  <div class="stats">
    <div class="stat"><b>__N__</b><span>images</span></div>
    <div class="stat"><b>__AGREE3__</b><span>all-3 agree</span></div>
    <div class="stat"><b>__AGREE3PCT__%</b><span>agree rate</span></div>
    <div class="stat"><b>__NBLUR__</b><span>blurry</span></div>
    <div class="stat"><b id="s-reviewed">0</b><span>reviewed</span></div>
    <div class="stat"><b id="s-progress">0%</b><span>tracker</span></div>
  </div>
</header>

<div class="tabs" id="tabs"></div>
<div class="toolbar">
  <input type="search" id="q" placeholder="filter by reading or filename…">
  <label><input type="checkbox" id="only-diff"> only disagreements</label>
  <label><input type="checkbox" id="only-blur"> only blurry</label>
  <label><input type="checkbox" id="only-unreviewed"> only un-reviewed</label>
  <span style="flex:1"></span>
  <button class="ghost" id="btn-json">Download JSON</button>
  <button id="btn-csv">Download summary CSV</button>
  <button class="ghost" id="btn-reset">Reset review</button>
</div>

<div id="view" class="wrap"></div>

<script>
const DATA = __DATA__;
const BLANKS = __BLANKS__;
const BLUR_TH = __BLURTH__;
const KEY = "__STORE_KEY__";
const MODELS = [["fp32","FP32"],["qat","QAT"],["ptq","PTQ"]];
let review = JSON.parse(localStorage.getItem(KEY) || "{}");   // {file:{model:'yes'|'no'}}
let tab = "compare";

function agreeClass(r){
  const v=[r.fp32,r.qat,r.ptq];
  const nonblank=v.filter(x=>x);
  if(nonblank.length<3) return "partial";
  return (v[0]===v[1]&&v[1]===v[2])?"all":"split";
}
function esc(s){return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function readCell(t){return t? esc(t) : '<span class="blank">— blank —</span>';}
function blurBadge(r){
  if(r.blur===null) return "";
  return r.blurry
    ? `<span class="badge b-blur" title="variance of Laplacian ${r.blur} < ${BLUR_TH}">blurry · ${r.blur}</span>`
    : "";
}

function setReview(file,model,val){
  review[file]=review[file]||{};
  if(review[file][model]===val) delete review[file][model];
  else review[file][model]=val;
  if(Object.keys(review[file]).length===0) delete review[file];
  localStorage.setItem(KEY, JSON.stringify(review));
  updateTracker(); render();
}

function tabsHtml(){
  const tabsDef=[["compare","Compare all"],...MODELS,["tracker","Tracker"]];
  return tabsDef.map(([id,label])=>{
    let cnt="";
    if(id!=="compare"&&id!=="tracker") cnt=`<span class="cnt"> · ${BLANKS[id]} blank</span>`;
    if(id==="compare") cnt=`<span class="cnt"> · ${DATA.length}</span>`;
    return `<div class="tab ${id===tab?'active':''}" data-tab="${id}">${label}${cnt}</div>`;
  }).join("");
}

function filtered(){
  const q=document.getElementById("q").value.trim().toLowerCase();
  const onlyDiff=document.getElementById("only-diff").checked;
  const onlyBlur=document.getElementById("only-blur").checked;
  const onlyUn=document.getElementById("only-unreviewed").checked;
  return DATA.filter(r=>{
    if(q && !(`${r.file} ${r.fp32} ${r.qat} ${r.ptq}`.toLowerCase().includes(q))) return false;
    if(onlyDiff && agreeClass(r)==="all") return false;
    if(onlyBlur && !r.blurry) return false;
    if(onlyUn && review[r.file] && (tab==="compare" ? Object.keys(review[r.file]).length>=1 : review[r.file][tab])) return false;
    return true;
  });
}

function cardCompare(r){
  const a=agreeClass(r);
  const badge = a==="all"?'<span class="badge b-ok">all 3 agree</span>'
              : a==="split"?'<span class="badge b-bad">disagree</span>'
              : '<span class="badge b-warn">partial / blank</span>';
  const rev=review[r.file]||{};
  const rr=MODELS.map(([k,lbl])=>{
    const cls = (a==="split" && r[k]!==r.fp32) ? 'diff' : '';
    return `<div class="k">${lbl}</div><div class="${cls}">${readCell(r[k])}</div>`;
  }).join("");
  return `<div class="card ${r.blurry?'is-blurry':''}">
     <img loading="lazy" src="${r.img}">
     <div class="body">
       <div class="rowreads">${rr}</div>
       ${badge} ${blurBadge(r)}
       <div class="fname">${esc(r.file)}</div>
     </div></div>`;
}

function cardModel(r,model){
  const val=(review[r.file]||{})[model];
  const markCls = val==='yes'?'marked-yes':val==='no'?'marked-no':val==='blur'?'marked-blur':'';
  return `<div class="card ${markCls} ${r.blurry?'is-blurry':''}">
     <img loading="lazy" src="${r.img}">
     <div class="body">
       <div class="read ${r[model]?'':'blank'}">${r[model]?esc(r[model]):'— blank —'}</div>
       ${blurBadge(r)}
       <div class="fname">${esc(r.file)}</div>
       <div class="rev">
         <button class="yes ${val==='yes'?'on':''}" data-f="${esc(r.file)}" data-m="${model}" data-v="yes">✓ correct</button>
         <button class="no ${val==='no'?'on':''}" data-f="${esc(r.file)}" data-m="${model}" data-v="no">✗ wrong</button>
         <button class="blur ${val==='blur'?'on':''}" data-f="${esc(r.file)}" data-m="${model}" data-v="blur">◐ blur</button>
       </div>
     </div></div>`;
}

function render(){
  document.getElementById("tabs").innerHTML=tabsHtml();
  const view=document.getElementById("view");
  const rows=filtered();
  if(tab==="tracker"){ view.innerHTML=trackerHtml(); return; }
  const mk = tab==="compare" ? cardCompare : (r)=>cardModel(r,tab);
  view.innerHTML = `<p class="hint">${rows.length} shown${tab!=='compare'?' — mark each ✓ correct / ✗ wrong; saved in your browser':''}.</p>
    <div class="grid">${rows.map(mk).join("")}</div>`;
}

function trackerHtml(){
  const total=DATA.length;
  let counts={};
  MODELS.forEach(([k])=>counts[k]={yes:0,no:0,blur:0});
  DATA.forEach(r=>{const rv=review[r.file]||{};MODELS.forEach(([k])=>{if(rv[k])counts[k][rv[k]]++;});});
  const perModel=MODELS.map(([k,lbl])=>{
    const y=counts[k].yes,n=counts[k].no,b=counts[k].blur,done=y+n+b;
    // accuracy = correct / (correct+wrong); blur excluded as un-scoreable
    const acc=(y+n)?(y/(y+n)*100).toFixed(1):"–";
    return `<tr><td><b>${lbl}</b></td><td>${y}</td><td>${n}</td><td>${b}</td><td>${done} / ${total}</td>
       <td>${acc==="–"?"–":acc+"%"}</td></tr>`;
  }).join("");
  // agreement breakdown
  let all=0,split=0,part=0;
  DATA.forEach(r=>{const a=agreeClass(r);a==="all"?all++:a==="split"?split++:part++;});
  const w=x=>(x/total*100).toFixed(1);
  // blur breakdown, and how blur correlates with disagreement
  const blurry=DATA.filter(r=>r.blurry);
  const blurSplit=blurry.filter(r=>agreeClass(r)!=="all").length;
  const sharpSplit=DATA.filter(r=>!r.blurry&&agreeClass(r)!=="all").length;
  return `<div class="track">
    <h3 style="margin:6px 0">Cross-model agreement (no ground truth — agreement ≈ confidence)</h3>
    <div class="bar">
      <i style="width:${w(all)}%;background:var(--ok)"></i>
      <i style="width:${w(split)}%;background:var(--bad)"></i>
      <i style="width:${w(part)}%;background:var(--warn)"></i>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:var(--ok)"></span><b>${all}</b> all-3 identical (${w(all)}%)</span>
      <span><span class="dot" style="background:var(--bad)"></span><b>${split}</b> models disagree (${w(split)}%)</span>
      <span><span class="dot" style="background:var(--warn)"></span><b>${part}</b> partial / has blank (${w(part)}%)</span>
    </div>
    <h3 style="margin:22px 0 6px">Image quality (blur = variance of Laplacian &lt; ${BLUR_TH})</h3>
    <table class="summary">
      <tr><th>Group</th><th>images</th><th>models disagree</th><th>disagree rate</th></tr>
      <tr><td><b>Blurry</b></td><td>${blurry.length}</td><td>${blurSplit}</td>
        <td>${blurry.length?(blurSplit/blurry.length*100).toFixed(1)+"%":"–"}</td></tr>
      <tr><td><b>Sharp</b></td><td>${total-blurry.length}</td><td>${sharpSplit}</td>
        <td>${(total-blurry.length)?(sharpSplit/(total-blurry.length)*100).toFixed(1)+"%":"–"}</td></tr>
    </table>
    <p class="hint">Blurry crops drive most disagreements — filter to them with “only blurry” to triage the worst-quality inputs.</p>
    <h3 style="margin:22px 0 6px">Human review progress (per model)</h3>
    <table class="summary">
      <tr><th>Model</th><th>✓ correct</th><th>✗ wrong</th><th>◐ blur</th><th>reviewed</th><th>observed accuracy</th></tr>
      ${perModel}
    </table>
    <p class="hint">Observed accuracy = ✓ / (✓ + ✗); crops you marked ◐ blur are excluded as un-scoreable. Use the model tabs to review; Download buttons export everything.</p>
  </div>`;
}

function updateTracker(){
  const reviewedFiles=Object.keys(review).length;
  document.getElementById("s-reviewed").textContent=reviewedFiles;
  document.getElementById("s-progress").textContent=(reviewedFiles/DATA.length*100).toFixed(0)+"%";
}

// ---- downloads ----
function downloadBlob(name,txt,type){
  const b=new Blob([txt],{type});const u=URL.createObjectURL(b);
  const a=document.createElement("a");a.href=u;a.download=name;a.click();URL.revokeObjectURL(u);
}
function summaryRows(){
  return DATA.map(r=>{
    const a=agreeClass(r);const rv=review[r.file]||{};
    return {filename:r.file,fp32:r.fp32,qat:r.qat,ptq:r.ptq,agreement:a,
            blur_score:(r.blur===null?"":r.blur),blurry:(r.blurry?"yes":"no"),
            fp32_review:rv.fp32||"",qat_review:rv.qat||"",ptq_review:rv.ptq||""};
  });
}
document.getElementById("btn-json").onclick=()=>downloadBlob(
  "meterocr_comparison_summary.json",JSON.stringify(summaryRows(),null,2),"application/json");
document.getElementById("btn-csv").onclick=()=>{
  const rows=summaryRows();const cols=Object.keys(rows[0]);
  const csv=[cols.join(",")].concat(rows.map(r=>cols.map(c=>{
    const v=(""+r[c]).replace(/"/g,'""');return /[",\n]/.test(v)?`"${v}"`:v;}).join(","))).join("\n");
  downloadBlob("meterocr_comparison_summary.csv",csv,"text/csv");
};
document.getElementById("btn-reset").onclick=()=>{
  if(confirm("Clear all review marks?")){review={};localStorage.removeItem(KEY);updateTracker();render();}
};

// ---- events ----
document.addEventListener("click",e=>{
  const t=e.target.closest(".tab");
  if(t){tab=t.dataset.tab;render();return;}
  const rb=e.target.closest(".rev button");
  if(rb){setReview(rb.dataset.f,rb.dataset.m,rb.dataset.v);return;}
});
["q","only-diff","only-blur","only-unreviewed"].forEach(id=>
  document.getElementById(id).addEventListener("input",render));

updateTracker();render();
</script>
</body></html>"""


if __name__ == "__main__":
    main()
