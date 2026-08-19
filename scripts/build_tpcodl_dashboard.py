#!/usr/bin/env python3
"""
Build an HTML dashboard comparing fp32 / qat / ptq extracted results against the
bot ground truth, with the actual meter image for every row.

- images referenced by RELATIVE path (full image, never cropped)
- image name shown in a copyable field with a Copy button
- 3 images per row
- summary table + per-model accuracy on top
- filter: all / any wrong / all correct + search box

Output: Some_analysis_part/tpcodl_dashboard.html
Open it from inside Some_analysis_part/ so the relative image paths resolve.
"""
import os
import json
import html
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, "Some_analysis_part")

GT_CSV = os.path.join(ANALYSIS, "1staug_4thaug_2026_serialno_tpcodl.csv")
PRED_CSV = os.path.join(ANALYSIS, "three_model_consensus_tpcodl.csv")
BLUR_REPORT = os.path.join(ANALYSIS, "any1_incorrect_blur_report.csv")
IMG_DIR = "1staug_4thaug_2026_serialno_tpcodl"  # relative to the html file
OUT = os.path.join(ANALYSIS, "tpcodl_dashboard.html")

MODELS = ["fp32", "qat", "ptq"]


def blurry_set():
    """Names flagged blurry (removed) by scripts/blur_filter_any1_incorrect.py."""
    if not os.path.exists(BLUR_REPORT):
        return set()
    b = pd.read_csv(BLUR_REPORT)
    if "removed_blurry" not in b.columns:
        return set()
    return set(b[b["removed_blurry"] == True]["image_name"].astype(str))


def load():
    gt = pd.read_csv(GT_CSV)
    gt["image_name"] = gt["readings.KWH.bigImg"].astype(str).str.split("/").str[-1]
    gt = gt.rename(columns={"readings.KWH.bot_annotation.serial_value": "actual_value"})
    gt["actual_value"] = gt["actual_value"].fillna("").astype(str).str.strip()

    pred = pd.read_csv(PRED_CSV)
    for m in MODELS:
        pred[m] = pred[m].fillna("").astype(str).str.strip()

    m = pd.merge(gt[["image_name", "actual_value"]],
                 pred[["image_name"] + MODELS], on="image_name", how="inner")
    return m


def main():
    df = load()
    total = len(df)

    stats = {}
    for mdl in MODELS:
        correct = int((df[mdl] == df["actual_value"]).sum())
        stats[mdl] = {"correct": correct, "wrong": total - correct,
                      "acc": round(100 * correct / total, 2) if total else 0}

    blurry = blurry_set()

    # cross-model counts
    wrong_mask = {m: (df[m] != df["actual_value"]) for m in MODELS}
    all3_wrong = int((wrong_mask["fp32"] & wrong_mask["qat"] & wrong_mask["ptq"]).sum())
    any_wrong_mask = wrong_mask["fp32"] | wrong_mask["qat"] | wrong_mask["ptq"]
    any_wrong = int(any_wrong_mask.sum())
    is_blurry = df["image_name"].isin(blurry)
    any_wrong_sharp = int((any_wrong_mask & ~is_blurry).sum())
    blurry_count = int(is_blurry.sum())

    rows = []
    for _, r in df.iterrows():
        gt = r["actual_value"]
        rows.append({
            "img": f"{IMG_DIR}/{r['image_name']}",
            "name": r["image_name"],
            "gt": gt,
            "fp32": r["fp32"], "qat": r["qat"], "ptq": r["ptq"],
            "fp32_ok": r["fp32"] == gt,
            "qat_ok": r["qat"] == gt,
            "ptq_ok": r["ptq"] == gt,
            "blurry": bool(r["image_name"] in blurry),
        })
    data_json = json.dumps(rows)

    # summary table html
    trs = ""
    for mdl in MODELS:
        s = stats[mdl]
        trs += (f"<tr><td class='m'>{mdl}</td><td class='ok'>{s['correct']:,}</td>"
                f"<td class='bad'>{s['wrong']:,}</td><td>{s['acc']:.2f}%</td></tr>")

    page = HTML_TEMPLATE.replace("__TOTAL__", f"{total:,}") \
                        .replace("__ROWS__", trs) \
                        .replace("__ALL3WRONG__", f"{all3_wrong:,}") \
                        .replace("__ANYWRONG__", f"{any_wrong_sharp:,}") \
                        .replace("__BLURRY__", f"{blurry_count:,}") \
                        .replace("__DATA__", data_json)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Wrote {OUT}")
    print(f"{total:,} images. " +
          "  ".join(f"{m}={stats[m]['acc']:.2f}%" for m in MODELS))


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TPCODL — 3-model vs Bot Ground Truth</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         background: #0f1115; color: #e6e8eb; }
  header { padding: 20px 24px; border-bottom: 1px solid #2a2f3a; position: sticky; top: 0;
           background: #0f1115; z-index: 10; }
  h1 { margin: 0 0 12px; font-size: 20px; }
  table.summary { border-collapse: collapse; font-size: 14px; }
  table.summary td, table.summary th { border: 1px solid #2a2f3a; padding: 6px 14px; text-align: right; }
  table.summary th { background: #171a21; }
  table.summary td.m { text-align: left; font-weight: 600; text-transform: uppercase; }
  .ok { color: #4ade80; } .bad { color: #f87171; }
  .controls { margin-top: 14px; display: flex; flex-direction: column; gap: 10px; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .glabel { font-size: 12px; color: #8b94a3; min-width: 92px; text-transform: uppercase;
            letter-spacing: .04em; }
  .controls button { background: #1c2129; color: #e6e8eb; border: 1px solid #2a2f3a;
                     padding: 7px 13px; border-radius: 6px; cursor: pointer; font-size: 13px; }
  .controls button:hover { border-color: #3b82f6; }
  .controls button.active { background: #2563eb; border-color: #2563eb; color: #fff; }
  .controls select, .controls input { background: #1c2129; color: #e6e8eb; border: 1px solid #2a2f3a;
                    padding: 7px 10px; border-radius: 6px; font-size: 13px; }
  .controls input { min-width: 260px; }
  #download { background: #16351f; border-color: #1f7a3d; color: #7ee2a1; }
  .count { font-size: 13px; color: #9aa4b2; }
  main { padding: 20px 24px; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  .card { background: #171a21; border: 1px solid #2a2f3a; border-radius: 10px; padding: 12px;
          display: flex; flex-direction: column; gap: 10px; }
  .imgwrap { background: #0b0d11; border-radius: 8px; padding: 8px; text-align: center; }
  .imgwrap img { max-width: 100%; height: auto; image-rendering: -webkit-optimize-contrast; cursor: zoom-in; }
  /* lightbox */
  #lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.9); display: none;
              align-items: center; justify-content: center; z-index: 100; cursor: zoom-out;
              flex-direction: column; gap: 14px; padding: 24px; }
  #lightbox.open { display: flex; }
  #lightbox img { max-width: 96vw; max-height: 82vh; image-rendering: -webkit-optimize-contrast;
                  transition: transform .15s ease; }
  #lbname { color: #cbd5e1; font-family: ui-monospace, monospace; font-size: 13px; }
  #lbclose { position: fixed; top: 16px; right: 22px; color: #fff; font-size: 30px;
             cursor: pointer; line-height: 1; }
  #lbhint { color: #6b7280; font-size: 12px; }
  .namerow { display: flex; gap: 6px; align-items: center; }
  .namerow input { flex: 1; background: #0b0d11; color: #cbd5e1; border: 1px solid #2a2f3a;
                   border-radius: 6px; padding: 5px 8px; font-size: 11px; font-family: ui-monospace, monospace; }
  .namerow button { background: #1c2129; color: #e6e8eb; border: 1px solid #2a2f3a;
                    padding: 5px 10px; border-radius: 6px; cursor: pointer; font-size: 12px; }
  .vals { font-size: 13px; }
  .vals .line { display: flex; justify-content: space-between; gap: 8px; padding: 3px 0;
                border-bottom: 1px solid #232833; }
  .vals .line:last-child { border-bottom: 0; }
  .vals .lab { color: #9aa4b2; }
  .vals .val { font-family: ui-monospace, monospace; text-align: right; word-break: break-all; }
  .gt .val { color: #93c5fd; font-weight: 600; }
  .badge { display: inline-block; min-width: 14px; margin-left: 6px; }
  /* manual review buttons */
  .annot { display: flex; gap: 8px; margin-top: 2px; }
  .annot button { flex: 1; padding: 7px 0; border-radius: 6px; cursor: pointer; font-size: 13px;
                  background: #1c2129; color: #e6e8eb; border: 1px solid #2a2f3a; }
  .annot button.c.sel { background: #166534; border-color: #22c55e; color: #fff; }
  .annot button.w.sel { background: #7f1d1d; border-color: #ef4444; color: #fff; }
  .annot button:hover { border-color: #3b82f6; }
</style>
</head>
<body>
<header>
  <h1>TPCODL — 3-model extracted results vs Bot Ground Truth (__TOTAL__ images)</h1>
  <table class="summary">
    <tr><th>Model</th><th>Correct</th><th>Incorrect</th><th>Accuracy</th></tr>
    __ROWS__
  </table>
  <div class="controls">
    <div class="row">
      <span class="glabel">Per model</span>
      <select id="model">
        <option value="fp32">fp32</option>
        <option value="qat">qat</option>
        <option value="ptq">ptq</option>
      </select>
      <button data-r="all" class="active">All</button>
      <button data-r="correct">Correct</button>
      <button data-r="wrong">Incorrect</button>
    </div>
    <div class="row">
      <span class="glabel">Cross model</span>
      <button data-r="all3wrong">All 3 incorrect (__ALL3WRONG__)</button>
      <button data-r="anywrong">Any 1 incorrect · sharp (__ANYWRONG__)</button>
      <button data-r="blurry">Blurry removed (__BLURRY__)</button>
    </div>
    <div class="row">
      <span class="glabel">Search</span>
      <input id="search" type="text" placeholder="Search image name or serial…">
      <button id="download">⬇ Download CSV</button>
      <span class="count" id="count"></span>
      <span class="count" id="annotcount" style="color:#7ee2a1;"></span>
    </div>
  </div>
</header>
<div id="lightbox">
  <span id="lbclose">&times;</span>
  <img id="lbimg" src="" alt="">
  <div id="lbname"></div>
  <div id="lbhint">click / scroll to zoom · Esc or click outside to close</div>
</div>
<main><div class="grid" id="grid"></div>
  <div id="more" style="text-align:center; padding:24px;">
    <button id="loadmore" style="background:#2563eb;color:#fff;border:0;padding:10px 20px;border-radius:8px;cursor:pointer;">Load more</button>
  </div>
</main>
<script>
const DATA = __DATA__;
const PAGE = 60;
let model = "fp32", result = "all", query = "", shown = 0, view = [];

// manual review labels {image_name: 'correct'|'incorrect'}, persisted in the browser
const ANNOT_KEY = "tpcodl_any1_review_v1";
let annot = {};
try { annot = JSON.parse(localStorage.getItem(ANNOT_KEY) || "{}"); } catch(e) { annot = {}; }

function setAnnot(name, val, btn){
  if (annot[name] === val) { delete annot[name]; }   // click again to clear
  else { annot[name] = val; }
  localStorage.setItem(ANNOT_KEY, JSON.stringify(annot));
  // update the two buttons in this card
  const wrap = btn.parentElement;
  wrap.querySelector(".c").classList.toggle("sel", annot[name] === "correct");
  wrap.querySelector(".w").classList.toggle("sel", annot[name] === "incorrect");
  updateAnnotCount();
}

function updateAnnotCount(){
  const el = document.getElementById("annotcount");
  if (!el) return;
  const n = Object.keys(annot).length;
  const c = Object.values(annot).filter(v => v === "correct").length;
  const w = Object.values(annot).filter(v => v === "incorrect").length;
  el.textContent = `reviewed ${n}  (correct ${c} · incorrect ${w})`;
}

function esc(s){ return (s==null?"":String(s)); }

function applyFilter(){
  view = DATA.filter(d => {
    const ok = d[model + "_ok"];             // correctness of the selected model
    const allWrong = !d.fp32_ok && !d.qat_ok && !d.ptq_ok;
    const anyWrong = !d.fp32_ok || !d.qat_ok || !d.ptq_ok;
    if (result === "correct" && !ok) return false;
    if (result === "wrong" && ok) return false;
    if (result === "all3wrong" && !allWrong) return false;              // cross-model
    if (result === "anywrong" && (!anyWrong || d.blurry)) return false; // cross-model, sharp only
    if (result === "blurry" && !d.blurry) return false;                 // removed blurry set
    if (query){
      const q = query.toLowerCase();
      const hay = (d.name+" "+d.gt+" "+d.fp32+" "+d.qat+" "+d.ptq).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  shown = 0;
  document.getElementById("grid").innerHTML = "";
  render();
}

function badge(ok){ return ok ? '<span class="ok">✔</span>' : '<span class="bad">✘</span>'; }

function card(d){
  const blurTag = d.blurry ? '<span style="float:right;font-size:11px;color:#fbbf24;">blurry</span>' : '';
  return `<div class="card">
    <div class="imgwrap">${blurTag}<img loading="lazy" src="${esc(d.img)}" alt="${esc(d.name)}" onclick="openLB('${esc(d.img)}','${esc(d.name)}')"></div>
    <div class="namerow">
      <input readonly value="${esc(d.name)}" onclick="this.select()">
      <button onclick="navigator.clipboard.writeText('${esc(d.name)}');this.textContent='Copied';setTimeout(()=>this.textContent='Copy',1000)">Copy</button>
    </div>
    <div class="vals">
      <div class="line gt"><span class="lab">Ground truth (bot)</span><span class="val">${esc(d.gt)||'—'}</span></div>
      <div class="line"><span class="lab">fp32 ${badge(d.fp32_ok)}</span><span class="val">${esc(d.fp32)||'—'}</span></div>
      <div class="line"><span class="lab">qat ${badge(d.qat_ok)}</span><span class="val">${esc(d.qat)||'—'}</span></div>
      <div class="line"><span class="lab">ptq ${badge(d.ptq_ok)}</span><span class="val">${esc(d.ptq)||'—'}</span></div>
    </div>
    ${annotHTML(d)}
  </div>`;
}

// manual Correct/Incorrect review buttons — shown ONLY in the "Any 1 incorrect · sharp" filter
function annotHTML(d){
  if (result !== "anywrong") return "";
  const v = annot[d.name] || "";
  return `<div class="annot" data-name="${esc(d.name)}">
    <button class="c ${v==='correct'?'sel':''}" onclick="setAnnot('${esc(d.name)}','correct',this)">Correct</button>
    <button class="w ${v==='incorrect'?'sel':''}" onclick="setAnnot('${esc(d.name)}','incorrect',this)">Incorrect</button>
  </div>`;
}

function render(){
  const grid = document.getElementById("grid");
  const next = view.slice(shown, shown + PAGE);
  grid.insertAdjacentHTML("beforeend", next.map(card).join(""));
  shown += next.length;
  const crossModel = (result === "all3wrong" || result === "anywrong" || result === "blurry");
  const labels = { all: "all", correct: "correct", wrong: "incorrect",
                   all3wrong: "all 3 incorrect", anywrong: "any 1 incorrect · sharp",
                   blurry: "blurry removed" };
  const scope = crossModel ? labels[result] : `${model} · ${labels[result]}`;
  document.getElementById("count").textContent =
    `${scope} — showing ${shown.toLocaleString()} of ${view.length.toLocaleString()} (total ${DATA.length.toLocaleString()})`;
  // dim the model dropdown when the filter is cross-model (dropdown has no effect)
  document.getElementById("model").style.opacity = crossModel ? 0.4 : 1;
  document.getElementById("more").style.display = shown < view.length ? "block" : "none";
}

document.querySelectorAll(".controls button[data-r]").forEach(b => {
  b.onclick = () => {
    document.querySelectorAll(".controls button[data-r]").forEach(x=>x.classList.remove("active"));
    b.classList.add("active"); result = b.dataset.r; applyFilter();
  };
});
document.getElementById("model").onchange = e => { model = e.target.value; applyFilter(); };
document.getElementById("search").oninput = e => { query = e.target.value.trim(); applyFilter(); };

// ---- download the CURRENT filtered view as CSV ----
function csvCell(v){ v = (v==null?"":String(v)); return '"' + v.replace(/"/g,'""') + '"'; }
document.getElementById("download").onclick = () => {
  const cols = ["image_name","ground_truth","fp32","qat","ptq",
                "fp32_correct","qat_correct","ptq_correct","blurry","manual_review"];
  const lines = [cols.join(",")];
  view.forEach(d => {
    lines.push([
      d.name, d.gt, d.fp32, d.qat, d.ptq,
      d.fp32_ok, d.qat_ok, d.ptq_ok, d.blurry, (annot[d.name] || "")
    ].map(csvCell).join(","));
  });
  const rlabel = (result === "anywrong") ? "any1_incorrect_sharp" : result;
  const blob = new Blob([lines.join("\n")], {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `tpcodl_${rlabel}_${view.length}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
};
document.getElementById("loadmore").onclick = render;

// ---- lightbox / zoom ----
const lb = document.getElementById("lightbox");
const lbimg = document.getElementById("lbimg");
let lbscale = 1;
function openLB(src, name){
  lbimg.src = src; lbscale = 1; lbimg.style.transform = "scale(1)";
  document.getElementById("lbname").textContent = name;
  lb.classList.add("open");
}
function closeLB(){ lb.classList.remove("open"); lbimg.src = ""; }
document.getElementById("lbclose").onclick = closeLB;
lb.onclick = e => { if (e.target === lb) closeLB(); };
lbimg.onclick = e => {  // click image toggles 1x <-> 2.5x zoom
  e.stopPropagation();
  lbscale = lbscale > 1 ? 1 : 2.5;
  lbimg.style.transform = `scale(${lbscale})`;
};
lb.addEventListener("wheel", e => {  // scroll to zoom
  if (!lb.classList.contains("open")) return;
  e.preventDefault();
  lbscale = Math.min(8, Math.max(1, lbscale + (e.deltaY < 0 ? 0.25 : -0.25)));
  lbimg.style.transform = `scale(${lbscale})`;
}, { passive: false });
document.addEventListener("keydown", e => { if (e.key === "Escape") closeLB(); });

applyFilter();
updateAnnotCount();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
