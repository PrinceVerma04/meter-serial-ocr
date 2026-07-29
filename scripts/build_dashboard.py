#!/usr/bin/env python3
"""
Build an offline, interactive dashboard from combined_results.xlsx.

For every image in combined_results.xlsx it shows the crop plus the ground truth
(final_results) and the three model reads (fp32/qat/ptq), each marked correct or
wrong by EXACT string match. Per-model accuracy sits at the top.

Output is a single self-contained HTML file that references the crops by RELATIVE
path, so it works offline (file://) as long as it stays next to serialno_testing/.
The 175 MB of crops are NOT embedded — that would make the page unopenable.

Inputs
    datasets/final_test/combined_results.xlsx
    datasets/final_test/serialno_testing/images/

Output
    datasets/final_test/dashboard.html

Run:
    python3 scripts/build_dashboard.py
then open "datasets/final_test/dashboard.html" in a browser.
"""
import os, json, html

import numpy as np
# numpy-alias shim (old openpyxl in the import chain on this box)
for _a in ("float", "int", "bool", "object", "str"):
    if not hasattr(np, _a):
        setattr(np, _a, getattr(__builtins__, _a, None) if not isinstance(__builtins__, dict)
                else __builtins__.get(_a))
import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

BASE = "datasets/final_test"
COMBINED = f"{BASE}/combined_results.xlsx"
IMG_DIR = f"{BASE}/serialno_testing/images"
IMG_REL = "serialno_testing/images"          # relative to the html file
OUT = f"{BASE}/dashboard.html"


def load_rows():
    wb = openpyxl.load_workbook(COMBINED, read_only=True, data_only=True)
    rows = []
    for r in wb.worksheets[0].iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        rows.append([("" if v is None else str(v)) for v in r[:5]])
    wb.close()
    return rows


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meter OCR — model comparison dashboard</title>
<style>
  :root {
    --bg:#0f1216; --panel:#171b21; --panel2:#1e242c; --line:#2a323d;
    --text:#e6edf3; --muted:#8b98a5; --ok:#3fb950; --bad:#f85149; --accent:#58a6ff;
  }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f4f6f8; --panel:#fff; --panel2:#f0f3f6; --line:#d7dee6;
            --text:#1a2029; --muted:#5c6b7a; --ok:#1a7f37; --bad:#cf222e; --accent:#0969da; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  header { position:sticky; top:0; z-index:20; background:var(--panel);
           border-bottom:1px solid var(--line); padding:14px 18px; }
  h1 { margin:0 0 12px; font-size:16px; font-weight:600; letter-spacing:.2px; }
  h1 small { color:var(--muted); font-weight:400; margin-left:8px; }
  .stats { display:flex; gap:10px; flex-wrap:wrap; }
  .stat { background:var(--panel2); border:1px solid var(--line); border-radius:8px;
          padding:9px 14px; min-width:150px; }
  .stat .name { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.6px; }
  .stat .val { font-size:22px; font-weight:650; margin-top:2px; }
  .stat .sub { font-size:11px; color:var(--muted); }
  .controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:12px; }
  input, select, button { background:var(--panel2); color:var(--text);
      border:1px solid var(--line); border-radius:6px; padding:6px 10px; font:inherit; }
  input:focus, select:focus { outline:2px solid var(--accent); outline-offset:-1px; }
  button { cursor:pointer; }
  button:hover:not(:disabled) { border-color:var(--accent); }
  button:disabled { opacity:.4; cursor:default; }
  #count { color:var(--muted); margin-left:auto; font-variant-numeric:tabular-nums; }

  .grid { display:grid; gap:12px; padding:16px 18px;
          grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:9px;
          overflow:hidden; display:flex; flex-direction:column; }
  .shot { background:#fff; display:flex; align-items:center; justify-content:center;
          min-height:86px; cursor:zoom-in; border-bottom:1px solid var(--line); padding:6px; }
  .shot img { max-width:100%; max-height:120px; display:block; }
  .rows { padding:8px 10px 10px; }
  .fn { display:flex; align-items:center; gap:6px; margin-bottom:6px; }
  .fn .name { font:11px ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted);
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis; user-select:all; flex:1; }
  .copybtn { flex:none; cursor:pointer; border:1px solid var(--line); background:var(--panel2);
        color:var(--muted); border-radius:5px; font-size:11px; padding:1px 7px; line-height:1.5; }
  .copybtn:hover { border-color:var(--accent); color:var(--accent); }
  .copybtn.done { color:var(--ok); border-color:var(--ok); }
  .r { display:flex; align-items:center; gap:8px; padding:3px 0;
       border-top:1px solid var(--line); }
  .r:first-of-type { border-top:0; }
  .chk { width:15px; height:15px; flex:none; cursor:pointer; accent-color:var(--accent); margin:0; }
  .gt .chkspace { width:15px; flex:none; }
  .lbl { font-size:10px; color:var(--muted); width:34px; flex:none;
         text-transform:uppercase; letter-spacing:.5px; }
  .val { font:13px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace; word-break:break-all; }
  .gt .val { font-weight:700; }
  .mark { margin-left:auto; font-size:12px; font-weight:700; flex:none; }
  .ok .val, .ok .mark { color:var(--ok); }
  .bad .val, .bad .mark { color:var(--bad); }
  .empty { color:var(--muted); font-style:italic; }

  .pager { display:flex; gap:6px; justify-content:center; align-items:center;
           padding:6px 18px 28px; flex-wrap:wrap; }
  .pager input { width:72px; text-align:center; }

  #modal { position:fixed; inset:0; z-index:50; background:rgba(0,0,0,.82);
           display:none; align-items:center; justify-content:center; flex-direction:column; gap:14px; padding:24px; }
  #modal.on { display:flex; }
  #modal img { background:#fff; padding:8px; border-radius:6px;
               max-width:94vw; max-height:58vh; image-rendering:pixelated; }
  #mInfo { background:var(--panel); border:1px solid var(--line); border-radius:9px;
           padding:12px 16px; max-width:min(680px,94vw); width:100%; }
  #mNav { display:flex; gap:10px; align-items:center; }
  .zoomctl { display:flex; gap:6px; align-items:center; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>Meter OCR — model comparison <small id="sub"></small></h1>
  <div class="stats" id="stats"></div>
  <div class="controls">
    <input id="q" type="search" placeholder="Search filename or serial…" size="26">
    <select id="filter">
      <option value="all">All images</option>
      <option value="accept_all">✓ Accepted by all 3 (all match GT)</option>
      <option value="reject_any">✗ Not accepted (any model ≠ GT)</option>
      <option value="gt_ok">Ground truth OK</option>
      <option value="gt_bad">Ground truth flagged wrong</option>
      <option value="fp32bad">FP32 wrong</option>
      <option value="qatbad">QAT wrong</option>
      <option value="ptqbad">PTQ wrong</option>
      <option value="allok">All 3 correct</option>
      <option value="allbad">All 3 wrong</option>
      <option value="split">Models disagree</option>
    </select>
    <select id="size">
      <option>100</option><option selected>200</option><option>500</option>
    </select>
    <button id="export" style="display:none" title="Download a CSV of the CURRENT filter only, with an accuracy summary">⭳ Export CSV (this filter)</button>
    <span id="count"></span>
  </div>
</header>

<div class="grid" id="grid"></div>
<div class="pager" id="pager"></div>

<div id="modal">
  <img id="mImg" alt="">
  <div class="zoomctl">
    <button id="zOut">−</button><span id="zLvl">4×</span><button id="zIn">+</button>
    <span style="margin-left:10px">click outside or Esc to close · ← → to move</span>
  </div>
  <div id="mInfo"></div>
  <div id="mNav">
    <button id="prev">← Prev</button><span id="mPos"></span><button id="next">Next →</button>
  </div>
</div>

<script>
const DATA = __DATA__;            // [filename, gt, fp32, qat, ptq]
const IMGDIR = __IMGDIR__;
const MODELS = [["fp32",2],["qat",3],["ptq",4]];
const src = r => IMGDIR + "/" + encodeURIComponent(r[0]);
const ok  = (r,i) => r[i] === r[1];               // exact string match

// ---- manual checkboxes: default to the auto match, remember overrides ----
// model rows default to the exact-match result; the "gt" row defaults to true
// (ground truth assumed correct until a reviewer unticks it).
const MARKS = JSON.parse(localStorage.getItem("meterocr_marks_v1") || "{}");
const mkey  = (r,m) => r[0] + "|" + m;
const dflt   = (r,m,i) => m === "gt" ? true : ok(r,i);
const marked = (r,m,i) => { const k=mkey(r,m); return k in MARKS ? MARKS[k] : dflt(r,m,i); };
function setMark(r,m,i,val) {
  const k=mkey(r,m);
  if (val === dflt(r,m,i)) delete MARKS[k];       // back to default -> don't store
  else MARKS[k] = val;
  localStorage.setItem("meterocr_marks_v1", JSON.stringify(MARKS));
}
const chkbox = (r,m,i) =>
  `<input type="checkbox" class="chk" data-fn="${escape_(r[0])}" data-m="${m}" data-i="${i}" ${marked(r,m,i)?"checked":""}>`;

// filename with a copy-to-clipboard button
const fnLine = r =>
  `<div class="fn"><span class="name" title="${escape_(r[0])}">${escape_(r[0])}</span>` +
  `<button class="copybtn" data-fn="${escape_(r[0])}" title="Copy image name">copy</button></div>`;
function copyName(btn) {
  const txt = btn.dataset.fn;
  const done = () => { btn.classList.add("done"); btn.textContent="copied";
    setTimeout(()=>{ btn.classList.remove("done"); btn.textContent="copy"; }, 1100); };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(txt).then(done, () => fallbackCopy(txt, done));
  } else fallbackCopy(txt, done);
}
function fallbackCopy(txt, done) {                 // file:// without clipboard API
  const ta=document.createElement("textarea"); ta.value=txt;
  ta.style.position="fixed"; ta.style.opacity="0"; document.body.appendChild(ta);
  ta.select(); try { document.execCommand("copy"); done(); } catch(_){}
  document.body.removeChild(ta);
}

// ---- accuracy (top of dashboard) ----
const total = DATA.length;
document.getElementById("sub").textContent =
  total.toLocaleString() + " images · ground truth = final_results · exact string match";
document.getElementById("stats").innerHTML = MODELS.map(([m,i]) => {
  const c = DATA.reduce((a,r)=>a+(ok(r,i)?1:0),0);
  return `<div class="stat"><div class="name">${m}</div>
    <div class="val">${(100*c/total).toFixed(2)}%</div>
    <div class="sub">${c.toLocaleString()} / ${total.toLocaleString()} correct</div></div>`;
}).join("") + `<div class="stat"><div class="name">images</div>
    <div class="val">${total.toLocaleString()}</div><div class="sub">blank GT removed</div></div>`;

// ---- filtering ----
let view = DATA, page = 0;
const q = document.getElementById("q"), filter = document.getElementById("filter"),
      sizeSel = document.getElementById("size"), grid = document.getElementById("grid");

// count how many images fall in each filter, and show it in the dropdown label.
// gt_ok / gt_bad depend on the truth checkboxes, so this can be recomputed live.
function refreshCounts() {
  const fc = {all:0,accept_all:0,reject_any:0,fp32bad:0,qatbad:0,ptqbad:0,
              allok:0,allbad:0,split:0,gt_ok:0,gt_bad:0};
  for (const r of DATA) {
    const n = MODELS.reduce((a,[,i])=>a+(ok(r,i)?1:0),0);
    fc.all++;
    if (n===3) { fc.accept_all++; fc.allok++; }
    if (n<3) fc.reject_any++;
    if (!ok(r,2)) fc.fp32bad++;
    if (!ok(r,3)) fc.qatbad++;
    if (!ok(r,4)) fc.ptqbad++;
    if (n===0) fc.allbad++;
    if (n>0 && n<3) fc.split++;
    if (marked(r,"gt",1)) fc.gt_ok++; else fc.gt_bad++;
  }
  for (const o of filter.options) {
    const base = o.textContent.replace(/\s*\(\d[\d,]*\)\s*$/, "");
    o.textContent = `${base} (${(fc[o.value]||0).toLocaleString()})`;
  }
}
refreshCounts();

function apply() {
  const t = q.value.trim().toUpperCase(), f = filter.value;
  // the export button only appears once a real filter is chosen
  document.getElementById("export").style.display = (f === "all") ? "none" : "";
  view = DATA.filter(r => {
    if (t && !(r[0].toUpperCase().includes(t) || r[1].toUpperCase().includes(t) ||
               r[2].toUpperCase().includes(t) || r[3].toUpperCase().includes(t) ||
               r[4].toUpperCase().includes(t))) return false;
    const n = MODELS.reduce((a,[,i])=>a+(ok(r,i)?1:0),0);
    switch (f) {
      case "accept_all": return n === 3;          // all 3 models match ground truth
      case "reject_any": return n < 3;             // at least one model does not match
      case "gt_ok":   return marked(r,"gt",1);     // truth checkbox still ticked
      case "gt_bad":  return !marked(r,"gt",1);    // truth flagged wrong by reviewer
      case "fp32bad": return !ok(r,2);
      case "qatbad":  return !ok(r,3);
      case "ptqbad":  return !ok(r,4);
      case "allok":   return n === 3;
      case "allbad":  return n === 0;
      case "split":   return n > 0 && n < 3;
      default:        return true;
    }
  });
  page = 0; render();
}

function render() {
  const per = +sizeSel.value, pages = Math.max(1, Math.ceil(view.length/per));
  page = Math.min(page, pages-1);
  const from = page*per, slice = view.slice(from, from+per);

  document.getElementById("count").textContent = view.length
    ? `showing ${(from+1).toLocaleString()}–${(from+slice.length).toLocaleString()} of ${view.length.toLocaleString()}`
    : "no matches";

  grid.innerHTML = slice.map((r, k) => {
    const rows = MODELS.map(([m,i]) => {
      const good = ok(r,i);
      const val = r[i] ? escape_(r[i]) : '<span class="empty">(no read)</span>';
      return `<div class="r ${good?'ok':'bad'}">${chkbox(r,m,i)}<span class="lbl">${m}</span>
        <span class="val">${val}</span><span class="mark">${good?'✓':'✗'}</span></div>`;
    }).join("");
    return `<div class="card">
      <div class="shot" data-i="${from+k}"><img loading="lazy" src="${src(r)}" alt=""></div>
      <div class="rows">
        ${fnLine(r)}
        <div class="r gt" title="Untick if this ground-truth label is wrong">${chkbox(r,"gt",1)}<span class="lbl">truth</span><span class="val">${escape_(r[1])}</span></div>
        ${rows}
      </div></div>`;
  }).join("");

  const pg = document.getElementById("pager");
  pg.innerHTML = `<button id="pf" ${page?'':'disabled'}>« first</button>
    <button id="pp" ${page?'':'disabled'}>‹ prev</button>
    <span>page <input id="pi" value="${page+1}"> of ${pages.toLocaleString()}</span>
    <button id="pn" ${page<pages-1?'':'disabled'}>next ›</button>
    <button id="pl" ${page<pages-1?'':'disabled'}>last »</button>`;
  pf.onclick=()=>{page=0;render();scrollTo(0,0);};
  pp.onclick=()=>{page--;render();scrollTo(0,0);};
  pn.onclick=()=>{page++;render();scrollTo(0,0);};
  pl.onclick=()=>{page=pages-1;render();scrollTo(0,0);};
  pi.onchange=()=>{const v=+pi.value; if(v>=1&&v<=pages){page=v-1;render();scrollTo(0,0);}};
}
const escape_ = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

q.oninput = apply; filter.onchange = apply; sizeSel.onchange = () => { page=0; render(); };

// ---- zoom modal ----
const modal=document.getElementById("modal"), mImg=document.getElementById("mImg"),
      mInfo=document.getElementById("mInfo"), mPos=document.getElementById("mPos"),
      zLvl=document.getElementById("zLvl");
let cur=-1, zoom=4;

grid.addEventListener("click", e => {
  const b = e.target.closest(".copybtn"); if (b) { copyName(b); return; }
  const s = e.target.closest(".shot"); if (s) open(+s.dataset.i);
});
grid.addEventListener("change", e => {
  const c = e.target.closest(".chk"); if (!c) return;
  const r = DATA.find(x => x[0] === c.dataset.fn);
  if (r) { setMark(r, c.dataset.m, +c.dataset.i, c.checked); onMark(c.dataset.m); }
});
// after a truth mark changes, gt counts move; refresh labels and re-filter if needed
function onMark(m) {
  if (m !== "gt") return;
  refreshCounts();
  if (filter.value === "gt_ok" || filter.value === "gt_bad") apply();
}
function open(i) {
  if (i<0 || i>=view.length) return;
  cur=i; const r=view[i];
  mImg.src=src(r); applyZoom();
  mInfo.innerHTML = fnLine(r) +
    `<div class="r gt" title="Untick if this ground-truth label is wrong">${chkbox(r,"gt",1)}<span class="lbl">truth</span><span class="val">${escape_(r[1])}</span></div>` +
    MODELS.map(([m,j]) => {
      const good=ok(r,j);
      const val = r[j] ? escape_(r[j]) : '<span class="empty">(no read)</span>';
      return `<div class="r ${good?'ok':'bad'}">${chkbox(r,m,j)}<span class="lbl">${m}</span>
        <span class="val">${val}</span><span class="mark">${good?'✓ correct':'✗ wrong'}</span></div>`;
    }).join("");
  mPos.textContent = `${(i+1).toLocaleString()} / ${view.length.toLocaleString()}`;
  modal.classList.add("on");
}
mInfo.addEventListener("change", e => {
  const c = e.target.closest(".chk"); if (!c) return;
  const r = DATA.find(x => x[0] === c.dataset.fn);
  if (r) { setMark(r, c.dataset.m, +c.dataset.i, c.checked); onMark(c.dataset.m); }
});
mInfo.addEventListener("click", e => {
  const b = e.target.closest(".copybtn"); if (b) copyName(b);
});
function applyZoom(){ mImg.style.width = ""; mImg.style.transform = `scale(${zoom})`;
  mImg.style.transformOrigin="center"; zLvl.textContent = zoom+"×"; }
document.getElementById("zIn").onclick = e => { e.stopPropagation(); zoom=Math.min(12,zoom+1); applyZoom(); };
document.getElementById("zOut").onclick = e => { e.stopPropagation(); zoom=Math.max(1,zoom-1); applyZoom(); };
document.getElementById("prev").onclick = e => { e.stopPropagation(); step(-1); };
document.getElementById("next").onclick = e => { e.stopPropagation(); step(1); };
function step(d) {                      // walks the whole filtered set, across pages
  const n = cur+d; if (n<0 || n>=view.length) return;
  const per=+sizeSel.value, p=Math.floor(n/per);
  if (p!==page) { page=p; render(); }
  open(n);
}
modal.onclick = e => { if (e.target===modal) close_(); };
function close_(){ modal.classList.remove("on"); mImg.src=""; }
addEventListener("keydown", e => {
  if (!modal.classList.contains("on")) return;
  if (e.key==="Escape") close_();
  if (e.key==="ArrowLeft") step(-1);
  if (e.key==="ArrowRight") step(1);
  if (e.key==="+"||e.key==="=") { zoom=Math.min(12,zoom+1); applyZoom(); }
  if (e.key==="-") { zoom=Math.max(1,zoom-1); applyZoom(); }
});

// ---- CSV export: only the CURRENT filter's rows, named after that filter ----
const csvcell = v => `"${String(v).replace(/"/g,'""')}"`;
const FSLUG = {all:"all_images", accept_all:"accepted_by_all3", reject_any:"not_accepted",
  gt_ok:"ground_truth_ok", gt_bad:"ground_truth_flagged_wrong",
  fp32bad:"fp32_wrong", qatbad:"qat_wrong", ptqbad:"ptq_wrong",
  allok:"all3_correct", allbad:"all3_wrong", split:"models_disagree"};

function exportCSV() {
  const rows = view;                         // exactly what the active filter/search shows
  if (!rows.length) { alert("Nothing to export — this filter has no rows."); return; }
  const fval = filter.value, flabel = filter.options[filter.selectedIndex].text;

  const head = ["filename","final_results","gt_label_correct",
    "fp32_result","fp32_exact_match","fp32_reviewed_correct",
    "qat_result","qat_exact_match","qat_reviewed_correct",
    "ptq_result","ptq_exact_match","ptq_reviewed_correct"];
  const lines = [head.join(",")];

  // tallies over THIS filter only. "auto" = original exact match; "reviewed" =
  // your checkboxes, scored over images whose ground-truth label you kept valid.
  const auto = {fp32:0,qat:0,ptq:0}, rev = {fp32:0,qat:0,ptq:0};
  let validGT = 0;
  for (const r of rows) {
    const gtok = marked(r,"gt",1);
    if (gtok) validGT++;
    const c = [csvcell(r[0]), csvcell(r[1]), gtok ? "TRUE":"FALSE"];
    for (const [m,i] of MODELS) {
      const exact = ok(r,i), reviewed = marked(r,m,i);
      if (exact) auto[m]++;
      if (gtok && reviewed) rev[m]++;
      c.push(csvcell(r[i]), exact?"TRUE":"FALSE", reviewed?"TRUE":"FALSE");
    }
    lines.push(c.join(","));
  }

  const N = rows.length, pct = (a,b) => b ? (100*a/b).toFixed(2)+"%" : "0%";
  lines.push("");
  lines.push(csvcell("SUMMARY"));
  lines.push([csvcell("filter"), csvcell(flabel.replace(/\\s*\\(\\d[\\d,]*\\)\\s*$/,""))].join(","));
  const qv = q.value.trim(); if (qv) lines.push([csvcell("search"), csvcell(qv)].join(","));
  lines.push([csvcell("images in this filter"), N].join(","));
  lines.push([csvcell("labels flagged wrong (gt unticked)"), N - validGT].join(","));
  lines.push([csvcell("images with valid ground truth"), validGT].join(","));
  lines.push("");
  lines.push(["metric","fp32","qat","ptq"].map(csvcell).join(","));
  lines.push([csvcell("exact-match correct (auto)"), auto.fp32, auto.qat, auto.ptq].join(","));
  lines.push([csvcell("exact-match accuracy (auto, /filter)"),
    csvcell(pct(auto.fp32,N)), csvcell(pct(auto.qat,N)), csvcell(pct(auto.ptq,N))].join(","));
  lines.push([csvcell("reviewed correct"), rev.fp32, rev.qat, rev.ptq].join(","));
  lines.push([csvcell("reviewed accuracy (/valid ground truth)"),
    csvcell(pct(rev.fp32,validGT)), csvcell(pct(rev.qat,validGT)), csvcell(pct(rev.ptq,validGT))].join(","));

  const blob = new Blob(["\\ufeff"+lines.join("\\r\\n")], {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `review_${FSLUG[fval]||fval}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}
document.getElementById("export").onclick = exportCSV;

apply();
</script>
</body>
</html>
"""


def main():
    rows = load_rows()
    missing = [r[0] for r in rows if not os.path.exists(os.path.join(IMG_DIR, r[0]))]
    print(f"rows in combined_results.xlsx : {len(rows)}")
    print(f"crops found on disk           : {len(rows) - len(missing)}")
    if missing:
        print(f"WARNING: {len(missing)} rows have no image file, e.g. {missing[:3]}")

    correct = {m: sum(1 for r in rows if r[i] == r[1]) for m, i in (("fp32", 2), ("qat", 3), ("ptq", 4))}
    for m, c in correct.items():
        print(f"  {m.upper():<5} {c:>6}/{len(rows)}  {100.0*c/len(rows):.2f}%")

    page = (TEMPLATE
            .replace("__DATA__", json.dumps(rows, separators=(",", ":")))
            .replace("__IMGDIR__", json.dumps(IMG_REL)))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"open it with:  xdg-open '{OUT}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
