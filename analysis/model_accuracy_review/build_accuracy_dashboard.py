#!/usr/bin/env python3
"""
Offline dashboard to MEASURE model accuracy, correcting for a distrusted label.

Universe: all readable images where PTQ == FP32 (26,148 - 6,727 = 19,421). The 6,727
images where PTQ disagrees with FP32 are dropped as unreliable/ambiguous.

Each kept image falls in one of three categories:
  A  match    (13,252)  all 3 models == ground-truth label            -> model correct
  B  agree    ( 5,114)  all 3 models agree but != label               -> model correct, LABEL wrong
  C  review   ( 1,055)  PTQ==FP32 but QAT differs, not all match label -> needs your review

Baseline model accuracy = (A + B) / total = 18,366 / 19,421 = 94.57%.
The page shows a LIVE accuracy that updates as you tick/untick "count as correct" on the
category-C images (and any override you make elsewhere).

Kept features from the other dashboards: click-to-zoom crops, filename copy button,
filters with live counts, per-filter CSV export. Source images are referenced by
relative path so it works offline inside the project.

Run:
    python3 model_accuracy_review/build_accuracy_dashboard.py
"""
import os, json

import numpy as np
for _a in ("float", "int", "bool", "object", "str"):
    if not hasattr(np, _a):
        setattr(np, _a, getattr(__builtins__, _a, None) if not isinstance(__builtins__, dict)
                else __builtins__.get(_a))
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

COMBINED = "../../datasets/final_test/combined_results.xlsx"
IMG_REL = "../../datasets/final_test/serialno_testing/images"
IMG_DIR = IMG_REL
OUT = "dashboard.html"


def load_rows():
    """Keep PTQ==FP32 rows; tag each with category A=match / B=agree / C=review."""
    wb = openpyxl.load_workbook(COMBINED, read_only=True, data_only=True)
    out = []
    for r in wb.worksheets[0].iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        fn, gt, fp, qa, pt = [("" if v is None else str(v)) for v in r[:5]]
        if pt != fp:                       # drop PTQ != FP32
            continue
        if fp == gt and qa == gt and pt == gt:
            cat = "match"
        elif fp == qa == pt and fp != gt:
            cat = "agree"
        else:
            cat = "review"
        out.append([fn, gt, fp, qa, pt, cat])
    wb.close()
    return out


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model accuracy review — meter OCR</title>
<style>
  :root {
    --bg:#0f1216; --panel:#171b21; --panel2:#1e242c; --line:#2a323d;
    --text:#e6edf3; --muted:#8b98a5; --ok:#3fb950; --bad:#f85149; --accent:#58a6ff; --warn:#d29922;
  }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f4f6f8; --panel:#fff; --panel2:#f0f3f6; --line:#d7dee6;
            --text:#1a2029; --muted:#5c6b7a; --ok:#1a7f37; --bad:#cf222e; --accent:#0969da; --warn:#9a6700; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  header { position:sticky; top:0; z-index:20; background:var(--panel);
           border-bottom:1px solid var(--line); padding:14px 18px; }
  h1 { margin:0 0 8px; font-size:16px; font-weight:600; letter-spacing:.2px; }
  h1 small { color:var(--muted); font-weight:400; margin-left:8px; }
  .stats { display:flex; gap:10px; flex-wrap:wrap; }
  .stat { background:var(--panel2); border:1px solid var(--line); border-radius:8px;
          padding:8px 13px; min-width:130px; }
  .stat .name { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; }
  .stat .val { font-size:20px; font-weight:650; margin-top:2px; }
  .stat .sub { font-size:11px; color:var(--muted); }
  .stat.hero { border-color:var(--accent); }
  .stat.hero .val { color:var(--accent); font-size:24px; }
  .stat.warn { border-color:var(--warn); } .stat.warn .val { color:var(--warn); }
  .controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:12px; }
  input, select, button { background:var(--panel2); color:var(--text);
      border:1px solid var(--line); border-radius:6px; padding:6px 10px; font:inherit; }
  input:focus, select:focus { outline:2px solid var(--accent); outline-offset:-1px; }
  button { cursor:pointer; } button:hover:not(:disabled){ border-color:var(--accent); }
  button:disabled { opacity:.4; cursor:default; }
  #count { color:var(--muted); margin-left:auto; font-variant-numeric:tabular-nums; }

  .grid { display:grid; gap:12px; padding:16px 18px;
          grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:9px;
          overflow:hidden; display:flex; flex-direction:column; }
  .card.review { border-color:var(--warn); }
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
  .correctrow { display:flex; align-items:center; gap:7px; padding:5px 6px; margin-bottom:4px;
        background:var(--panel2); border-radius:6px; }
  .correctrow .chk { width:16px; height:16px; cursor:pointer; accent-color:var(--ok); margin:0; }
  .correctrow label { cursor:pointer; font-size:12px; }
  .cattag { margin-left:auto; font-size:10px; text-transform:uppercase; letter-spacing:.5px;
        padding:1px 6px; border-radius:4px; border:1px solid var(--line); color:var(--muted); }
  .cat-match .cattag { color:var(--ok); border-color:var(--ok); }
  .cat-agree .cattag { color:var(--accent); border-color:var(--accent); }
  .cat-review .cattag { color:var(--warn); border-color:var(--warn); }
  .r { display:flex; align-items:center; gap:8px; padding:3px 0; border-top:1px solid var(--line); }
  .r:first-of-type { border-top:0; }
  .lbl { font-size:10px; color:var(--muted); width:34px; flex:none;
         text-transform:uppercase; letter-spacing:.5px; }
  .val { font:13px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace; word-break:break-all; }
  .gt .val { font-weight:700; }
  .mark { margin-left:auto; font-size:12px; font-weight:700; flex:none; }
  .ok .val, .ok .mark { color:var(--ok); } .bad .val, .bad .mark { color:var(--bad); }
  .empty { color:var(--muted); font-style:italic; }

  .pager { display:flex; gap:6px; justify-content:center; align-items:center;
           padding:6px 18px 28px; flex-wrap:wrap; }
  .pager input { width:72px; text-align:center; }
  #modal { position:fixed; inset:0; z-index:50; background:rgba(0,0,0,.82);
           display:none; align-items:center; justify-content:center; flex-direction:column; gap:14px; padding:24px; }
  #modal.on { display:flex; }
  #modal img { background:#fff; padding:8px; border-radius:6px; max-width:94vw; max-height:58vh; image-rendering:pixelated; }
  #mInfo { background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:12px 16px; max-width:min(680px,94vw); width:100%; }
  #mNav { display:flex; gap:10px; align-items:center; }
  .zoomctl { display:flex; gap:6px; align-items:center; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>Model accuracy review <small id="sub"></small></h1>
  <div class="stats" id="stats"></div>
  <div class="controls">
    <input id="q" type="search" placeholder="Search filename or serial…" size="26">
    <select id="filter">
      <option value="all">All kept images</option>
      <option value="match">✓ Correct — matches label</option>
      <option value="agree">✓ Correct — label wrong, all agree</option>
      <option value="review">⚠ Needs review — QAT differs</option>
      <option value="counted_ok">Counted correct (live)</option>
      <option value="counted_no">Counted incorrect (live)</option>
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
  <div id="mNav"><button id="prev">← Prev</button><span id="mPos"></span><button id="next">Next →</button></div>
</div>

<script>
const DATA = __DATA__;            // [filename, gt, fp32, qat, ptq, cat]  cat: match|agree|review
const IMGDIR = __IMGDIR__;
const MODELS = [["fp32",2],["qat",3],["ptq",4]];
const src = r => IMGDIR + "/" + encodeURIComponent(r[0]);
const ok  = (r,i) => r[i] === r[1];
const CATLBL = {match:"matches label", agree:"label wrong", review:"review"};

// "count as correct" per image. Default: match & agree = correct; review = not (pending).
const MARKS = JSON.parse(localStorage.getItem("accuracy_marks_v1") || "{}");
const dfltCorrect = r => r[5] !== "review";
const isCorrect = r => (r[0] in MARKS) ? MARKS[r[0]] : dfltCorrect(r);
function setCorrect(r, val) {
  if (val === dfltCorrect(r)) delete MARKS[r[0]]; else MARKS[r[0]] = val;
  localStorage.setItem("accuracy_marks_v1", JSON.stringify(MARKS));
}

const fnLine = r =>
  `<div class="fn"><span class="name" title="${escape_(r[0])}">${escape_(r[0])}</span>` +
  `<button class="copybtn" data-fn="${escape_(r[0])}" title="Copy image name">copy</button></div>`;
function copyName(btn) {
  const txt = btn.dataset.fn;
  const done = () => { btn.classList.add("done"); btn.textContent="copied";
    setTimeout(()=>{ btn.classList.remove("done"); btn.textContent="copy"; }, 1100); };
  if (navigator.clipboard && navigator.clipboard.writeText)
    navigator.clipboard.writeText(txt).then(done, () => fallbackCopy(txt, done));
  else fallbackCopy(txt, done);
}
function fallbackCopy(txt, done) {
  const ta=document.createElement("textarea"); ta.value=txt;
  ta.style.position="fixed"; ta.style.opacity="0"; document.body.appendChild(ta);
  ta.select(); try { document.execCommand("copy"); done(); } catch(_){}
  document.body.removeChild(ta);
}

const total = DATA.length;
const CAT = {match:0, agree:0, review:0};
for (const r of DATA) CAT[r[5]]++;

function renderStats() {
  const correct = DATA.reduce((a,r)=>a+(isCorrect(r)?1:0),0);
  const acc = (100*correct/total).toFixed(2);
  document.getElementById("sub").textContent =
    total.toLocaleString() + " images kept (PTQ = FP32) · " + (26148-total).toLocaleString() + " dropped (PTQ ≠ FP32)";
  const tile = (name,val,sub,cls="") =>
    `<div class="stat ${cls}"><div class="name">${name}</div><div class="val">${val}</div><div class="sub">${sub}</div></div>`;
  document.getElementById("stats").innerHTML =
    tile("Model accuracy", acc+"%", `${correct.toLocaleString()} / ${total.toLocaleString()} correct`, "hero") +
    tile("Total kept", total.toLocaleString(), "PTQ = FP32") +
    tile("Correct — matches label", CAT.match.toLocaleString(), "all 3 == ground truth") +
    tile("Correct — label wrong", CAT.agree.toLocaleString(), "all 3 agree, ≠ label") +
    tile("Needs review", CAT.review.toLocaleString(), "QAT differs — tick to include", "warn");
}
renderStats();

let view = DATA, page = 0;
const q = document.getElementById("q"), filter = document.getElementById("filter"),
      sizeSel = document.getElementById("size"), grid = document.getElementById("grid");

function pass(r, f) {
  switch (f) {
    case "match":       return r[5]==="match";
    case "agree":       return r[5]==="agree";
    case "review":      return r[5]==="review";
    case "counted_ok":  return isCorrect(r);
    case "counted_no":  return !isCorrect(r);
    default:            return true;
  }
}
function refreshCounts() {
  const fc = {};
  for (const o of filter.options) fc[o.value] = 0;
  for (const r of DATA) for (const o of filter.options) if (pass(r,o.value)) fc[o.value]++;
  for (const o of filter.options) {
    const base = o.textContent.replace(/\\s*\\(\\d[\\d,]*\\)\\s*$/, "");
    o.textContent = `${base} (${(fc[o.value]||0).toLocaleString()})`;
  }
}
refreshCounts();

function apply() {
  const t = q.value.trim().toUpperCase(), f = filter.value;
  document.getElementById("export").style.display = (f === "all") ? "none" : "";
  view = DATA.filter(r => {
    if (t && !(r[0].toUpperCase().includes(t) || r[1].toUpperCase().includes(t) ||
               r[2].toUpperCase().includes(t) || r[3].toUpperCase().includes(t))) return false;
    return pass(r, f);
  });
  page = 0; render();
}

function modelRows(r, longmark) {
  return MODELS.map(([m,i]) => {
    const good = ok(r,i);
    const val = r[i] ? escape_(r[i]) : '<span class="empty">(no read)</span>';
    const mk = good ? (longmark?'✓ = label':'✓') : (longmark?'✗ ≠ label':'✗');
    return `<div class="r ${good?'ok':'bad'}"><span class="lbl">${m}</span>
      <span class="val">${val}</span><span class="mark">${mk}</span></div>`;
  }).join("");
}
function correctRow(r) {
  return `<div class="correctrow">
    <input type="checkbox" class="cc" data-fn="${escape_(r[0])}" ${isCorrect(r)?"checked":""}>
    <label>count as correct</label><span class="cattag">${CATLBL[r[5]]}</span></div>`;
}

function render() {
  const per = +sizeSel.value, pages = Math.max(1, Math.ceil(view.length/per));
  page = Math.min(page, pages-1);
  const from = page*per, slice = view.slice(from, from+per);
  document.getElementById("count").textContent = view.length
    ? `showing ${(from+1).toLocaleString()}–${(from+slice.length).toLocaleString()} of ${view.length.toLocaleString()}`
    : "no matches";

  grid.innerHTML = slice.map((r,k) => `<div class="card cat-${r[5]} ${r[5]==='review'?'review':''}">
      <div class="shot" data-i="${from+k}"><img loading="lazy" src="${src(r)}" alt=""></div>
      <div class="rows">
        ${fnLine(r)}
        ${correctRow(r)}
        <div class="r gt"><span class="lbl">truth</span><span class="val">${escape_(r[1])}</span></div>
        ${modelRows(r,false)}
      </div></div>`).join("");

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

// react to a "count as correct" toggle
function onCorrect(fn, checked) {
  const r = DATA.find(x => x[0] === fn); if (!r) return;
  setCorrect(r, checked);
  renderStats(); refreshCounts();
  if (filter.value === "counted_ok" || filter.value === "counted_no") apply();
}

const modal=document.getElementById("modal"), mImg=document.getElementById("mImg"),
      mInfo=document.getElementById("mInfo"), mPos=document.getElementById("mPos"),
      zLvl=document.getElementById("zLvl");
let cur=-1, zoom=4;
grid.addEventListener("click", e => {
  const b = e.target.closest(".copybtn"); if (b) { copyName(b); return; }
  if (e.target.closest(".correctrow")) return;   // let the checkbox handle itself
  const s = e.target.closest(".shot"); if (s) open(+s.dataset.i);
});
grid.addEventListener("change", e => {
  const c = e.target.closest(".cc"); if (c) onCorrect(c.dataset.fn, c.checked);
});
function open(i) {
  if (i<0 || i>=view.length) return;
  cur=i; const r=view[i];
  mImg.src=src(r); applyZoom();
  mInfo.innerHTML = fnLine(r) + correctRow(r) +
    `<div class="r gt"><span class="lbl">truth</span><span class="val">${escape_(r[1])}</span></div>` +
    modelRows(r,true);
  mPos.textContent = `${(i+1).toLocaleString()} / ${view.length.toLocaleString()}`;
  modal.classList.add("on");
}
mInfo.addEventListener("change", e => {
  const c = e.target.closest(".cc"); if (c) onCorrect(c.dataset.fn, c.checked);
});
mInfo.addEventListener("click", e => {
  const b = e.target.closest(".copybtn"); if (b) copyName(b);
});
function applyZoom(){ mImg.style.width=""; mImg.style.transform=`scale(${zoom})`;
  mImg.style.transformOrigin="center"; zLvl.textContent = zoom+"×"; }
document.getElementById("zIn").onclick = e => { e.stopPropagation(); zoom=Math.min(12,zoom+1); applyZoom(); };
document.getElementById("zOut").onclick = e => { e.stopPropagation(); zoom=Math.max(1,zoom-1); applyZoom(); };
document.getElementById("prev").onclick = e => { e.stopPropagation(); step(-1); };
document.getElementById("next").onclick = e => { e.stopPropagation(); step(1); };
function step(d){ const n=cur+d; if(n<0||n>=view.length) return;
  const per=+sizeSel.value,p=Math.floor(n/per); if(p!==page){page=p;render();} open(n); }
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

// ---- CSV export: current filter only, with accuracy summary ----
const csvcell = v => `"${String(v).replace(/"/g,'""')}"`;
const FSLUG = {all:"all_kept", match:"correct_matches_label", agree:"correct_label_wrong",
  review:"needs_review_qat_differs", counted_ok:"counted_correct", counted_no:"counted_incorrect"};
function exportCSV() {
  const rows = view;
  if (!rows.length) { alert("Nothing to export — this filter has no rows."); return; }
  const fval = filter.value, flabel = filter.options[filter.selectedIndex].text;
  const head = ["filename","final_results","fp32_result","qat_result","ptq_result",
    "category","fp32_eq_label","counted_correct"];
  const lines = [head.join(",")];
  let correct=0, cm=0,ca=0,cr=0;
  for (const r of rows) {
    const cc = isCorrect(r);
    if (cc) correct++;
    if (r[5]==="match") cm++; else if (r[5]==="agree") ca++; else cr++;
    lines.push([csvcell(r[0]),csvcell(r[1]),csvcell(r[2]),csvcell(r[3]),csvcell(r[4]),
      csvcell(r[5]), ok(r,2)?"TRUE":"FALSE", cc?"TRUE":"FALSE"].join(","));
  }
  const N=rows.length, pct=(a,b)=>b?(100*a/b).toFixed(2)+"%":"0%";
  const push=(...c)=>lines.push(c.map(csvcell).join(","));
  lines.push(""); lines.push(csvcell("SUMMARY"));
  push("filter", flabel.replace(/\\s*\\(\\d[\\d,]*\\)\\s*$/,""));
  const qv=q.value.trim(); if (qv) push("search", qv);
  push("images in filter", N);
  push("category: matches label", cm);
  push("category: label wrong (all agree)", ca);
  push("category: needs review (qat differs)", cr);
  push("counted correct", correct);
  push("counted incorrect", N-correct);
  push("MODEL ACCURACY", pct(correct,N));
  const blob = new Blob(["\\ufeff"+lines.join("\\r\\n")], {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `accuracy_${FSLUG[fval]||fval}.csv`;
  a.click(); URL.revokeObjectURL(a.href);
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
    cats = {"match": 0, "agree": 0, "review": 0}
    for r in rows:
        cats[r[5]] += 1
    print(f"kept (PTQ==FP32) : {len(rows)}")
    print(f"crops on disk    : {len(rows) - len(missing)}")
    if missing:
        print(f"WARNING: {len(missing)} rows have no image file, e.g. {missing[:3]}")
    print(f"  match  (== label)        : {cats['match']}")
    print(f"  agree  (label wrong)     : {cats['agree']}")
    print(f"  review (qat differs)     : {cats['review']}")
    correct = cats["match"] + cats["agree"]
    print(f"  baseline accuracy        : {correct}/{len(rows)} = {100.0*correct/len(rows):.2f}%")

    page = (TEMPLATE
            .replace("__DATA__", json.dumps(rows, separators=(",", ":")))
            .replace("__IMGDIR__", json.dumps(IMG_REL)))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"\nwrote {os.path.join(HERE, OUT)}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"open it with:  xdg-open '{os.path.join(HERE, OUT)}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
