#!/usr/bin/env python3
"""
Offline dashboard for the DISPUTED "not accepted" pile (review_not_accepted.csv).

These 12,896 images each had at least one model disagree with the hand-typed label,
and the label itself is not trusted. So this dashboard reports TWO things side by side:
  - model vs ground-truth label (fp32/qat/ptq exact match)
  - model vs model agreement, with FP32 as the reference (PTQ=FP32, QAT=FP32, all-3-agree)
and it foregrounds the "label suspect" pile: images where all 3 models agree on a
reading the label rejects — the strongest sign the label is wrong.

Everything from the main dashboard is kept: click-to-zoom crops, a checkbox in front
of every model row AND the truth row (untick when the label is wrong), a copy button
for the filename, filters that show their counts, and per-filter CSV export.

Self-contained HTML; crops are referenced by relative path (../../datasets/final_test/…)
so it works offline as long as this folder stays inside the project.

Run:
    python3 not_accepted_review/build_not_accepted_dashboard.py
"""
import os, csv, json

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

SRC_CSV = "review_not_accepted.csv"
IMG_REL = "../../datasets/final_test/serialno_testing/images"   # relative to this html
IMG_DIR = IMG_REL                                               # same, for existence check
OUT = "dashboard.html"


def load_rows():
    """[filename, gt, fp32, qat, ptq] from the review CSV (ignoring its summary block)."""
    rows = []
    with open(SRC_CSV, newline="", encoding="utf-8-sig") as fh:
        for r in csv.reader(fh):
            if len(r) != 12 or r[0] in ("", "filename", "SUMMARY"):
                continue
            # cols: 0 filename, 1 final_results, 3 fp32_result, 6 qat_result, 9 ptq_result
            rows.append([r[0], r[1], r[3], r[6], r[9]])
    return rows


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Not-accepted review — meter OCR</title>
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
  h1 { margin:0 0 4px; font-size:16px; font-weight:600; letter-spacing:.2px; }
  h1 small { color:var(--muted); font-weight:400; margin-left:8px; }
  .grouplabel { font-size:10px; color:var(--muted); text-transform:uppercase;
                letter-spacing:.7px; margin:10px 0 4px; }
  .stats { display:flex; gap:10px; flex-wrap:wrap; }
  .stat { background:var(--panel2); border:1px solid var(--line); border-radius:8px;
          padding:8px 13px; min-width:132px; }
  .stat .name { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; }
  .stat .val { font-size:20px; font-weight:650; margin-top:2px; }
  .stat .sub { font-size:11px; color:var(--muted); }
  .stat.suspect { border-color:var(--warn); }
  .stat.suspect .val { color:var(--warn); }
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
  .card.suspect { border-color:var(--warn); }
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
  <h1>Not-accepted review <small id="sub"></small></h1>
  <div class="grouplabel">model vs ground-truth label &nbsp;·&nbsp; model vs model (FP32 = reference)</div>
  <div class="stats" id="stats"></div>
  <div class="controls">
    <input id="q" type="search" placeholder="Search filename or serial…" size="26">
    <select id="filter">
      <option value="all">All images</option>
      <option value="label_suspect">⚠ Label suspect (all models agree ≠ GT)</option>
      <option value="ptq_eq_fp32">PTQ = FP32 (agree)</option>
      <option value="ptq_ne_fp32">PTQ ≠ FP32 (differ)</option>
      <option value="qat_eq_fp32">QAT = FP32 (agree)</option>
      <option value="allagree">All 3 models agree (any label)</option>
      <option value="split">Models disagree with each other</option>
      <option value="fp32_eq_gt">FP32 = ground truth</option>
      <option value="fp32bad">FP32 ≠ ground truth</option>
      <option value="qatbad">QAT ≠ ground truth</option>
      <option value="ptqbad">PTQ ≠ ground truth</option>
      <option value="allbad">All 3 ≠ ground truth</option>
      <option value="gt_ok">Ground truth OK</option>
      <option value="gt_bad">Ground truth flagged wrong</option>
    </select>
    <select id="size">
      <option>100</option><option selected>200</option><option>500</option>
    </select>
    <button id="export" style="display:none" title="Download a CSV of the CURRENT filter only, with a summary">⭳ Export CSV (this filter)</button>
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
const DATA = __DATA__;            // [filename, gt, fp32, qat, ptq]  (indices 0..4)
const IMGDIR = __IMGDIR__;
const MODELS = [["fp32",2],["qat",3],["ptq",4]];
const src = r => IMGDIR + "/" + encodeURIComponent(r[0]);
const ok  = (r,i) => r[i] === r[1];               // model i matches ground-truth label
const agree3  = r => r[2]===r[3] && r[3]===r[4];   // all 3 models identical
const suspect = r => agree3(r) && r[2]!==r[1];     // models agree, label disagrees

// ---- checkboxes: model rows default to exact match; the "gt" row defaults to true ----
const MARKS = JSON.parse(localStorage.getItem("notaccepted_marks_v1") || "{}");
const mkey  = (r,m) => r[0] + "|" + m;
const dflt   = (r,m,i) => m === "gt" ? true : ok(r,i);
const marked = (r,m,i) => { const k=mkey(r,m); return k in MARKS ? MARKS[k] : dflt(r,m,i); };
function setMark(r,m,i,val) {
  const k=mkey(r,m);
  if (val === dflt(r,m,i)) delete MARKS[k]; else MARKS[k] = val;
  localStorage.setItem("notaccepted_marks_v1", JSON.stringify(MARKS));
}
const chkbox = (r,m,i) =>
  `<input type="checkbox" class="chk" data-fn="${escape_(r[0])}" data-m="${m}" data-i="${i}" ${marked(r,m,i)?"checked":""}>`;

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

// ---- top stats ----
const total = DATA.length;
const pctv = c => (100*c/total).toFixed(2)+"%";
const cnt  = fn => DATA.reduce((a,r)=>a+(fn(r)?1:0),0);
const S = {
  fp32gt: cnt(r=>ok(r,2)), qatgt: cnt(r=>ok(r,3)), ptqgt: cnt(r=>ok(r,4)),
  ptqfp: cnt(r=>r[4]===r[2]), qatfp: cnt(r=>r[3]===r[2]),
  agree: cnt(agree3), susp: cnt(suspect),
};
document.getElementById("sub").textContent =
  total.toLocaleString() + " disputed images · label may be wrong · exact string match";
const tile = (name,val,sub,cls="") =>
  `<div class="stat ${cls}"><div class="name">${name}</div><div class="val">${val}</div><div class="sub">${sub}</div></div>`;
document.getElementById("stats").innerHTML =
  tile("FP32 vs GT", pctv(S.fp32gt), `${S.fp32gt.toLocaleString()} match label`) +
  tile("QAT vs GT",  pctv(S.qatgt),  `${S.qatgt.toLocaleString()} match label`) +
  tile("PTQ vs GT",  pctv(S.ptqgt),  `${S.ptqgt.toLocaleString()} match label`) +
  tile("PTQ = FP32", pctv(S.ptqfp),  `${S.ptqfp.toLocaleString()} agree w/ FP32`) +
  tile("QAT = FP32", pctv(S.qatfp),  `${S.qatfp.toLocaleString()} agree w/ FP32`) +
  tile("All 3 agree", pctv(S.agree), `${S.agree.toLocaleString()} identical reads`) +
  tile("⚠ Label suspect", S.susp.toLocaleString(), "models agree, ≠ label", "suspect");

// ---- filtering ----
let view = DATA, page = 0;
const q = document.getElementById("q"), filter = document.getElementById("filter"),
      sizeSel = document.getElementById("size"), grid = document.getElementById("grid");

function pass(r, f) {
  const n = MODELS.reduce((a,[,i])=>a+(ok(r,i)?1:0),0);
  switch (f) {
    case "label_suspect": return suspect(r);
    case "ptq_eq_fp32":   return r[4]===r[2];
    case "ptq_ne_fp32":   return r[4]!==r[2];
    case "qat_eq_fp32":   return r[3]===r[2];
    case "allagree":      return agree3(r);
    case "split":         return !agree3(r);
    case "fp32_eq_gt":    return ok(r,2);
    case "fp32bad":       return !ok(r,2);
    case "qatbad":        return !ok(r,3);
    case "ptqbad":        return !ok(r,4);
    case "allbad":        return n === 0;
    case "gt_ok":         return marked(r,"gt",1);
    case "gt_bad":        return !marked(r,"gt",1);
    default:              return true;
  }
}

function refreshCounts() {
  const fc = {};
  for (const o of filter.options) fc[o.value] = 0;
  for (const r of DATA) for (const o of filter.options) if (pass(r, o.value)) fc[o.value]++;
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
               r[2].toUpperCase().includes(t) || r[3].toUpperCase().includes(t) ||
               r[4].toUpperCase().includes(t))) return false;
    return pass(r, f);
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
    return `<div class="card ${suspect(r)?'suspect':''}">
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
    }).join("") +
    (suspect(r) ? `<div class="r" style="color:var(--warn)"><span class="lbl">note</span><span class="val">all 3 models agree — label is likely wrong</span></div>` : "");
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
function applyZoom(){ mImg.style.width=""; mImg.style.transform=`scale(${zoom})`;
  mImg.style.transformOrigin="center"; zLvl.textContent = zoom+"×"; }
document.getElementById("zIn").onclick = e => { e.stopPropagation(); zoom=Math.min(12,zoom+1); applyZoom(); };
document.getElementById("zOut").onclick = e => { e.stopPropagation(); zoom=Math.max(1,zoom-1); applyZoom(); };
document.getElementById("prev").onclick = e => { e.stopPropagation(); step(-1); };
document.getElementById("next").onclick = e => { e.stopPropagation(); step(1); };
function step(d) {
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

// ---- CSV export: current filter only ----
const csvcell = v => `"${String(v).replace(/"/g,'""')}"`;
const FSLUG = {all:"all", label_suspect:"label_suspect", ptq_eq_fp32:"ptq_eq_fp32",
  ptq_ne_fp32:"ptq_ne_fp32", qat_eq_fp32:"qat_eq_fp32", allagree:"all3_models_agree",
  split:"models_disagree", fp32_eq_gt:"fp32_eq_gt", fp32bad:"fp32_ne_gt",
  qatbad:"qat_ne_gt", ptqbad:"ptq_ne_gt", allbad:"all3_ne_gt",
  gt_ok:"ground_truth_ok", gt_bad:"ground_truth_flagged_wrong"};

function exportCSV() {
  const rows = view;
  if (!rows.length) { alert("Nothing to export — this filter has no rows."); return; }
  const fval = filter.value, flabel = filter.options[filter.selectedIndex].text;

  const head = ["filename","final_results","gt_label_correct",
    "fp32_result","fp32_eq_gt","qat_result","qat_eq_gt","ptq_result","ptq_eq_gt",
    "ptq_eq_fp32","qat_eq_fp32","all3_agree","label_suspect"];
  const lines = [head.join(",")];

  let e_fp=0,e_qa=0,e_pt=0, a_ptfp=0,a_qafp=0, a3=0, susp=0, validGT=0;
  for (const r of rows) {
    const gtok = marked(r,"gt",1); if (gtok) validGT++;
    const efp=ok(r,2), eqa=ok(r,3), ept=ok(r,4);
    const ptfp=r[4]===r[2], qafp=r[3]===r[2], ag=agree3(r), su=suspect(r);
    if (efp) e_fp++; if (eqa) e_qa++; if (ept) e_pt++;
    if (ptfp) a_ptfp++; if (qafp) a_qafp++; if (ag) a3++; if (su) susp++;
    lines.push([csvcell(r[0]), csvcell(r[1]), gtok?"TRUE":"FALSE",
      csvcell(r[2]), efp?"TRUE":"FALSE", csvcell(r[3]), eqa?"TRUE":"FALSE",
      csvcell(r[4]), ept?"TRUE":"FALSE",
      ptfp?"TRUE":"FALSE", qafp?"TRUE":"FALSE", ag?"TRUE":"FALSE", su?"TRUE":"FALSE"].join(","));
  }

  const N = rows.length, pct = (a,b) => b ? (100*a/b).toFixed(2)+"%" : "0%";
  const push = (...c) => lines.push(c.map(csvcell).join(","));
  lines.push(""); lines.push(csvcell("SUMMARY"));
  push("filter", flabel.replace(/\\s*\\(\\d[\\d,]*\\)\\s*$/,""));
  const qv = q.value.trim(); if (qv) push("search", qv);
  push("images in this filter", N);
  push("labels flagged wrong (gt unticked)", N - validGT);
  lines.push("");
  push("metric vs GROUND TRUTH", "fp32", "qat", "ptq");
  push("match count", e_fp, e_qa, e_pt);
  push("accuracy", pct(e_fp,N), pct(e_qa,N), pct(e_pt,N));
  lines.push("");
  push("model-vs-model (FP32 = reference)", "count", "percent");
  push("PTQ = FP32", a_ptfp, pct(a_ptfp,N));
  push("QAT = FP32", a_qafp, pct(a_qafp,N));
  push("all 3 models agree", a3, pct(a3,N));
  push("label suspect (agree, != GT)", susp, pct(susp,N));

  const blob = new Blob(["\\ufeff"+lines.join("\\r\\n")], {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `notaccepted_${FSLUG[fval]||fval}.csv`;
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
    print(f"rows in {SRC_CSV} : {len(rows)}")
    print(f"crops found on disk : {len(rows) - len(missing)}")
    if missing:
        print(f"WARNING: {len(missing)} rows have no image file, e.g. {missing[:3]}")

    agree = sum(1 for r in rows if r[2] == r[3] == r[4])
    suspect = sum(1 for r in rows if r[2] == r[3] == r[4] and r[2] != r[1])
    print(f"  all 3 models agree : {agree}")
    print(f"  label suspect      : {suspect}")

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
