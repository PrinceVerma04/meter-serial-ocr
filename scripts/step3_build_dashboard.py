#!/usr/bin/env python3
"""
STEP 3 (optional) - build an offline dashboard of the test: every image, its ground truth and
the read of every model, side by side. No model is run here; it only reads the outputs of
step 1 and step 2 plus the images.

Inputs:
    results/2026-11-04_all_models_test/model_reads.xlsx      (step 1: reads + confidence)
    new testing data 4 nov/clean_extracted_serials.xlsx      (ground truth)
    new testing data 4 nov/clean/                            (images, embedded as-is)
    datasets/v4_train/labels.csv                             (to mark images v4 was trained on)
Output:
    results/2026-11-04_all_models_test/test_dashboard.html   (one file, works offline,
                                                              open it in any browser)

Run from the project root:
    .venv/bin/python scripts/step3_build_dashboard.py
"""
import argparse
import base64
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "2026-11-04_all_models_test")
READS = os.path.join(RES, "model_reads.xlsx")
TRUTH = os.path.join(ROOT, "new testing data 4 nov", "clean_extracted_serials.xlsx")
IMAGES = os.path.join(ROOT, "new testing data 4 nov", "clean")
V4_LABELS = os.path.join(ROOT, "datasets", "v4_train", "labels.csv")
OUT = os.path.join(RES, "test_dashboard.html")

TITLES = {"v3_fp32": "v3 FP32", "v3_ptq": "v3 PTQ", "v4_fp32": "v4 FP32", "v4_ptq": "v4 PTQ", "v4.2_fp32": "v4.2 FP32", "v4.2_ptq": "v4.2 PTQ"}
NOTES = {"v3_ptq": "deployed", "v4.2_ptq": "candidate"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp", ".webp": "image/webp"}


def cell_text(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def main():
    import openpyxl
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", default=READS)
    ap.add_argument("--truth", default=TRUTH)
    ap.add_argument("--images", default=IMAGES)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if not os.path.exists(a.reads):
        sys.exit(f"{a.reads} not found - run step 1 first")

    wb = openpyxl.load_workbook(a.reads, read_only=True)
    er = list(wb["extracted_results"].iter_rows(values_only=True))
    cf = list(wb["confidence"].iter_rows(values_only=True))
    wb.close()
    keys = [cell_text(k) for k in er[0][1:]]
    conf = {cell_text(r[0]): [float(v or 0) for v in r[1:]] for r in cf[1:] if r[0] is not None}

    wb = openpyxl.load_workbook(a.truth, read_only=True)
    tr = list(wb.active.iter_rows(values_only=True))
    wb.close()
    th = [cell_text(h) for h in tr[0]]
    i_name = th.index("image_name")
    i_gt = [i for i in range(len(th)) if i != i_name][0]
    truth = {cell_text(r[i_name]): cell_text(r[i_gt]) for r in tr[1:] if r[i_name] is not None}

    v4 = {}
    if os.path.exists(V4_LABELS):
        for r in csv.DictReader(open(V4_LABELS, encoding="utf-8")):
            v4[r["image_name"]] = 1 if r["split"] == "train" else 2

    rows, imgs, missing = [], {}, 0
    for r in er[1:]:
        if r[0] is None:
            continue
        name = cell_text(r[0])
        rows.append({"n": name, "g": truth.get(name, ""), "s": v4.get(name, 0),
                     "r": [cell_text(v) for v in r[1:]], "c": conf.get(name, [0.0] * len(keys))})
        p = os.path.join(a.images, name)
        if os.path.exists(p):
            mime = MIME.get(os.path.splitext(name)[1].lower(), "image/jpeg")
            imgs[name] = f"data:{mime};base64," + base64.b64encode(open(p, "rb").read()).decode()
        else:
            missing += 1

    meta = {"keys": keys, "titles": [TITLES.get(k, k) for k in keys], "notes": [NOTES.get(k, "") for k in keys],
            "built": time.strftime("%Y-%m-%d %H:%M"), "threshold": 0.95,
            "truth": os.path.relpath(a.truth, ROOT), "images": os.path.relpath(a.images, ROOT)}
    n = len(keys)
    html = (TEMPLATE.replace("__NCOLS__", str(n)).replace("__MCOLS__", str(2 if n == 4 else min(n, 3)))
            .replace("__TCOLS__", str(2 if n == 4 else min(n, 3)))
            .replace("__META__", json.dumps(meta))
            .replace("__ROWS__", json.dumps(rows, separators=(",", ":")))
            .replace("__IMGS__", json.dumps(imgs, separators=(",", ":"))))
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(tmp, a.out)
    print(f"images {len(rows)}  (embedded {len(imgs)}, missing {missing})   models {len(keys)}   "
          f"in v4 data {sum(1 for r in rows if r['s'])}")
    print(f"saved {os.path.relpath(a.out, ROOT)}  ({os.path.getsize(a.out) / 1e6:.1f} MB) - open it in a browser")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meter OCR Test Results</title>
<style>
:root{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#1b1f24; --muted:#5f6b7a; --line:#e2e6eb;
  --accent:#2457c5; --accent-soft:#e8eefb;
  --ok:#17803d; --ok-bg:#e7f6ec; --bad:#b42318; --bad-bg:#fdecea; --warn:#9a6700; --warn-bg:#fff4d6;
  --chip:#eef1f5; --shadow:0 1px 2px rgba(16,24,40,.06),0 1px 3px rgba(16,24,40,.08);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0f1216; --panel:#171b21; --ink:#e6e9ee; --muted:#9aa5b4; --line:#2a3039;
    --accent:#7aa2ff; --accent-soft:#1d2840;
    --ok:#5fd28a; --ok-bg:#11291b; --bad:#ff8a80; --bad-bg:#341716; --warn:#f2c14e; --warn-bg:#33290f;
    --chip:#222832; --shadow:none;
  }
}
:root[data-theme="dark"]{
  --bg:#0f1216; --panel:#171b21; --ink:#e6e9ee; --muted:#9aa5b4; --line:#2a3039;
  --accent:#7aa2ff; --accent-soft:#1d2840;
  --ok:#5fd28a; --ok-bg:#11291b; --bad:#ff8a80; --bad-bg:#341716; --warn:#f2c14e; --warn-bg:#33290f;
  --chip:#222832; --shadow:none;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:20px;margin:0}
.sub{color:var(--muted);font-size:13px;margin-top:2px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
button,select,input{font:inherit;color:inherit}
.btn{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:6px 11px;cursor:pointer}
.btn:hover{border-color:var(--accent)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
/* summary table */
.sumhead{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 14px 4px}
.sumhead h2{font-size:15px;margin:0}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{border:0;background:transparent;padding:6px 12px;cursor:pointer;border-right:1px solid var(--line)}
.seg button:last-child{border-right:0}
.seg button.on{background:var(--accent-soft);color:var(--accent);font-weight:600}
.tablewrap{overflow-x:auto;padding:4px 6px 10px}
table.sum{border-collapse:collapse;width:100%;min-width:760px}
table.sum th,table.sum td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
table.sum th{font-size:12px;color:var(--muted);font-weight:600}
table.sum th:first-child,table.sum td:first-child{text-align:left}
table.sum tr:last-child td{border-bottom:0}
table.sum td.model{font-weight:600;cursor:pointer}
table.sum td.model:hover{color:var(--accent)}
.bar{display:inline-block;width:90px;height:7px;border-radius:4px;background:var(--chip);vertical-align:middle;margin-left:8px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--ok)}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:999px;background:var(--chip);color:var(--muted);margin-left:6px;font-weight:500}
.tag.dep{background:var(--accent-soft);color:var(--accent)}
.best{color:var(--ok);font-weight:700}
.note{color:var(--muted);font-size:12px;padding:0 14px 12px}
/* filters */
.filters{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:flex-end;padding:12px 14px;margin:14px 0}
.filters label{display:flex;flex-direction:column;gap:3px;font-size:12px;color:var(--muted)}
.filters select,.filters input{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:6px 8px;min-width:150px}
.filters input{min-width:220px}
.count{margin-left:auto;font-size:13px;color:var(--muted);align-self:center}
.count b{color:var(--ink)}
/* rows */
.list{display:flex;flex-direction:column;gap:8px}
.row{display:grid;grid-template-columns:260px 1fr;gap:12px;padding:10px 12px}
.imgcell{display:flex;flex-direction:column;gap:6px;min-width:0}
.imgbox{background:var(--chip);border-radius:6px;display:flex;align-items:center;justify-content:center;height:74px;cursor:zoom-in;overflow:hidden}
.imgbox img{max-width:100%;max-height:74px;display:block}
.fname{font-size:11px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.gt{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.gt .lbl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.gt .val{font-size:16px;font-weight:700}
.tag.seen{background:var(--warn-bg);color:var(--warn)}
.reads{display:grid;grid-template-columns:repeat(__NCOLS__,minmax(0,1fr));gap:6px}
.read{border-radius:7px;padding:6px 8px;border:1px solid transparent;min-width:0}
.read.ok{background:var(--ok-bg)}
.read.bad{background:var(--bad-bg);border-color:color-mix(in srgb,var(--bad) 35%,transparent)}
.read .m{font-size:11px;color:var(--muted);display:flex;justify-content:space-between;gap:4px}
.read .t{font-size:15px;font-weight:600;word-break:break-all;margin:1px 0}
.read .t .x{color:var(--bad);text-decoration:underline;text-decoration-thickness:2px}
.read .t .empty{color:var(--muted);font-weight:400;font-style:italic}
.read .c{font-size:11px;color:var(--muted)}
.read .c.rev{color:var(--warn);font-weight:600}
.mark{font-weight:700}
.read.ok .mark{color:var(--ok)} .read.bad .mark{color:var(--bad)}
.pager{display:flex;gap:8px;justify-content:center;align-items:center;margin-top:16px;flex-wrap:wrap}
.empty-state{padding:40px;text-align:center;color:var(--muted)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:-4px 0 10px 2px}
.legend span i{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:4px}
/* modal */
.modal{position:fixed;inset:0;background:rgba(10,12,16,.72);display:none;align-items:center;justify-content:center;padding:16px;z-index:10}
.modal.on{display:flex}
.mbox{background:var(--panel);border-radius:12px;max-width:980px;width:100%;max-height:92vh;overflow:auto;padding:16px}
.mtop{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px}
.mimg{background:var(--chip);border-radius:8px;display:flex;justify-content:center;padding:12px}
.mimg img{width:100%;max-width:900px;height:auto;image-rendering:auto}
.mreads{display:grid;grid-template-columns:repeat(__MCOLS__,minmax(0,1fr));gap:8px;margin-top:12px}
.kbd{font-size:11px;color:var(--muted)}
@media (max-width:1100px){.reads{grid-template-columns:repeat(__TCOLS__,minmax(0,1fr))}}
@media (max-width:720px){
  .row{grid-template-columns:1fr}
  .reads{grid-template-columns:repeat(2,minmax(0,1fr))}
  .mreads{grid-template-columns:repeat(2,minmax(0,1fr))}
  .filters select,.filters input{min-width:0;width:100%}
  .filters label{flex:1 1 140px}
  .count{margin-left:0}
}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>Meter OCR test results: all models vs ground truth</h1>
      <div class="sub" id="sub"></div>
    </div>
    <div style="display:flex;gap:8px">
      <button class="btn" id="csvBtn">Download current view (CSV)</button>
      <button class="btn" id="themeBtn" title="Switch light / dark">Theme</button>
    </div>
  </header>

  <section class="panel">
    <div class="sumhead">
      <h2>Accuracy by model</h2>
      <div class="seg" id="setSeg">
        <button data-set="all" class="on">All images</button>
        <button data-set="fair">Not in v4 training data (fair)</button>
        <button data-set="seen">In v4 training data</button>
      </div>
    </div>
    <div class="tablewrap"><table class="sum" id="sumTable"></table></div>
    <div class="note" id="sumNote"></div>
  </section>

  <section class="panel filters">
    <label>Show
      <select id="fShow">
        <option value="all">All images</option>
        <option value="anywrong">Any model wrong</option>
        <option value="allwrong">All models wrong</option>
        <option value="allright">All models right</option>
        <option value="disagree">Models give different reads</option>
        <option value="only3">Only v3 PTQ right (candidate PTQ wrong)</option>
        <option value="only4">Only candidate PTQ right (v3 PTQ wrong)</option>
        <option value="both34">v3 PTQ and candidate PTQ both wrong</option>
        <option value="longer">A read is longer than ground truth</option>
        <option value="accwrong">Accepted (&ge;0.95) but wrong, any model</option>
      </select>
    </label>
    <label>Model is wrong
      <select id="fModel"><option value="">(any)</option></select>
    </label>
    <label>Ground truth length
      <select id="fLen"><option value="">(any)</option></select>
    </label>
    <label>Sort
      <select id="fSort">
        <option value="name">File name</option>
        <option value="wrong">Most models wrong first</option>
        <option value="conf">Lowest confidence first</option>
      </select>
    </label>
    <label>Search (file, ground truth or read)
      <input id="fQ" type="search" placeholder="e.g. 41725713">
    </label>
    <label>Per page
      <select id="fPer"><option>25</option><option selected>50</option><option>100</option><option>250</option></select>
    </label>
    <div class="count" id="count"></div>
  </section>

  <div class="legend">
    <span><i style="background:var(--ok-bg);border:1px solid var(--ok)"></i>matches ground truth</span>
    <span><i style="background:var(--bad-bg);border:1px solid var(--bad)"></i>wrong (red underline = characters not in ground truth)</span>
    <span><b style="color:var(--warn)">review</b> = confidence below 0.95</span>
    <span><span class="tag seen" style="margin:0">in v4 train</span> = v4 was trained on this image</span>
    <span>click an image to enlarge, then use the left/right arrow keys</span>
  </div>

  <div class="list" id="list"></div>
  <div class="pager" id="pager"></div>
</div>

<div class="modal" id="modal">
  <div class="mbox">
    <div class="mtop">
      <div><div class="gt" id="mGt"></div><div class="fname mono" id="mName"></div></div>
      <div style="display:flex;gap:6px;align-items:center">
        <span class="kbd" id="mPos"></span>
        <button class="btn" id="mPrev">&larr;</button><button class="btn" id="mNext">&rarr;</button>
        <button class="btn" id="mClose">Close</button>
      </div>
    </div>
    <div class="mimg"><img id="mImg" alt=""></div>
    <div class="mreads" id="mReads"></div>
  </div>
</div>

<script>
const META = __META__;
const ROWS = __ROWS__;
const IMGS = __IMGS__;
const K = META.keys, T = META.titles, NOTE = META.notes, TH = META.threshold;
const norm = s => (s || "").toUpperCase().replace(/\s+/g, "");
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const i3 = K.indexOf("v3_ptq"), i4 = K.indexOf("v4.2_ptq") >= 0 ? K.indexOf("v4.2_ptq") : K.indexOf("v4_ptq");
const CAND = i4 >= 0 ? T[i4] : "candidate";

ROWS.forEach(r => {
  const g = norm(r.g);
  r.ok = r.r.map(t => !!t && norm(t) === g && g !== "");
  r.nw = r.ok.filter(x => !x).length;
  r.minc = Math.min(...r.r.map((t, j) => t ? r.c[j] : 0));
  r.dis = new Set(r.r.map(norm)).size > 1;
  r.longer = r.r.some(t => norm(t).length > g.length);
  r.accw = r.r.some((t, j) => t && r.c[j] >= TH && !r.ok[j]);
  r.hay = (r.n + " " + r.g + " " + r.r.join(" ")).toUpperCase();
});

// ---------- marks characters of `read` that are not part of the longest common subsequence with gt
function diffHtml(read, gt) {
  const a = norm(read), b = norm(gt);
  if (!a) return '<span class="empty">(nothing read)</span>';
  if (a === b) return esc(a);
  const n = a.length, m = b.length, L = Array.from({length: n + 1}, () => new Int16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  let i = 0, j = 0, out = "";
  while (i < n) {
    if (j < m && a[i] === b[j]) { out += esc(a[i]); i++; j++; }
    else if (j < m && L[i][j + 1] >= L[i + 1][j]) { j++; }
    else { out += '<span class="x">' + esc(a[i]) + "</span>"; i++; }
  }
  return out;
}

// ---------- state
const st = {set: "all", show: "all", model: "", len: "", sort: "name", q: "", per: 50, page: 0};
let view = [];
const inSet = r => st.set === "all" || (st.set === "fair" ? r.s === 0 : r.s > 0);

function filterRows() {
  const q = st.q.trim().toUpperCase().replace(/\s+/g, " ");
  const mi = st.model === "" ? -1 : +st.model;
  view = ROWS.filter(r => {
    if (!inSet(r)) return false;
    if (mi >= 0 && r.ok[mi]) return false;
    if (st.len !== "" && norm(r.g).length !== +st.len) return false;
    switch (st.show) {
      case "anywrong": if (r.nw === 0) return false; break;
      case "allwrong": if (r.nw !== K.length) return false; break;
      case "allright": if (r.nw !== 0) return false; break;
      case "disagree": if (!r.dis) return false; break;
      case "only3": if (!(r.ok[i3] && !r.ok[i4])) return false; break;
      case "only4": if (!(r.ok[i4] && !r.ok[i3])) return false; break;
      case "both34": if (r.ok[i3] || r.ok[i4]) return false; break;
      case "longer": if (!r.longer) return false; break;
      case "accwrong": if (!r.accw) return false; break;
    }
    if (q && !q.split(" ").every(w => r.hay.includes(w))) return false;
    return true;
  });
  if (st.sort === "wrong") view.sort((a, b) => b.nw - a.nw || a.n.localeCompare(b.n));
  else if (st.sort === "conf") view.sort((a, b) => a.minc - b.minc || a.n.localeCompare(b.n));
  else view.sort((a, b) => a.n.localeCompare(b.n));
}

// ---------- summary table
function renderSummary() {
  const rows = ROWS.filter(inSet), n = rows.length;
  const stats = K.map((k, j) => {
    const right = rows.filter(r => r.ok[j]).length;
    const acc = rows.filter(r => r.r[j] && r.c[j] >= TH);
    const accR = acc.filter(r => r.ok[j]).length;
    const blank = rows.filter(r => !r.r[j]).length;
    return {right, wrong: n - right, acc: acc.length, accR, accW: acc.length - accR, review: n - acc.length - blank, blank};
  });
  const best = Math.max(...stats.map(s => s.right));
  const pct = (a, b) => b ? (100 * a / b).toFixed(2) + "%" : "-";
  let h = "<tr><th>Model</th><th>Right</th><th>Wrong</th><th>Accuracy</th><th>Accepted (&ge;0.95)</th>" +
          "<th>Accepted but wrong</th><th>Accepted precision</th><th>Review (&lt;0.95)</th><th>Blank</th></tr>";
  stats.forEach((s, j) => {
    const p = n ? 100 * s.right / n : 0;
    h += `<tr><td class="model" data-j="${j}" title="Show images this model gets wrong">${esc(T[j])}` +
         (NOTE[j] ? `<span class="tag ${NOTE[j] === "deployed" ? "dep" : ""}">${NOTE[j]}</span>` : "") + `</td>` +
         `<td class="${s.right === best ? "best" : ""}">${s.right.toLocaleString()}</td><td>${s.wrong.toLocaleString()}</td>` +
         `<td><b>${pct(s.right, n)}</b><span class="bar"><i style="width:${p}%"></i></span></td>` +
         `<td>${s.acc.toLocaleString()}</td><td>${s.accW.toLocaleString()}</td><td>${pct(s.accR, s.acc)}</td>` +
         `<td>${s.review.toLocaleString()}</td><td>${s.blank.toLocaleString()}</td></tr>`;
  });
  document.getElementById("sumTable").innerHTML = h;
  const both = rows.filter(r => r.ok[i3] && r.ok[i4]).length, o3 = rows.filter(r => r.ok[i3] && !r.ok[i4]).length,
        o4 = rows.filter(r => !r.ok[i3] && r.ok[i4]).length, none = n - both - o3 - o4;
  const setName = {all: "all images", fair: "images v4 never saw (fair comparison)", seen: "images that are in v4's training data"}[st.set];
  document.getElementById("sumNote").innerHTML =
    `${n.toLocaleString()} ${setName}. Right + wrong = ${n.toLocaleString()}; accepted + review + blank = ${n.toLocaleString()}. ` +
    `v3 PTQ vs ${CAND}: both right ${both}, only v3 ${o3}, only ${CAND} ${o4}, both wrong ${none}. ` +
    `A read is right when it equals the ground truth after upper-casing and removing spaces. Click a model name to list its wrong images.`;
  document.querySelectorAll("td.model").forEach(td => td.onclick = () => {
    st.model = td.dataset.j; st.show = "all"; st.page = 0;
    document.getElementById("fModel").value = st.model; document.getElementById("fShow").value = "all";
    update(); document.querySelector(".filters").scrollIntoView({behavior: "smooth"});
  });
}

function readCell(r, j) {
  const t = r.r[j], c = r.c[j], ok = r.ok[j];
  const conf = t ? `<div class="c ${c < TH ? "rev" : ""}">${c.toFixed(4)}${c < TH ? " review" : ""}</div>` : `<div class="c">blank</div>`;
  return `<div class="read ${ok ? "ok" : "bad"}"><div class="m"><span>${esc(T[j])}</span><span class="mark">${ok ? "&#10003;" : "&#10007;"}</span></div>` +
         `<div class="t mono">${diffHtml(t, r.g)}</div>${conf}</div>`;
}
function seenTag(r) { return r.s === 1 ? '<span class="tag seen">in v4 train</span>' : r.s === 2 ? '<span class="tag seen">in v4 val/test</span>' : ""; }

function renderList() {
  const pages = Math.max(1, Math.ceil(view.length / st.per));
  st.page = Math.min(st.page, pages - 1);
  const start = st.page * st.per, part = view.slice(start, start + st.per);
  document.getElementById("count").innerHTML = `<b>${view.length.toLocaleString()}</b> of ${ROWS.filter(inSet).length.toLocaleString()} images`;
  const list = document.getElementById("list");
  if (!part.length) { list.innerHTML = '<div class="panel empty-state">No images match these filters.</div>'; }
  else list.innerHTML = part.map((r, i) =>
    `<div class="panel row"><div class="imgcell">` +
      `<div class="imgbox" data-i="${start + i}"><img loading="lazy" alt="${esc(r.n)}" src="${IMGS[r.n] || ""}"></div>` +
      `<div class="gt"><span class="lbl">Ground truth</span><span class="val mono">${esc(r.g) || "<i>none</i>"}</span>${seenTag(r)}</div>` +
      `<div class="fname mono" title="${esc(r.n)}">${esc(r.n)}</div></div>` +
    `<div class="reads">${K.map((k, j) => readCell(r, j)).join("")}</div></div>`).join("");
  list.querySelectorAll(".imgbox").forEach(b => b.onclick = () => openModal(+b.dataset.i));
  const pg = document.getElementById("pager");
  pg.innerHTML = pages > 1 ?
    `<button class="btn" id="pPrev" ${st.page ? "" : "disabled"}>&larr; Previous</button>` +
    `<span>Page ${st.page + 1} of ${pages}</span>` +
    `<button class="btn" id="pNext" ${st.page < pages - 1 ? "" : "disabled"}>Next &rarr;</button>` : "";
  if (pages > 1) {
    document.getElementById("pPrev").onclick = () => { st.page--; renderList(); window.scrollTo({top: document.querySelector(".filters").offsetTop - 8}); };
    document.getElementById("pNext").onclick = () => { st.page++; renderList(); window.scrollTo({top: document.querySelector(".filters").offsetTop - 8}); };
  }
}

// ---------- modal
let mIdx = -1;
function openModal(i) {
  if (i < 0 || i >= view.length) return;
  mIdx = i; const r = view[i];
  document.getElementById("mImg").src = IMGS[r.n] || "";
  document.getElementById("mName").textContent = r.n;
  document.getElementById("mGt").innerHTML = `<span class="lbl">Ground truth</span><span class="val mono" style="font-size:22px">${esc(r.g) || "<i>none</i>"}</span>${seenTag(r)}`;
  document.getElementById("mReads").innerHTML = K.map((k, j) => readCell(r, j)).join("");
  document.getElementById("mPos").textContent = `${i + 1} / ${view.length}`;
  document.getElementById("modal").classList.add("on");
}
function closeModal() { document.getElementById("modal").classList.remove("on"); mIdx = -1; }
document.getElementById("mClose").onclick = closeModal;
document.getElementById("mPrev").onclick = () => openModal(mIdx - 1);
document.getElementById("mNext").onclick = () => openModal(mIdx + 1);
document.getElementById("modal").onclick = e => { if (e.target.id === "modal") closeModal(); };
document.addEventListener("keydown", e => {
  if (mIdx < 0) return;
  if (e.key === "Escape") closeModal();
  else if (e.key === "ArrowLeft") openModal(mIdx - 1);
  else if (e.key === "ArrowRight") openModal(mIdx + 1);
});

// ---------- CSV of current view
document.getElementById("csvBtn").onclick = () => {
  const q = v => /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  const head = ["image_name", "ground_truth", "in_v4_data"];
  K.forEach(k => head.push(k + "_result", k + "_confidence", k + "_correct"));
  const lines = [head.join(",")];
  view.forEach(r => {
    const v = [r.n, r.g, ["no", "train", "val/test"][r.s]];
    K.forEach((k, j) => v.push(r.r[j], r.r[j] ? r.c[j].toFixed(4) : "0", r.ok[j] ? "yes" : "no"));
    lines.push(v.map(x => q(String(x))).join(","));
  });
  const blob = new Blob([lines.join("\n") + "\n"], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "test_view_" + view.length + "_images.csv";
  document.body.appendChild(a); a.click(); a.remove();
};

// ---------- theme
function setTheme(t) { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; try { t ? localStorage.setItem("theme", t) : localStorage.removeItem("theme"); } catch (e) {} }
try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) {}
document.getElementById("themeBtn").onclick = () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  setTheme(dark ? "light" : "dark");
};

// ---------- wiring
function update() { filterRows(); renderSummary(); renderList(); }
const fModel = document.getElementById("fModel");
K.forEach((k, j) => fModel.insertAdjacentHTML("beforeend", `<option value="${j}">${esc(T[j])}</option>`));
const lens = [...new Set(ROWS.map(r => norm(r.g).length))].sort((a, b) => a - b);
const fLen = document.getElementById("fLen");
lens.forEach(l => fLen.insertAdjacentHTML("beforeend", `<option value="${l}">${l} characters (${ROWS.filter(r => norm(r.g).length === l).length})</option>`));
const bind = (id, key, num) => document.getElementById(id).addEventListener(id === "fQ" ? "input" : "change", e => {
  st[key] = num ? +e.target.value : e.target.value; st.page = 0; update();
});
bind("fShow", "show"); bind("fModel", "model"); bind("fLen", "len"); bind("fSort", "sort"); bind("fQ", "q"); bind("fPer", "per", true);
document.querySelectorAll("#setSeg button").forEach(b => b.onclick = () => {
  document.querySelectorAll("#setSeg button").forEach(x => x.classList.toggle("on", x === b));
  st.set = b.dataset.set; st.page = 0; update();
});
const nSeen = ROWS.filter(r => r.s).length;
document.getElementById("sub").textContent =
  `${ROWS.length.toLocaleString()} images from ${META.images} · ground truth: ${META.truth} · ` +
  `${nSeen.toLocaleString()} of them are in v4's training data · built ${META.built}`;
update();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
