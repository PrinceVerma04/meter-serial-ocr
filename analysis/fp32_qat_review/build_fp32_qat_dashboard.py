#!/usr/bin/env python3
"""
FP32-vs-QAT dashboard for the 1,055 images in accuracy_counted_incorrect.csv.

Two fields per card: fp32_result and qat_result. Two filters:
  - FP32 = QAT  (matching)      -> both readings highlighted green
  - FP32 != QAT (not matching)  -> both readings highlighted red

NOTE: this file is the "QAT differs from FP32" review pile, so FP32 and QAT never
match here -> FP32 = QAT is 0, FP32 != QAT is 1,055. The two values are shown side
by side on every card so this can be verified by eye. Built exactly as requested.

Images are embedded (base64) so the page loads offline from any location.

Run:
    python3 fp32_qat_review/build_fp32_qat_dashboard.py
"""
import os, csv, json, base64

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

SRC_CSV = "accuracy_counted_incorrect.csv"
IMG_DIR = "../../datasets/final_test/serialno_testing/images"
OUT = "dashboard.html"


def data_uri(path):
    try:
        with open(path, "rb") as fh:
            return "data:image/jpeg;base64," + base64.b64encode(fh.read()).decode("ascii")
    except OSError:
        return ""


def load_rows():
    """[filename, fp32, qat, dataURI] from the CSV (skip header + summary block)."""
    rows = []
    with open(SRC_CSV, newline="", encoding="utf-8-sig") as fh:
        for r in csv.reader(fh):
            if len(r) < 5 or r[0] in ("", "filename", "SUMMARY"):
                continue
            # csv cols: 0 filename, 1 final_results, 2 fp32, 3 qat, 4 ptq
            rows.append([r[0], r[2], r[3], data_uri(os.path.join(IMG_DIR, r[0]))])
    return rows


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FP32 vs QAT — review</title>
<style>
  :root { --bg:#0f1216; --panel:#171b21; --panel2:#1e242c; --line:#2a323d;
    --text:#e6edf3; --muted:#8b98a5; --ok:#3fb950; --bad:#f85149; --accent:#58a6ff;
    --okbg:rgba(63,185,80,.14); --badbg:rgba(248,81,73,.14); }
  @media (prefers-color-scheme: light) {
    :root { --bg:#f4f6f8; --panel:#fff; --panel2:#f0f3f6; --line:#d7dee6;
      --text:#1a2029; --muted:#5c6b7a; --ok:#1a7f37; --bad:#cf222e; --accent:#0969da;
      --okbg:rgba(26,127,55,.12); --badbg:rgba(207,34,46,.12); } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  header { position:sticky; top:0; z-index:20; background:var(--panel);
           border-bottom:1px solid var(--line); padding:14px 18px; }
  h1 { margin:0 0 8px; font-size:16px; font-weight:600; }
  h1 small { color:var(--muted); font-weight:400; margin-left:8px; }
  .stats { display:flex; gap:10px; flex-wrap:wrap; }
  .stat { background:var(--panel2); border:1px solid var(--line); border-radius:8px; padding:8px 13px; min-width:130px; }
  .stat .name { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; }
  .stat .val { font-size:20px; font-weight:650; margin-top:2px; }
  .stat.ok .val { color:var(--ok); } .stat.bad .val { color:var(--bad); }
  .controls { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:12px; }
  input, select, button { background:var(--panel2); color:var(--text); border:1px solid var(--line);
      border-radius:6px; padding:6px 10px; font:inherit; }
  input:focus, select:focus { outline:2px solid var(--accent); outline-offset:-1px; }
  button { cursor:pointer; } button:hover:not(:disabled){ border-color:var(--accent); }
  button:disabled { opacity:.4; cursor:default; }
  #count { color:var(--muted); margin-left:auto; font-variant-numeric:tabular-nums; }

  .grid { display:grid; gap:12px; padding:16px 18px; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:9px; overflow:hidden; display:flex; flex-direction:column; }
  .card.match { border-color:var(--ok); } .card.nomatch { border-color:var(--bad); }
  .shot { background:#fff; display:flex; align-items:center; justify-content:center;
          min-height:86px; cursor:zoom-in; border-bottom:1px solid var(--line); padding:6px; }
  .shot img { max-width:100%; max-height:120px; display:block; }
  .rows { padding:8px 10px 10px; }
  .fn { display:flex; align-items:center; gap:6px; margin-bottom:8px; }
  .fn .name { font:11px ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted);
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis; user-select:all; flex:1; }
  .copybtn { flex:none; cursor:pointer; border:1px solid var(--line); background:var(--panel2);
        color:var(--muted); border-radius:5px; font-size:11px; padding:1px 7px; line-height:1.5; }
  .copybtn:hover { border-color:var(--accent); color:var(--accent); }
  .copybtn.done { color:var(--ok); border-color:var(--ok); }
  .r { display:flex; align-items:center; gap:8px; padding:5px 7px; margin-top:5px; border-radius:6px; }
  .r .lbl { font-size:10px; color:var(--muted); width:38px; flex:none; text-transform:uppercase; letter-spacing:.5px; }
  .r .val { font:14px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace; word-break:break-all; font-weight:700; }
  .r.hl-ok  { background:var(--okbg); }  .r.hl-ok .val  { color:var(--ok); }
  .r.hl-bad { background:var(--badbg); } .r.hl-bad .val { color:var(--bad); }
  .status { margin-left:auto; font-size:11px; font-weight:700; flex:none; }
  .status.ok { color:var(--ok); } .status.bad { color:var(--bad); }
  .empty { color:var(--muted); font-style:italic; font-weight:400; }

  .pager { display:flex; gap:6px; justify-content:center; align-items:center; padding:6px 18px 28px; flex-wrap:wrap; }
  .pager input { width:72px; text-align:center; }
  #modal { position:fixed; inset:0; z-index:50; background:rgba(0,0,0,.82); display:none;
           align-items:center; justify-content:center; flex-direction:column; gap:14px; padding:24px; }
  #modal.on { display:flex; }
  #modal img { background:#fff; padding:8px; border-radius:6px; max-width:94vw; max-height:58vh; image-rendering:pixelated; }
  #mInfo { background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:12px 16px; max-width:min(680px,94vw); width:100%; }
  #mNav { display:flex; gap:10px; align-items:center; }
  .zoomctl { display:flex; gap:6px; align-items:center; color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>FP32 vs QAT <small id="sub"></small></h1>
  <div class="stats" id="stats"></div>
  <div class="controls">
    <input id="q" type="search" placeholder="Search filename or serial…" size="26">
    <select id="filter">
      <option value="all">All images</option>
      <option value="match">FP32 = QAT (matching)</option>
      <option value="nomatch">FP32 ≠ QAT (not matching)</option>
    </select>
    <select id="size"><option>100</option><option selected>200</option><option>500</option></select>
    <button id="export" style="display:none" title="Download a CSV of the CURRENT filter only">⭳ Export CSV (this filter)</button>
    <span id="count"></span>
  </div>
</header>

<div class="grid" id="grid"></div>
<div class="pager" id="pager"></div>

<div id="modal">
  <img id="mImg" alt="">
  <div class="zoomctl"><button id="zOut">−</button><span id="zLvl">4×</span><button id="zIn">+</button>
    <span style="margin-left:10px">click outside or Esc to close · ← → to move</span></div>
  <div id="mInfo"></div>
  <div id="mNav"><button id="prev">← Prev</button><span id="mPos"></span><button id="next">Next →</button></div>
</div>

<script>
const DATA = __DATA__;            // [filename, fp32, qat, dataURI]
const src = r => r[3];           // embedded base64 image
const match = r => r[1] === r[2];              // FP32 == QAT (exact string match)
const escape_ = s => s.replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

const total = DATA.length;
const nMatch = DATA.reduce((a,r)=>a+(match(r)?1:0),0);
document.getElementById("sub").textContent = total.toLocaleString() + " images · FP32 vs QAT · exact string match";
document.getElementById("stats").innerHTML =
  `<div class="stat"><div class="name">Total</div><div class="val">${total.toLocaleString()}</div><div class="sub">images</div></div>` +
  `<div class="stat ok"><div class="name">FP32 = QAT</div><div class="val">${nMatch.toLocaleString()}</div><div class="sub">matched</div></div>` +
  `<div class="stat bad"><div class="name">FP32 ≠ QAT</div><div class="val">${(total-nMatch).toLocaleString()}</div><div class="sub">not matched</div></div>`;

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

let view = DATA, page = 0;
const q = document.getElementById("q"), filter = document.getElementById("filter"),
      sizeSel = document.getElementById("size"), grid = document.getElementById("grid");

function pass(r, f) {
  if (f === "match") return match(r);
  if (f === "nomatch") return !match(r);
  return true;
}
function refreshCounts() {
  const fc = {}; for (const o of filter.options) fc[o.value] = 0;
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
    if (t && !(r[0].toUpperCase().includes(t) || r[1].toUpperCase().includes(t) || r[2].toUpperCase().includes(t))) return false;
    return pass(r, f);
  });
  page = 0; render();
}

function valRow(lbl, txt, good, tag) {
  const v = txt ? escape_(txt) : '<span class="empty">(no read)</span>';
  return `<div class="r ${good?'hl-ok':'hl-bad'}"><span class="lbl">${lbl}</span>
    <span class="val">${v}</span><span class="status ${good?'ok':'bad'}">${tag}</span></div>`;
}
function cardBody(r) {
  const good = match(r);
  return valRow("fp32", r[1], good, good?"= QAT":"≠ QAT") +
         valRow("qat",  r[2], good, good?"= FP32":"≠ FP32");
}

function render() {
  const per = +sizeSel.value, pages = Math.max(1, Math.ceil(view.length/per));
  page = Math.min(page, pages-1);
  const from = page*per, slice = view.slice(from, from+per);
  document.getElementById("count").textContent = view.length
    ? `showing ${(from+1).toLocaleString()}–${(from+slice.length).toLocaleString()} of ${view.length.toLocaleString()}`
    : "no matches";

  grid.innerHTML = slice.map((r,k) => `<div class="card ${match(r)?'match':'nomatch'}">
      <div class="shot" data-i="${from+k}"><img loading="lazy" src="${src(r)}" alt=""></div>
      <div class="rows">${fnLine(r)}${cardBody(r)}</div></div>`).join("");

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
q.oninput = apply; filter.onchange = apply; sizeSel.onchange = () => { page=0; render(); };

const modal=document.getElementById("modal"), mImg=document.getElementById("mImg"),
      mInfo=document.getElementById("mInfo"), mPos=document.getElementById("mPos"), zLvl=document.getElementById("zLvl");
let cur=-1, zoom=4;
grid.addEventListener("click", e => {
  const b = e.target.closest(".copybtn"); if (b) { copyName(b); return; }
  const s = e.target.closest(".shot"); if (s) open(+s.dataset.i);
});
function open(i) {
  if (i<0 || i>=view.length) return;
  cur=i; const r=view[i]; mImg.src=src(r); applyZoom();
  mInfo.innerHTML = fnLine(r) + cardBody(r);
  mPos.textContent = `${(i+1).toLocaleString()} / ${view.length.toLocaleString()}`;
  modal.classList.add("on");
}
mInfo.addEventListener("click", e => { const b = e.target.closest(".copybtn"); if (b) copyName(b); });
function applyZoom(){ mImg.style.width=""; mImg.style.transform=`scale(${zoom})`; mImg.style.transformOrigin="center"; zLvl.textContent = zoom+"×"; }
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

// ---- CSV export: current filter only ----
const csvcell = v => `"${String(v).replace(/"/g,'""')}"`;
const FSLUG = {all:"all", match:"fp32_eq_qat", nomatch:"fp32_ne_qat"};
function exportCSV() {
  const rows = view;
  if (!rows.length) { alert("Nothing to export — this filter has no rows."); return; }
  const fval = filter.value, flabel = filter.options[filter.selectedIndex].text;
  const lines = [["filename","fp32_result","qat_result","fp32_eq_qat"].join(",")];
  let m=0;
  for (const r of rows) { const g=match(r); if(g)m++;
    lines.push([csvcell(r[0]),csvcell(r[1]),csvcell(r[2]),g?"TRUE":"FALSE"].join(",")); }
  const N=rows.length;
  lines.push(""); lines.push(csvcell("SUMMARY"));
  lines.push([csvcell("filter"),csvcell(flabel.replace(/\\s*\\(\\d[\\d,]*\\)\\s*$/,""))].join(","));
  lines.push([csvcell("images in filter"),N].join(","));
  lines.push([csvcell("FP32 = QAT"),m].join(","));
  lines.push([csvcell("FP32 != QAT"),N-m].join(","));
  const blob = new Blob(["\\ufeff"+lines.join("\\r\\n")], {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = `fp32_qat_${FSLUG[fval]||fval}.csv`; a.click(); URL.revokeObjectURL(a.href);
}
document.getElementById("export").onclick = exportCSV;

apply();
</script>
</body>
</html>
"""


def main():
    rows = load_rows()
    embedded = sum(1 for r in rows if r[3])
    n_match = sum(1 for r in rows if r[1] == r[2])
    print(f"rows in {SRC_CSV} : {len(rows)}")
    print(f"crops embedded     : {embedded}")
    print(f"  FP32 == QAT (matched)     : {n_match}")
    print(f"  FP32 != QAT (not matched) : {len(rows) - n_match}")

    page = TEMPLATE.replace("__DATA__", json.dumps(rows, separators=(",", ":")))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"\nwrote {os.path.join(HERE, OUT)}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"open it with:  xdg-open '{os.path.join(HERE, OUT)}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
