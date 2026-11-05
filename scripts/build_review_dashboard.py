#!/usr/bin/env python3
"""
Offline dashboard: every crop with each model's read, filtered by what the models ACCEPT at a
confidence cut-off you move yourself (production gate 0.95) - accepted by every model, by none,
by some - by whether the models read the SAME serial (whatever the confidence), and by whether
the reads match the correct serial.

Only crops still in the review folder are shown; images in all_models_same/images (separated out by
scripts/export_all_same.py) are left out as well.

Reads only saved files (no model runs):
  results/2026-11-05_review_21425/model_reads.xlsx                 (step 1: each model's read + confidence)
  new testing data 4 nov/5 nov testing data/review_ground_truth.xlsx  (image_name, correct_serial)
Writes:
  results/2026-11-05_review_21425/review_dashboard.html   - open it in a browser (double-click)

The page loads the crops from the review folder by relative path, so keep the html in the results
folder (images are not copied into it and nothing leaves the PC).

    .venv/bin/python scripts/build_review_dashboard.py
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "2026-11-05_review_21425")
REVIEW = os.path.join(ROOT, "new testing data 4 nov", "5 nov testing data")
TITLES = {"v3_fp32": "v3 FP32", "v3_ptq": "v3 INT8 (deployed)", "v4_fp32": "v4 FP32", "v4_ptq": "v4 INT8 (candidate)"}


def load(reads, gt_path):
    import openpyxl
    wb = openpyxl.load_workbook(reads, read_only=True)
    er = list(wb["extracted_results"].iter_rows(values_only=True))
    cf = {r[0]: r[1:] for r in list(wb["confidence"].iter_rows(values_only=True))[1:] if r[0]}
    wb.close()
    keys = list(er[0][1:])
    ws = openpyxl.load_workbook(gt_path, read_only=True).worksheets[0]
    hdr = [str(h or "").strip() for h in next(ws.iter_rows(max_row=1, values_only=True))]
    if "image_name" not in hdr or "correct_serial" not in hdr:
        sys.exit(f"!! {gt_path} needs columns image_name and correct_serial, has {hdr}")
    i_n, i_s = hdr.index("image_name"), hdr.index("correct_serial")
    gt = {r[i_n]: str(r[i_s] if r[i_s] is not None else "").strip()
          for r in ws.iter_rows(min_row=2, values_only=True) if r[i_n]}
    rows = [r for r in er[1:] if r[0]]
    for r in rows:   # images deleted from the review folder have no ground truth any more
        gt.setdefault(r[0], "")
    data = {"keys": keys, "titles": [TITLES.get(k, k) for k in keys], "names": [], "gt": [],
            "reads": [[] for _ in keys], "conf": [[] for _ in keys]}
    for r in sorted(rows, key=lambda r: r[0]):
        data["names"].append(r[0])
        data["gt"].append(gt[r[0]])
        for i in range(len(keys)):
            data["reads"][i].append(str(r[1 + i] or "").strip())
            data["conf"][i].append(round(float(cf[r[0]][i] or 0.0), 4))
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reads", default=os.path.join(OUT_DIR, "model_reads.xlsx"))
    ap.add_argument("--gt", default=os.path.join(REVIEW, "review_ground_truth.xlsx"))
    ap.add_argument("--images", default=os.path.join(REVIEW, "review"))
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "review_dashboard.html"))
    ap.add_argument("--exclude", default=os.path.join(REVIEW, "all_models_same", "images"),
                    help='folder of images to leave out (default: the all-models-same export); "" = none')
    a = ap.parse_args()
    data = load(a.reads, a.gt)
    # show only crops that are still in the review folder, minus the ones separated out
    present = set(os.listdir(a.images))
    drop = set(os.listdir(a.exclude)) if a.exclude and os.path.isdir(a.exclude) else set()
    keep = [i for i, n in enumerate(data["names"]) if n in present and n not in drop]
    data["excluded"] = sum(n in drop for n in data["names"])
    data["gone"] = sum(n not in present and n not in drop for n in data["names"])
    data["excludedFrom"] = os.path.relpath(os.path.abspath(a.exclude), ROOT) if a.exclude else ""
    data["names"] = [data["names"][i] for i in keep]
    data["gt"] = [data["gt"][i] for i in keep]
    data["reads"] = [[col[i] for i in keep] for col in data["reads"]]
    data["conf"] = [[col[i] for i in keep] for col in data["conf"]]
    data["imgBase"] = os.path.relpath(os.path.abspath(a.images), os.path.dirname(os.path.abspath(a.out))).replace(os.sep, "/")
    data["imgAbs"] = os.path.abspath(a.images).replace(os.sep, "/")
    data["made"] = time.strftime("%Y-%m-%d %H:%M")
    data["source"] = os.path.relpath(os.path.abspath(a.images), ROOT)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = PAGE.replace("/*__DATA__*/null", blob)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html)
    os.replace(tmp, a.out)
    print(f"saved {os.path.relpath(a.out, ROOT)}  ({len(data['names'])} images, {os.path.getsize(a.out) / 1e6:.1f} MB)")
    print("open it in a browser (double-click); keep it in this folder so the crops load")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Dashboard</title>
<style>
:root{--bg:#f6f5f1;--panel:#ffffff;--ink:#17212b;--muted:#5d6873;--line:#dde1e4;--accent:#b8650f;
--ok:#1d7a4f;--okbg:#e3f3ea;--bad:#b3261e;--badbg:#fbe6e4;--warnbg:#fdf1dc;--chip:#eef1f3;--imgbg:#e3e8eb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#12171c;--panel:#1b2229;--ink:#e6eaee;--muted:#9aa6b1;
--line:#2c3640;--accent:#f0a24a;--ok:#5ccf93;--okbg:#173327;--bad:#ff8a80;--badbg:#3a1d1b;--warnbg:#3a2c16;--chip:#25303a;--imgbg:#2a333c}}
:root[data-theme="dark"]{--bg:#12171c;--panel:#1b2229;--ink:#e6eaee;--muted:#9aa6b1;--line:#2c3640;--accent:#f0a24a;
--ok:#5ccf93;--okbg:#173327;--bad:#ff8a80;--badbg:#3a1d1b;--warnbg:#3a2c16;--chip:#25303a;--imgbg:#2a333c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
.mono{font-family:ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace}
header{padding:20px 24px 8px}
h1{margin:0;font-size:22px}
.sub{color:var(--muted);font-size:13px;margin-top:4px}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 24px;display:flex;flex-wrap:wrap;gap:14px 22px;align-items:flex-end}
.ctl{display:flex;flex-direction:column;gap:4px}
.ctl>label,.lab{font-size:12px;color:var(--muted);font-weight:600;letter-spacing:.3px;text-transform:uppercase}
select,input[type=search],input[type=number],button{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 10px}
button{cursor:pointer}button:hover{border-color:var(--accent)}button:disabled{opacity:.45;cursor:default}
.slider{display:flex;align-items:center;gap:10px}
.slider input[type=range]{width:260px;accent-color:var(--accent)}
.slider input[type=number]{width:84px}
.models{display:flex;flex-wrap:wrap;gap:6px}
.models label{display:flex;align-items:center;gap:5px;background:var(--chip);border-radius:999px;padding:4px 10px;font-size:13px;cursor:pointer}
.hint{font-size:12px;color:var(--muted);max-width:420px}
main{padding:16px 24px 48px}
.stats{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin-bottom:16px;overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:640px}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-size:12px;color:var(--muted);font-weight:600}
td.bad{color:var(--bad);font-weight:600}td.ok{color:var(--ok);font-weight:600}
.count{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:10px;margin:4px 0 12px}
.count b{font-size:18px}
.pager{display:flex;gap:8px;align-items:center}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;display:flex;flex-direction:column;gap:8px;min-width:0}
.card a.img{display:flex;align-items:center;justify-content:center;background:var(--imgbg);border-radius:8px;height:92px;overflow:hidden}
.card img{max-width:100%;max-height:92px;object-fit:contain}
.name{font-size:11px;color:var(--muted);word-break:break-all}
.gt{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.gt .v{font-size:20px;font-weight:700;letter-spacing:.5px}
.rows{display:flex;flex-direction:column;gap:4px}
.r{display:grid;grid-template-columns:118px 1fr 92px;gap:8px;align-items:center;font-size:13px;padding:3px 6px;border-radius:6px}
.r.right{background:var(--okbg)}.r.wrong{background:var(--badbg)}
.r .m{color:var(--muted);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.r .t{font-size:15px;overflow-wrap:anywhere}
.r .t .d{color:var(--bad);text-decoration:underline;font-weight:700}
.r .c{display:flex;align-items:center;gap:6px;justify-content:flex-end}
.r .c i{display:block;height:6px;width:36px;background:var(--line);border-radius:3px;overflow:hidden}
.r .c i s{display:block;height:100%;background:var(--muted)}
.r.acc .c i s{background:var(--accent)}
.ag{font-size:11px;font-weight:600;padding:1px 8px;border-radius:999px;background:var(--chip);color:var(--muted);white-space:nowrap}
.ag.same{background:var(--okbg);color:var(--ok)}
.foot{display:flex;justify-content:space-between;align-items:center;gap:8px}
.path{display:flex;align-items:center;gap:6px}
.path code{flex:1;min-width:0;font-size:11px;color:var(--muted);background:var(--chip);border-radius:6px;padding:3px 6px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;user-select:all;cursor:text}
.path button{font-size:11px;padding:3px 8px;border-radius:6px;white-space:nowrap}
.path button.done{border-color:var(--ok);color:var(--ok)}
.tag{font-size:10px;font-weight:700;padding:1px 6px;border-radius:999px;background:var(--chip);color:var(--muted)}
.r.acc .tag{background:var(--warnbg);color:var(--accent)}
.empty{padding:40px;text-align:center;color:var(--muted)}
@media (max-width:560px){header,.bar,main{padding-left:16px;padding-right:16px}.slider input[type=range]{width:160px}.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
<h1>Review set: what each model reads and accepts</h1>
<div class="sub" id="sub"></div>
</header>
<div class="bar">
  <div class="ctl"><label for="acc">Accepted at the cut-off</label>
    <select id="acc">
      <option value="all">accepted by every selected model</option>
      <option value="any">accepted by at least one model</option>
      <option value="some">accepted by some, not all</option>
      <option value="none">accepted by no model</option>
      <option value="off">any (no acceptance filter)</option>
    </select></div>
  <div class="ctl"><label for="agree">Do the models agree?</label>
    <select id="agree">
      <option value="off">any (no agreement filter)</option>
      <option value="allsame">every model read the same serial</option>
      <option value="notsame">not all the same</option>
      <option value="majority">most agree, one or more differ</option>
      <option value="alldiff">every model read something different</option>
    </select></div>
  <div class="ctl"><label for="mode">Read vs correct serial</label>
    <select id="mode">
      <option value="any">any</option>
      <option value="allright">every model's read is right</option>
      <option value="none">no model's read is right</option>
      <option value="some">at least one read is wrong</option>
    </select></div>
  <div class="ctl"><span class="lab">Models</span><div class="models" id="models"></div></div>
  <div class="ctl"><label for="thr">Confidence cut-off</label>
    <div class="slider"><input type="range" id="thr" min="0" max="1" step="0.005" value="0.95">
      <input type="number" id="thrn" min="0" max="1" step="0.005" value="0.95"></div></div>
  <div class="ctl"><label for="q">Search</label><input type="search" id="q" placeholder="image name or serial"></div>
  <div class="ctl"><label for="sort">Sort</label>
    <select id="sort">
      <option value="confdesc">weakest read's confidence, high to low</option>
      <option value="confasc">weakest read's confidence, low to high</option>
      <option value="name">image name</option>
    </select></div>
  <div class="ctl"><span class="lab">&nbsp;</span><button id="csv">Download shown list (CSV)</button></div>
</div>
<main>
  <div class="stats"><table id="stats"></table></div>
  <div class="count"><div><b id="nshow">0</b> <span id="nlabel"></span></div>
    <div class="pager"><button id="prev">‹ Prev</button><span id="page"></span><button id="next">Next ›</button></div></div>
  <div class="grid" id="grid"></div>
  <div class="count"><span></span><div class="pager"><button id="prev2">‹ Prev</button><button id="next2">Next ›</button></div></div>
</main>

<script id="data" type="application/json">/*__DATA__*/null</script>
<script id="logic">
// ---- pure logic (no DOM) - also used by the build test
function norm(s){return String(s||"").toUpperCase().replace(/ /g,"");}
function prepare(D){
  const n=D.names.length,K=D.keys.length,right=[];
  for(let k=0;k<K;k++){const a=new Uint8Array(n);for(let i=0;i<n;i++)a[i]=(D.reads[k][i]!==""&&norm(D.reads[k][i])===norm(D.gt[i]))?1:0;right.push(a);}
  D.right=right;return D;
}
function accepted(D,k,i,thr){return D.reads[k][i]!==""&&D.conf[k][i]>=thr;}
// acceptance filter at the cut-off, over the selected models
function accOk(D,i,sel,acc,thr){
  if(acc==="off")return true;
  const na=sel.reduce((s,k)=>s+(accepted(D,k,i,thr)?1:0),0);
  if(acc==="all")return na===sel.length;
  if(acc==="none")return na===0;
  if(acc==="any")return na>0;
  if(acc==="some")return na>0&&na<sel.length;
  return true;
}
// how the selected models' reads group: top = size of the biggest group of identical (non-blank) reads
function agreeInfo(D,i,sel){const m=new Map();let blank=0;
  for(const k of sel){const r=norm(D.reads[k][i]);if(!r){blank++;continue;}m.set(r,(m.get(r)||0)+1);}
  let top=0;for(const v of m.values())top=Math.max(top,v);return {top,distinct:m.size,blank,n:sel.length};}
// agreement filter - the serial the models read, confidence ignored (a blank read never agrees)
function agreeOk(D,i,sel,ag){
  if(ag==="off")return true;
  const g=agreeInfo(D,i,sel);
  if(ag==="allsame")return g.top===g.n;
  if(ag==="notsame")return g.top<g.n;
  if(ag==="majority")return g.top>g.n/2&&g.top<g.n;
  if(ag==="alldiff")return g.n>1&&g.blank===0&&g.distinct===g.n;
  return true;
}
// read vs correct serial filter
function matchOk(D,i,sel,mode){
  const nr=sel.reduce((s,k)=>s+D.right[k][i],0);
  if(mode==="allright")return nr===sel.length;
  if(mode==="none")return nr===0;
  if(mode==="some")return nr<sel.length;
  return true;
}
const OPTS={acc:["all","any","some","none","off"],agree:["off","allsame","notsame","majority","alldiff"],mode:["any","allright","none","some"]};
function keep(D,i,sel,f,thr,q){
  if(!sel.length)return false;
  if(!accOk(D,i,sel,f.acc,thr)||!agreeOk(D,i,sel,f.agree)||!matchOk(D,i,sel,f.mode))return false;
  if(q){const Q=norm(q);if(!(D.names[i].toUpperCase().includes(Q)||norm(D.gt[i]).includes(Q)||sel.some(k=>norm(D.reads[k][i]).includes(Q))))return false;}
  return true;
}
function filterIdx(D,sel,f,thr,q){const out=[];for(let i=0;i<D.names.length;i++)if(keep(D,i,sel,f,thr,q))out.push(i);return out;}
// for every option of every drop-down: how many images it would give with the OTHER two filters as set (before search)
function optCounts(D,sel,f,thr){const c={};for(const d in OPTS){c[d]={};for(const o of OPTS[d])c[d][o]=0;}if(!sel.length)return c;
  const fn={acc:o=>i=>accOk(D,i,sel,o,thr),agree:o=>i=>agreeOk(D,i,sel,o),mode:o=>i=>matchOk(D,i,sel,o)};
  for(let i=0;i<D.names.length;i++){const ok={};for(const d in OPTS)ok[d]=fn[d](f[d])(i);
    for(const d in OPTS){let rest=true;for(const e in OPTS)if(e!==d&&!ok[e])rest=false;if(!rest)continue;for(const o of OPTS[d])if(fn[d](o)(i))c[d][o]++;}}
  return c;}
function modelStats(D,k,thr){
  let acc=0,accR=0,right=0,blank=0;const n=D.names.length;
  for(let i=0;i<n;i++){const r=D.right[k][i];right+=r;if(D.reads[k][i]===""){blank++;continue;}if(D.conf[k][i]>=thr){acc++;accR+=r;}}
  return {n,right,acc,accR,accW:acc-accR,below:n-acc-blank,blank};
}
function sortKey(D,i,sel){return Math.min(...sel.map(k=>D.conf[k][i]));}
</script>
<script>
const D=prepare(JSON.parse(document.getElementById("data").textContent));
const $=id=>document.getElementById(id);
const PER=60;let idx=[],page=0;
const store={get(k,d){try{const v=localStorage.getItem("rvd_"+k);return v===null?d:JSON.parse(v);}catch(e){return d;}},
  set(k,v){try{localStorage.setItem("rvd_"+k,JSON.stringify(v));}catch(e){}}};
$("sub").textContent=`${D.names.length.toLocaleString()} crops from ${D.source}${D.excluded?` (${D.excluded.toLocaleString()} separated out to ${D.excludedFrom})`:""}${D.gone?` · ${D.gone.toLocaleString()} deleted`:""} · built ${D.made} · "right" = read equals correct_serial (upper-case, spaces ignored) · production cut-off is 0.95`;
D.keys.forEach((k,j)=>{const l=document.createElement("label");l.innerHTML=`<input type="checkbox" value="${j}" checked> ${D.titles[j]}`;$("models").appendChild(l);});
// restore last settings
(()=>{const s=store.get("s",null);if(!s)return;["acc","agree","mode","sort"].forEach(id=>{if(s[id]&&[...$(id).options].some(o=>o.value===s[id]))$(id).value=s[id];});
  if(typeof s.thr==="number"){$("thr").value=s.thr;$("thrn").value=s.thr;}
  if(Array.isArray(s.sel))$("models").querySelectorAll("input").forEach(c=>c.checked=s.sel.includes(+c.value));})();
function sel(){return [...$("models").querySelectorAll("input:checked")].map(c=>+c.value);}
function thr(){const v=parseFloat($("thrn").value);return isNaN(v)?0.95:Math.min(1,Math.max(0,v));}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function diff(read,gt){const a=norm(read),b=norm(gt);if(!a)return '<span class="m">(blank)</span>';
  if(a.length!==b.length)return esc(read);let o="";for(let i=0;i<a.length;i++)o+=a[i]===b[i]?esc(a[i]):`<span class="d">${esc(a[i])}</span>`;return o;}
function absPath(name){return D.imgAbs+"/"+name;}
function fileUrl(name){return "file://"+absPath(name).split("/").map(encodeURIComponent).join("/");}
function copyText(t){if(navigator.clipboard&&window.isSecureContext)return navigator.clipboard.writeText(t).catch(()=>fallbackCopy(t));return Promise.resolve(fallbackCopy(t));}
function fallbackCopy(t){const ta=document.createElement("textarea");ta.value=t;ta.style.position="fixed";ta.style.opacity="0";document.body.appendChild(ta);ta.select();
  try{document.execCommand("copy");}catch(e){}ta.remove();}
function imgSrc(name){return D.imgBase.split("/").map(encodeURIComponent).join("/")+"/"+encodeURIComponent(name);}
function stats(){const t=thr();let h=`<tr><th>Model (all ${D.names.length.toLocaleString()} images)</th><th>right overall</th><th>accepted ≥ ${t.toFixed(3)}</th><th>accepted &amp; right</th><th>accepted but WRONG</th><th>below cut-off</th><th>blank</th></tr>`;
  D.keys.forEach((k,j)=>{const s=modelStats(D,j,t);const p=x=>`${x.toLocaleString()} <span class="m">(${(100*x/s.n).toFixed(1)}%)</span>`;
    h+=`<tr><td>${D.titles[j]}</td><td>${p(s.right)}</td><td>${p(s.acc)}</td><td class="ok">${p(s.accR)}</td><td class="bad">${p(s.accW)}</td><td>${p(s.below)}</td><td>${s.blank}</td></tr>`;});
  $("stats").innerHTML=h;}
const TXT={};for(const d in OPTS)[...$(d).options].forEach(o=>{(TXT[d]=TXT[d]||{})[o.value]=o.textContent;});
function filt(){return {acc:$("acc").value,agree:$("agree").value,mode:$("mode").value};}
function apply(){const s=sel(),f=filt(),t=thr();
  store.set("s",{...f,sort:$("sort").value,thr:t,sel:s});
  const cnt=optCounts(D,s,f,t);for(const d in OPTS)[...$(d).options].forEach(o=>o.textContent=`${TXT[d][o.value]} (${cnt[d][o.value].toLocaleString()})`);
  idx=filterIdx(D,s,f,t,$("q").value.trim());
  const so=$("sort").value;
  if(so==="name")idx.sort((a,b)=>D.names[a]<D.names[b]?-1:1);
  else{const key=new Float64Array(D.names.length);idx.forEach(i=>key[i]=sortKey(D,i,s));idx.sort((a,b)=>so==="confdesc"?key[b]-key[a]:key[a]-key[b]);}
  page=0;stats();render();}
function agreeChip(i,s){const g=agreeInfo(D,i,s);
  if(g.n<2)return "";
  if(g.top===g.n)return `<span class="ag same">all ${g.n} read the same</span>`;
  if(g.distinct===g.n&&!g.blank)return `<span class="ag">all ${g.n} read differently</span>`;
  return `<span class="ag">${g.top} of ${g.n} agree${g.blank?` · ${g.blank} blank`:""}</span>`;}
function render(){const s=sel(),t=thr(),pages=Math.max(1,Math.ceil(idx.length/PER));page=Math.min(page,pages-1);
  $("nshow").textContent=idx.length.toLocaleString();$("nlabel").textContent=`of ${D.names.length.toLocaleString()} images shown`;
  $("page").textContent=`page ${page+1} of ${pages}`;["prev","prev2"].forEach(i=>$(i).disabled=page===0);["next","next2"].forEach(i=>$(i).disabled=page>=pages-1);
  const g=$("grid");if(!idx.length){g.innerHTML='<div class="empty">No image matches these filters.</div>';return;}
  g.innerHTML=idx.slice(page*PER,page*PER+PER).map(i=>{const rows=s.map(k=>{const r=D.reads[k][i],c=D.conf[k][i],ok=D.right[k][i],acc=r!==""&&c>=t;
      return `<div class="r ${ok?"right":"wrong"} ${acc?"acc":""}"><span class="m">${esc(D.titles[k])}</span><span class="t mono">${ok?esc(r):diff(r,D.gt[i])}</span>
      <span class="c"><span class="tag">${acc?"ACCEPT":"review"}</span><i><s style="width:${Math.round(c*100)}%"></s></i><span class="mono">${c.toFixed(3)}</span></span></div>`;}).join("");
    const src=imgSrc(D.names[i]);
    return `<div class="card"><a class="img" href="${src}" target="_blank" rel="noopener"><img loading="lazy" src="${src}" alt="crop ${esc(D.names[i])}"></a>
      <div class="gt"><span class="lab">correct serial</span><span class="v mono">${esc(D.gt[i])}</span></div><div class="rows">${rows}</div><div class="foot"><span class="name mono">${esc(D.names[i])}</span>${agreeChip(i,s)}</div>
      <div class="path"><code class="mono" title="${esc(absPath(D.names[i]))}">${esc(absPath(D.names[i]))}</code>
      <button data-copy="${esc(absPath(D.names[i]))}" data-label="Copy path">Copy path</button><button data-copy="${esc(fileUrl(D.names[i]))}" data-label="Copy link">Copy link</button></div></div>`;}).join("");}
function csv(){const s=sel(),q=v=>'"'+String(v).replace(/"/g,'""')+'"';
  const head=["image_name","image_path","correct_serial",...s.flatMap(k=>[D.keys[k]+"_result",D.keys[k]+"_confidence",D.keys[k]+"_right"])];
  const lines=[head.join(",")].concat(idx.map(i=>[q(D.names[i]),q(absPath(D.names[i])),q(D.gt[i]),...s.flatMap(k=>[q(D.reads[k][i]),D.conf[k][i],D.right[k][i]?"yes":"no"])].join(",")));
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([lines.join("\n")],{type:"text/csv"}));
  a.download=`review_acc-${$("acc").value}_agree-${$("agree").value}_match-${$("mode").value}_conf${thr().toFixed(3)}_${idx.length}.csv`;document.body.appendChild(a);a.click();a.remove();}
let tm=null;const later=()=>{clearTimeout(tm);tm=setTimeout(apply,120);};
$("thr").addEventListener("input",()=>{$("thrn").value=$("thr").value;later();});
$("thrn").addEventListener("input",()=>{$("thr").value=thr();later();});
["acc","agree","mode","sort"].forEach(id=>$(id).addEventListener("change",apply));
$("models").addEventListener("change",apply);$("q").addEventListener("input",later);
const go=d=>{page+=d;render();window.scrollTo({top:0});};
$("prev").onclick=$("prev2").onclick=()=>go(-1);$("next").onclick=$("next2").onclick=()=>go(1);
$("csv").onclick=csv;
$("grid").addEventListener("click",e=>{const b=e.target.closest("button[data-copy]");if(!b)return;
  copyText(b.dataset.copy).then(()=>{b.textContent="Copied";b.classList.add("done");setTimeout(()=>{b.textContent=b.dataset.label;b.classList.remove("done");},1200);});});
apply();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
