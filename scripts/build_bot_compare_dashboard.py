#!/usr/bin/env python3
"""
Build a local HTML dashboard comparing the 3-model consensus `final_value`
against the source `bot_annotation.serial_value` for the TPCODL Aug 1-4 set.

Output: Some_analysis_part/bot_compare_dashboard.html   (open in a browser)
Images are referenced relative to that folder, so keep the html there.
"""
import os, csv, json, html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "Some_analysis_part")
SRC  = os.path.join(BASE, "1staug_4thaug_2026_serialno_tpcodl.csv")
PRED = os.path.join(BASE, "three_model_consensus_tpcodl.csv")  # original (keeps leading zeros)
IMGDIR = "1staug_4thaug_2026_serialno_tpcodl"           # relative to BASE
OUT  = os.path.join(BASE, "bot_compare_dashboard.html")

fname = lambda u: u.rstrip().split("/")[-1]
norm  = lambda s: s.strip().upper().replace(" ", "")

gt = {}
for r in csv.DictReader(open(SRC, newline="", encoding="utf-8")):
    gt[fname(r["readings.KWH.bigImg"])] = r["readings.KWH.bot_annotation.serial_value"]

rows = []
correct = wrong = lz_only = 0
for r in csv.DictReader(open(PRED, newline="", encoding="utf-8")):
    img, pred = r["image_name"], r["final_value"]
    bot = gt.get(img, "")
    ok = norm(pred) == norm(bot) and norm(bot) != ""
    lz = (not ok) and norm(pred).lstrip("0") == norm(bot).lstrip("0") and norm(pred).lstrip("0") != ""
    if ok: correct += 1
    else:
        wrong += 1
        if lz: lz_only += 1
    rows.append({"img": img, "bot": bot, "pred": pred,
                 "st": "ok" if ok else ("lz" if lz else "bad")})

total = correct + wrong
pct = lambda n: round(100.0 * n / total, 2) if total else 0
adj = correct + lz_only

wrong_rows = [x for x in rows if x["st"] != "ok"]

DATA = json.dumps(wrong_rows)
tpl = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>TPCODL — model vs bot annotation</title>
<style>
:root{color-scheme:light dark;
 --bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--mut:#52514e;--line:#e6e5e2;
 --good:#0ca30c;--goodink:#006300;--bad:#d03b3b;--warn:#fab219;--warnink:#7a5200;}
@media(prefers-color-scheme:dark){:root{
 --bg:#141413;--card:#1f1f1e;--ink:#f5f5f2;--mut:#b4b3ab;--line:#33322f;
 --good:#3ec13e;--goodink:#8ff08f;--bad:#ef6a6a;--warn:#fab219;--warnink:#ffd98a;}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
 background:var(--bg);color:var(--ink);padding:28px clamp(14px,4vw,48px)}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 24px;font-size:13.5px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:22px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
.kpi .n{font-size:30px;font-weight:650;letter-spacing:-.5px}
.kpi .l{color:var(--mut);font-size:12.5px;margin-top:2px;text-transform:uppercase;letter-spacing:.4px}
.kpi.good .n{color:var(--goodink)}.kpi.bad .n{color:var(--bad)}
.split{display:flex;gap:22px;flex-wrap:wrap;align-items:center;background:var(--card);
 border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:14px}
.bar{flex:1;min-width:240px;height:26px;border-radius:7px;overflow:hidden;display:flex;background:var(--line)}
.bar span{display:block;height:100%}.leg{display:flex;gap:18px;flex-wrap:wrap;font-size:13px}
.leg i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--warn);
 border-radius:10px;padding:12px 16px;margin-bottom:26px;font-size:13.5px;color:var(--mut)}
.note b{color:var(--ink)}
.gbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
.gbar h2{font-size:16px;margin:0}button{font:inherit;background:var(--card);color:var(--ink);
 border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
button.on{background:var(--ink);color:var(--bg)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card img{width:100%;height:120px;object-fit:cover;background:#000;display:block}
.card .b{padding:10px 12px;font-size:13px}
.row{display:flex;justify-content:space-between;gap:8px;margin:2px 0}
.row .k{color:var(--mut)}.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.tag{font-size:11px;padding:1px 7px;border-radius:20px;font-weight:600}
.tag.lz{background:color-mix(in srgb,var(--warn) 22%,transparent);color:var(--warnink)}
.tag.bad{background:color-mix(in srgb,var(--bad) 20%,transparent);color:var(--bad)}
.path{display:flex;align-items:center;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid var(--line)}
.path code{flex:1;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px;color:var(--mut);
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis;direction:rtl;text-align:left}
.cp{flex:none;font-size:11px;padding:2px 8px;border-radius:6px;border:1px solid var(--line);
 background:var(--bg);color:var(--ink);cursor:pointer}
.cp.done{background:var(--good);color:#fff;border-color:var(--good)}
.more{display:block;margin:22px auto;padding:9px 22px}
.bot{color:var(--bad)}.prd{color:var(--goodink)}
</style></head><body>
<h1>TPCODL serials — 3-model consensus vs bot annotation</h1>
<p class=sub>Aug 1–4 2026 · __TOTAL__ images compared · consensus <span class=mono>final_value</span> vs source <span class=mono>bot_annotation.serial_value</span> · normalized (uppercase, spaces ignored)</p>

<div class=kpis>
 <div class="kpi"><div class=n>__TOTAL__</div><div class=l>Compared</div></div>
 <div class="kpi good"><div class=n>__CORRECT__</div><div class=l>Correct · __CPCT__%</div></div>
 <div class="kpi bad"><div class=n>__WRONG__</div><div class=l>Wrong · __WPCT__%</div></div>
 <div class="kpi"><div class=n>__ADJPCT__%</div><div class=l>Correct if leading&nbsp;0s kept</div></div>
</div>

<div class=split>
 <div class=bar>
   <span style="width:__CPCT__%;background:var(--good)"></span>
   <span style="width:__LZPCT__%;background:var(--warn)"></span>
   <span style="width:__BADPCT__%;background:var(--bad)"></span>
 </div>
 <div class=leg>
   <span><i style="background:var(--good)"></i>Correct __CORRECT__</span>
   <span><i style="background:var(--warn)"></i>Leading-0 only __LZ__</span>
   <span><i style="background:var(--bad)"></i>Genuinely differs __TRUEBAD__</span>
 </div>
</div>

<div class=note><b>Read with care.</b> The <span class=mono>bot_annotation</span> is the utility's own bot output, <b>not verified ground truth</b> — in many mismatches the model is actually more correct than the bot. Also, __LZ__ "wrong" rows differ only by a leading zero stripped in the copy CSV (a spreadsheet artifact), not a real model error.</div>

<div class=gbar>
 <h2>Disagreements</h2>
 <button data-f=all class=on>All (__WRONG__)</button>
 <button data-f=bad>Genuinely differ (__TRUEBAD__)</button>
 <button data-f=lz>Leading-0 only (__LZ__)</button>
</div>
<div class=grid id=grid></div>
<button class=more id=more>Show more</button>

<script>
const DATA=__DATA__, DIR="__IMGDIR__/", ABS="__ABSDIR__/";
let filt="all", shown=0, PAGE=60;
const grid=document.getElementById('grid'), more=document.getElementById('more');
const view=()=>DATA.filter(d=>filt==="all"||d.st===filt);
function esc(s){return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))||"<em>∅</em>";}
function render(reset){
 if(reset){grid.innerHTML="";shown=0;}
 const v=view();
 for(let i=shown;i<Math.min(shown+PAGE,v.length);i++){const d=v[i];
  const c=document.createElement('div');c.className='card';
  c.innerHTML=`<img loading=lazy src="${DIR+encodeURIComponent(d.img)}" alt="">
   <div class=b>
    <div class=row><span class=k>bot</span><span class="mono bot">${esc(d.bot)}</span></div>
    <div class=row><span class=k>model</span><span class="mono prd">${esc(d.pred)}</span></div>
    <div class=row><span></span><span class="tag ${d.st}">${d.st==='lz'?'leading-0':'differs'}</span></div>
    <div class=path><code title="${ABS+d.img}">${ABS+d.img}</code><button class=cp>copy</button></div>
   </div>`;
  const btn=c.querySelector('.cp'), p=ABS+d.img;
  btn.onclick=()=>{navigator.clipboard.writeText(p).then(()=>{btn.textContent='✓';btn.classList.add('done');
    setTimeout(()=>{btn.textContent='copy';btn.classList.remove('done');},1200);});};
  grid.appendChild(c);}
 shown=Math.min(shown+PAGE,v.length);
 more.style.display=shown>=v.length?'none':'block';
}
document.querySelectorAll('[data-f]').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('[data-f]').forEach(x=>x.classList.remove('on'));
 b.classList.add('on');filt=b.dataset.f;render(true);});
more.onclick=()=>render(false);
render(true);
</script>
</body></html>"""

true_bad = wrong - lz_only
subst = {
    "__TOTAL__": f"{total:,}", "__CORRECT__": f"{correct:,}", "__WRONG__": f"{wrong:,}",
    "__CPCT__": f"{pct(correct)}", "__WPCT__": f"{pct(wrong)}", "__ADJPCT__": f"{pct(adj)}",
    "__LZ__": f"{lz_only:,}", "__TRUEBAD__": f"{true_bad:,}",
    "__LZPCT__": f"{pct(lz_only)}", "__BADPCT__": f"{pct(true_bad)}",
    "__DATA__": DATA, "__IMGDIR__": IMGDIR,
    "__ABSDIR__": os.path.join(BASE, IMGDIR),
}
out = tpl
for k, v in subst.items():
    out = out.replace(k, v)

open(OUT, "w", encoding="utf-8").write(out)
print("correct=%d wrong=%d (lz_only=%d true_bad=%d)  total=%d" % (correct, wrong, lz_only, true_bad, total))
print("wrote", OUT)
