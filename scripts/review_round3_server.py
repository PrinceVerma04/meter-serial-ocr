#!/usr/bin/env python3
"""
Review round 3 - check every doubtful label among the new crops before they go into v4.2 training.

Doubtful = one of:
  1. models disagree     at least one of v3 FP32 / v3 PTQ / v4 FP32 / v4 PTQ reads something else
  2. unusual length      ground truth is not 8 characters (a digit may be hidden), all models agree
  3. "07" meters         serial printed as "07 41659893"; labels are written 0741659893 or 741659893
                         depending on whether the 0 is inside the crop -> one rule for all of them
For each crop: keep the ground truth, keep a model's read, type the right serial, or remove it.
Every click is saved at once to
    new testing data 4 nov/review_round3_decisions.csv
The list of crops is frozen in review_round3_items.csv the first time this starts, so it does not
change while you work. Afterwards run
    .venv/bin/python scripts/build_v4_retrain_data.py

Run from the project root (foreground, Ctrl+C to stop):
    .venv/bin/python scripts/review_round3_server.py                 # http://127.0.0.1:8097
    .venv/bin/python scripts/review_round3_server.py --host 0.0.0.0  # also from the LAN
"""
import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import review_round2_server as R2   # noqa: E402  (paths, readers, lan_ip)

NEW = os.path.join(ROOT, "datasets", "v4_retrain_nov4", "labels.csv")
ITEMS_FILE = os.path.join(R2.TEST, "review_round3_items.csv")
DECISIONS = os.path.join(R2.TEST, "review_round3_decisions.csv")
RULE_KEY = "(07 rule)"
RULES = {"as_visible": "As printed and visible", "serial_only": "8-digit serial only", "always_07": "Always with 07"}
GROUPS = [
    ("disagree", "Models disagree with the ground truth",
     "At least one model reads something else. Pick the right serial, or remove the crop if a digit can't be read."),
    ("length", "Not 8 characters, all models agree",
     "Everyone reads the same, but most serials have 8 digits: check that no digit is hidden by glare or cut off."),
    ("prefix07", "\"07\" meters",
     "Printed as \"07 41659893\". Check each label matches the crop; the rule above decides how all of them are written."),
]
MODELS = ["v3_fp32", "v3_ptq", "v4_fp32", "v4_ptq"]
FAM07 = re.compile(r"0?7\d{8}")
LOCK = threading.Lock()
DATA = {}


def norm(s):
    return (s or "").strip().upper().replace(" ", "")


def is07(s):
    return bool(FAM07.fullmatch(norm(s)))


def apply_rule(serial, rule):
    s = norm(serial)
    if not is07(s) or rule in (None, "", "as_visible"):
        return serial
    return s[-8:] if rule == "serial_only" else "07" + s[-8:]


def load_reads_conf():
    import openpyxl
    wb = openpyxl.load_workbook(R2.READS, read_only=True)
    er = list(wb["extracted_results"].iter_rows(values_only=True))
    cf = list(wb["confidence"].iter_rows(values_only=True))
    wb.close()
    keys = [R2.cell_text(k) for k in er[0][1:]]
    reads = {R2.cell_text(r[0]): dict(zip(keys, [R2.cell_text(v) for v in r[1:]])) for r in er[1:] if r[0]}
    conf = {R2.cell_text(r[0]): dict(zip(keys, [float(v or 0) for v in r[1:]])) for r in cf[1:] if r[0]}
    return reads, conf


def freeze_items():
    """decide once which crops are doubtful; later starts reuse the same list"""
    if os.path.exists(ITEMS_FILE):
        return [r for r in csv.DictReader(open(ITEMS_FILE, encoding="utf-8"))]
    if not os.path.exists(NEW):
        sys.exit(f"{NEW} missing - run scripts/build_v4_retrain_data.py first")
    reads, _ = load_reads_conf()
    out = []
    for r in csv.DictReader(open(NEW, encoding="utf-8")):
        n, g = r["image_name"], r["serial"]
        rd = reads.get(n)
        if rd is None:
            sys.exit(f"no model reads for {n} - run scripts/step1_run_all_models.py first")
        if any(norm(rd.get(k, "")) != norm(g) for k in MODELS):
            grp = "disagree"
        elif is07(g):
            grp = "prefix07"
        elif len(norm(g)) != 8:
            grp = "length"
        else:
            continue
        out.append({"image_name": n, "group": grp, "ground_truth": g})
    with open(ITEMS_FILE + ".tmp", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_name", "group", "ground_truth"])
        w.writeheader()
        w.writerows(out)
    os.replace(ITEMS_FILE + ".tmp", ITEMS_FILE)
    return out


def load_decisions():
    d = {}
    if os.path.exists(DECISIONS):
        for r in csv.DictReader(open(DECISIONS, encoding="utf-8")):
            d[r["image_name"]] = r
    return d


def save_decisions(d):
    tmp = DECISIONS + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_name", "decision", "serial", "decided_at"])
        w.writeheader()
        for k in sorted(d, key=lambda x: (x != RULE_KEY, x.lower())):
            w.writerow(d[k])
    os.replace(tmp, DECISIONS)


def build():
    items = freeze_items()
    reads, conf = load_reads_conf()
    titles = R2.MODEL_TITLES
    out = []
    for it in items:
        n, g = it["image_name"], it["ground_truth"]
        rd, cf = reads.get(n, {}), conf.get(n, {})
        cands = {}
        cands.setdefault(norm(g), {"serial": g, "sources": []})["sources"].append("ground truth")
        for k in MODELS:
            v = rd.get(k, "")
            if v:
                cands.setdefault(norm(v), {"serial": v, "sources": []})["sources"].append(titles[k])
        out.append({"name": n, "group": it["group"], "gt": g,
                    "reads": [{"model": titles[k], "read": rd.get(k, ""), "conf": round(cf.get(k, 0.0), 3)} for k in MODELS],
                    "candidates": list(cands.values())})
    DATA.update({"items": out, "groups": [{"key": k, "title": t, "desc": d} for k, t, d in GROUPS],
                 "rules": [{"key": k, "title": t} for k, t in RULES.items()]})


class Handler(BaseHTTPRequestHandler):
    server_version = "ReviewRound3/1.0"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/state":
            with LOCK:
                return self._json(200, {**DATA, "decisions": load_decisions()})
        if path.startswith("/img/"):
            name = os.path.basename(path[5:])
            for folder in (os.path.join(ROOT, "datasets", "v4_retrain_nov4", "images"), R2.CLEAN, R2.REMOVED):
                p = os.path.join(folder, name)
                if os.path.isfile(p):
                    return self._send(200, open(p, "rb").read(), "image/jpeg")
            return self._json(404, {"error": "no such image"})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "bad request"})
        items = {it["name"]: it for it in DATA["items"]}
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with LOCK:
            d = load_decisions()
            if path == "/api/decide":
                name, dec = body.get("image_name", ""), body.get("decision", "")
                serial = (body.get("serial") or "").strip().upper()
                if name == RULE_KEY:
                    if dec not in RULES:
                        return self._json(400, {"error": "bad rule"})
                    d[name] = {"image_name": name, "decision": dec, "serial": "", "decided_at": stamp}
                elif name in items:
                    if dec == "":
                        d.pop(name, None)
                    elif dec == "remove":
                        d[name] = {"image_name": name, "decision": "remove", "serial": "", "decided_at": stamp}
                    elif dec == "keep" and serial and len(serial) <= 20 and serial.replace(" ", "").isalnum():
                        d[name] = {"image_name": name, "decision": "keep", "serial": serial, "decided_at": stamp}
                    else:
                        return self._json(400, {"error": "serial must be 1-20 letters/digits"})
                else:
                    return self._json(400, {"error": "unknown image"})
            elif path == "/api/bulk":
                grp = body.get("group", "")
                if grp not in {g[0] for g in GROUPS}:
                    return self._json(400, {"error": "unknown group"})
                for it in DATA["items"]:
                    if it["group"] == grp and it["name"] not in d:
                        d[it["name"]] = {"image_name": it["name"], "decision": "keep", "serial": it["gt"].upper(),
                                         "decided_at": stamp}
            else:
                return self._json(404, {"error": "not found"})
            save_decisions(d)
            return self._json(200, {"ok": True, "decisions": d})


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Round 3</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1b1f24;--muted:#5f6b7a;--line:#e2e6eb;--accent:#2457c5;--accent-soft:#e8eefb;
--ok:#17803d;--ok-bg:#e7f6ec;--bad:#b42318;--bad-bg:#fdecea;--warn:#9a6700;--warn-bg:#fff4d6;--chip:#eef1f5}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f1216;--panel:#171b21;--ink:#e6e9ee;--muted:#9aa5b4;--line:#2a3039;
--accent:#7aa2ff;--accent-soft:#1d2840;--ok:#5fd28a;--ok-bg:#11291b;--bad:#ff8a80;--bad-bg:#341716;--warn:#f2c14e;--warn-bg:#33290f;--chip:#222832}}
:root[data-theme="dark"]{--bg:#0f1216;--panel:#171b21;--ink:#e6e9ee;--muted:#9aa5b4;--line:#2a3039;
--accent:#7aa2ff;--accent-soft:#1d2840;--ok:#5fd28a;--ok-bg:#11291b;--bad:#ff8a80;--bad-bg:#341716;--warn:#f2c14e;--warn-bg:#33290f;--chip:#222832}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:0 16px 80px}
header{position:sticky;top:0;z-index:5;background:var(--bg);padding:12px 0;border-bottom:1px solid var(--line);display:flex;gap:12px;justify-content:space-between;align-items:center;flex-wrap:wrap}
h1{font-size:19px;margin:0}.sub{color:var(--muted);font-size:13px}
.prog{display:flex;align-items:center;gap:10px;font-size:14px;flex-wrap:wrap}.bar{width:160px;height:8px;background:var(--chip);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok)}
.btn{font:inherit;font-size:14px;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:6px 11px;cursor:pointer}
.btn:hover{border-color:var(--accent)}.btn.on{background:var(--accent-soft);border-color:var(--accent);color:var(--accent)}
.btn.sel{background:var(--ok-bg);border-color:var(--ok);color:var(--ink);font-weight:600}.btn.rm.sel{background:var(--bad-bg);border-color:var(--bad)}
.btn.arm{background:var(--warn-bg);border-color:var(--warn)}
.ghead{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;flex-wrap:wrap;margin:26px 0 10px}
h2{font-size:17px;margin:0}.gdesc{color:var(--muted);font-size:13px}
.rule{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:6px 0 4px}
.rule .opts{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}.rule .ex{font-size:12px;color:var(--muted);margin-top:8px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:10px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}
.card.done{border-color:color-mix(in srgb,var(--ok) 45%,transparent)}
.imgbox{background:var(--chip);border-radius:8px;padding:8px;display:flex;align-items:center;justify-content:center;min-height:100px}
.imgbox img{width:100%;max-width:460px;height:auto}
.fname{font:11px ui-monospace,Menlo,Consolas,monospace;color:var(--muted);margin-top:5px;word-break:break-all}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-weight:600}
.gt{font-size:13px;color:var(--muted)}.gt b{font-size:18px;color:var(--ink)}
table{border-collapse:collapse;font-size:13px;margin:6px 0 10px}td{padding:2px 10px 2px 0}td:first-child{color:var(--muted)}
.diff{color:var(--bad)}
.choices{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.choices input{font:inherit;font-family:ui-monospace,Menlo,Consolas,monospace;padding:5px 8px;border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--ink);width:150px}
.saved{font-size:12px;color:var(--ok);min-height:16px;margin-top:4px}.err{color:var(--bad)}
.empty{color:var(--muted);font-size:13px;padding:6px 0}
@media (max-width:760px){.card{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<header><div><h1>Review round 3: doubtful labels in the new crops</h1>
<div class="sub">Saved on every click to new testing data 4 nov/review_round3_decisions.csv</div></div>
<div class="prog"><button class="btn" id="onlyBtn">Show only undecided</button><span id="progTxt"></span><span class="bar"><i id="progBar"></i></span></div></header>
<div id="groups"></div></div>
<script>
let S=null, only=false;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const norm=s=>(s||"").toUpperCase().replace(/\s+/g,"");
async function api(path,body){const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw new Error(j.error||"failed");S.decisions=j.decisions;return j;}
async function load(){S=await(await fetch("/api/state")).json();render();}
function progress(){const n=S.items.length+1,d=S.items.filter(i=>S.decisions[i.name]).length+(S.decisions["(07 rule)"]?1:0);
 document.getElementById("progTxt").textContent=`${d} of ${n} decided`;document.getElementById("progBar").style.width=(100*d/n)+"%";}
function cardHtml(it){
 const d=S.decisions[it.name]||{}, g=norm(it.gt);
 const reads=it.reads.map(r=>`<tr><td>${esc(r.model)}</td><td class="mono ${norm(r.read)!==g?"diff":""}">${esc(r.read)||"-"}</td><td>${r.conf.toFixed(3)}</td></tr>`).join("");
 const known=it.candidates.map(c=>norm(c.serial)); const other=d.decision==="keep"&&!known.includes(norm(d.serial));
 const btns=it.candidates.map(c=>`<button class="btn ${d.decision==="keep"&&norm(d.serial)===norm(c.serial)?"sel":""}" data-act="keep" data-serial="${esc(c.serial)}" title="${esc(c.sources.join(", "))}">Keep <span class="mono">${esc(c.serial)}</span></button>`).join("");
 return `<div class="card ${d.decision?"done":""}" data-name="${esc(it.name)}"><div><div class="imgbox"><img loading="lazy" src="/img/${encodeURIComponent(it.name)}" alt=""></div><div class="fname">${esc(it.name)}</div></div>
 <div><div class="gt">Ground truth <b class="mono">${esc(it.gt)}</b></div><table>${reads}</table>
 <div class="choices">${btns}<input type="text" placeholder="type serial" value="${other?esc(d.serial):""}" class="${other?"sel":""}"><button class="btn ${other?"sel":""}" data-act="type">Save typed</button>
 <button class="btn rm ${d.decision==="remove"?"sel":""}" data-act="remove">Remove</button></div><div class="saved"></div></div></div>`;}
function render(){
 const rd=(S.decisions["(07 rule)"]||{}).decision;
 document.getElementById("groups").innerHTML=S.groups.map(g=>{
  const its=S.items.filter(i=>i.group===g.key), und=its.filter(i=>!S.decisions[i.name]).length, shown=only?its.filter(i=>!S.decisions[i.name]):its;
  const rule=g.key!=="prefix07"?"":`<div class="rule"><b>Rule for every "07" serial</b> (also applied to the test-set ground truth)
   <div class="opts">${S.rules.map(r=>`<button class="btn ${rd===r.key?"sel":""}" data-rule="${r.key}">${esc(r.title)}</button>`).join("")}</div>
   <div class="ex">As printed and visible: <span class="mono">0741659893</span>, or <span class="mono">741659893</span> when the 0 is cut off (no change to labels) &middot;
   8-digit serial only: <span class="mono">41659893</span> &middot; Always with 07: <span class="mono">0741659893</span> even when the 0 is cut off</div><div class="saved" id="ruleMsg"></div></div>`;
  return `<div class="ghead"><div><h2>${esc(g.title)} (${its.length}, ${und} undecided)</h2><div class="gdesc">${esc(g.desc)}</div></div>
   ${und?`<button class="btn" data-bulk="${g.key}">Keep ground truth for the ${und} undecided</button>`:""}</div>${rule}
   ${shown.length?shown.map(cardHtml).join(""):'<div class="empty">Nothing undecided here.</div>'}`;}).join("");
 wire(); progress();}
function wire(){
 document.querySelectorAll(".card").forEach(c=>{const name=c.dataset.name,msg=c.querySelector(".saved"),txt=c.querySelector("input");
  const go=async(decision,serial)=>{msg.textContent="saving...";msg.classList.remove("err");
   try{await api("/api/decide",{image_name:name,decision,serial});msg.textContent="saved";
    c.classList.add("done");c.querySelectorAll(".btn").forEach(b=>b.classList.remove("sel"));txt.classList.remove("sel");
    if(decision==="remove")c.querySelector('[data-act=remove]').classList.add("sel");
    else{const b=[...c.querySelectorAll('[data-act=keep]')].find(b=>norm(b.dataset.serial)===norm(serial));if(b)b.classList.add("sel");else{txt.classList.add("sel");c.querySelector('[data-act=type]').classList.add("sel");}}
    progress();}catch(e){msg.textContent="not saved: "+e.message;msg.classList.add("err");}};
  c.querySelectorAll("[data-act=keep]").forEach(b=>b.onclick=()=>go("keep",b.dataset.serial));
  c.querySelector("[data-act=remove]").onclick=()=>go("remove","");
  c.querySelector("[data-act=type]").onclick=()=>{if(txt.value.trim())go("keep",txt.value.trim());else{msg.textContent="type the serial first";txt.focus();}};
  txt.onkeydown=e=>{if(e.key==="Enter"&&txt.value.trim())go("keep",txt.value.trim());};});
 document.querySelectorAll("[data-rule]").forEach(b=>b.onclick=async()=>{const m=document.getElementById("ruleMsg");
  try{await api("/api/decide",{image_name:"(07 rule)",decision:b.dataset.rule});document.querySelectorAll("[data-rule]").forEach(x=>x.classList.toggle("sel",x===b));m.textContent="saved";progress();}catch(e){m.textContent="not saved: "+e.message;m.classList.add("err");}});
 document.querySelectorAll("[data-bulk]").forEach(b=>b.onclick=async()=>{
  if(!b.classList.contains("arm")){b.classList.add("arm");b.textContent="Click again to confirm";setTimeout(()=>{if(b.isConnected){b.classList.remove("arm");render();}},4000);return;}
  try{await api("/api/bulk",{group:b.dataset.bulk});render();}catch(e){b.textContent="failed: "+e.message;}});}
document.getElementById("onlyBtn").onclick=e=>{only=!only;e.target.classList.toggle("on",only);render();};
load();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8097)
    a = ap.parse_args()
    build()
    c = {}
    for it in DATA["items"]:
        c[it["group"]] = c.get(it["group"], 0) + 1
    print(f"Review round 3: {len(DATA['items'])} crops {c} + the 07 rule   decisions -> {os.path.relpath(DECISIONS, ROOT)}")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"Open  http://127.0.0.1:{a.port}")
    if a.host == "0.0.0.0":
        print(f"LAN   http://{R2.lan_ip()}:{a.port}")
    print("Stop with Ctrl+C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
