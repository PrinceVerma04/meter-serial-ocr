#!/usr/bin/env python3
"""
Review round 2 - decide the uncertain crops before building the v4 retraining data.

Shows each uncertain crop large, with every serial it could be and where that serial comes from
(ground truth file, v4 training label, model reads, your notes). For each you pick
"keep as <serial>", "remove", or type the right serial. Every click is saved at once to
    new testing data 4 nov/review_round2_decisions.csv
so you can close the page and come back. Nothing else is changed here; afterwards run
    .venv/bin/python scripts/build_v4_retrain_data.py

Run from the project root (foreground, Ctrl+C to stop):
    .venv/bin/python scripts/review_round2_server.py                 # http://127.0.0.1:8096
    .venv/bin/python scripts/review_round2_server.py --host 0.0.0.0  # also from the LAN
"""
import argparse
import csv
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.path.join(ROOT, "new testing data 4 nov")
CLEAN = os.path.join(TEST, "clean")
REMOVED = os.path.join(TEST, "removed")
GT_ORIG = os.path.join(TEST, "clean_extracted_serials_before_review.xlsx")
GT_NOW = os.path.join(TEST, "clean_extracted_serials.xlsx")
READS = os.path.join(ROOT, "results", "2026-11-04_all_models_test", "model_reads.xlsx")
V4_LABELS = os.path.join(ROOT, "datasets", "v4_train", "labels.csv")
DECISIONS = os.path.join(TEST, "review_round2_decisions.csv")
OVERLAP_KEY = "(overlap)"

MODEL_TITLES = {"v3_fp32": "v3 FP32", "v3_ptq": "v3 PTQ", "v4_fp32": "v4 FP32", "v4_ptq": "v4 PTQ"}
# reads of images whose rows were already dropped from model_reads.xlsx (from the 4-Nov run)
READS_FALLBACK = {
    "dkUKKqkGSCe-qNf71FMUi7_null_bigImg_1787044235.jpg":
        {"v3_fp32": "14388192", "v3_ptq": "14388192", "v4_fp32": "44388192", "v4_ptq": "44388192"},
}

GROUPS = [
    ("notes", "Your review notes - which way did you mean?",
     "Your notes for these can be read two ways. Pick what you meant."),
    ("v4train", "Serial differs from v4's training data",
     "v4 was trained on these exact images with a different serial. A training label must be certain."),
    ("agree", "All 4 models agree, ground truth differs",
     "Every model reads the same serial, but the ground truth file says something else."),
]
# image, group, extra candidates {serial: source}, note shown on the card
ITEMS = [
    ("dkUKKqkGSCe-qNf71FMUi7_null_bigImg_1787044235.jpg", "notes", {},
     'Your note above it: "This image ground truth is wrong, including V3 FP32, V3 PTQ are wrong". '
     '"Remove this image and related data" comes after it, directly above the next image.'),
    ("dLziqMQEQtSa5ZpNp_bP98_null_bigImg_1786007618.jpg", "notes", {},
     '"Remove this image and related data" is written directly above this image in your notes; '
     "no note below it."),
    ("dvimFBVvT8SMMd6cTWCUMZ_null_bigImg_1787304108.jpg", "notes", {"35137307": "your note"},
     'Listed twice in your notes: once "Remove the image and related data", once "ground truth is wrong, '
     'correct result is 35137307, remember this result". It is the only dvim... image with ground truth 137307.'),
    ("cfJtc--WTiG6mbTKA2gDf-_null_bigImg_1786689449.jpg", "v4train", {}, "First digit partly hidden by glare."),
    ("dnCqf07fQnq1-PG93Fedkm_null_bigImg_1788076355.jpg", "v4train", {}, "Third digit hidden by glare (3 or 8)."),
    ("euFF2SVqSpGifbVrZh0PFx_null_bigImg_1785918708.jpg", "v4train", {}, "First digit faint (1 or 4)."),
    ("ewA9Y3XHTmaIFBHovqYAoj_null_bigImg_1787579011.jpg", "v4train", {}, "Last digit cut off at the edge."),
    ("cKClmMrkRbe4Rhix3sbQiX_null_bigImg_1787464422.jpg", "agree", {}, "First digit hidden by glare."),
    ("cYK36UhOSbyvDjbCPEGr9d_null_bigImg_1788085245.jpg", "agree", {}, "Faint mark after the last digit."),
    ("d72_9kq_Snq25c4A2fWEl9_null_bigImg_1787395795.jpg", "agree", {}, "First digit faint."),
    ("f5vI4kaiRBGgw9qdHYxEEs_null_bigImg_1786869561.jpg", "agree", {}, "Fifth digit blurred (3 or 9)."),
]

LOCK = threading.Lock()
DATA = {}


def cell_text(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def norm(s):
    return (s or "").strip().upper().replace(" ", "")


def load_gt(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["roi_labels"] if "roi_labels" in wb.sheetnames else wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    head = [cell_text(h) for h in rows[0]]
    i_n, i_g = head.index("image_name"), head.index("extracted_serial_number")
    return {cell_text(r[i_n]): cell_text(r[i_g]) for r in rows[1:] if r[i_n] is not None}


def load_reads():
    import openpyxl
    if not os.path.exists(READS):
        return {}
    wb = openpyxl.load_workbook(READS, read_only=True)
    er = list(wb["extracted_results"].iter_rows(values_only=True))
    wb.close()
    keys = [cell_text(k) for k in er[0][1:]]
    return {cell_text(r[0]): dict(zip(keys, [cell_text(v) for v in r[1:]])) for r in er[1:] if r[0] is not None}


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
        for k in sorted(d, key=lambda x: (x != OVERLAP_KEY, x.lower())):
            w.writerow(d[k])
    os.replace(tmp, DECISIONS)


def build():
    orig, now, reads = load_gt(GT_ORIG), load_gt(GT_NOW), load_reads()
    v4 = {r["image_name"]: r for r in csv.DictReader(open(V4_LABELS, encoding="utf-8"))}
    items = []
    for name, group, extra, note in ITEMS:
        if name not in orig:
            sys.exit(f"{name} is not in {GT_ORIG}")
        r = reads.get(name) or READS_FALLBACK.get(name, {})
        cands = {}

        def add(serial, src):
            if serial:
                cands.setdefault(norm(serial), {"serial": serial, "sources": []})["sources"].append(src)
        add(orig[name], "ground truth file")
        if name in v4:
            src = {"new_human_reviewed": "human reviewed", "new_auto_4_way_agreement": "auto-agreement"}.get(
                v4[name]["label_source"], v4[name]["label_source"])
            add(v4[name]["serial"], f"v4 training label ({src}, {v4[name]['split']})")
        for k, t in MODEL_TITLES.items():
            add(r.get(k, ""), t)
        for s, src in extra.items():
            add(s, src)
        status = "removed" if name not in now else f"kept as {now[name]}"
        items.append({"name": name, "group": group, "note": note, "gt": orig[name],
                      "v4train": v4[name]["serial"] if name in v4 else "",
                      "reads": [{"model": t, "read": r.get(k, "")} for k, t in MODEL_TITLES.items()],
                      "candidates": list(cands.values()), "status": status})
    in_v4 = sum(1 for n in now if n in v4)
    DATA.update({"items": items, "groups": [{"key": k, "title": t, "desc": d} for k, t, d in GROUPS],
                 "overlap": {"in_v4": in_v4, "new": len(now) - in_v4, "total": len(now)}})


class Handler(BaseHTTPRequestHandler):
    server_version = "ReviewRound2/1.0"

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
            for folder in (CLEAN, REMOVED):
                p = os.path.join(folder, name)
                if os.path.isfile(p):
                    return self._send(200, open(p, "rb").read(), "image/jpeg")
            return self._json(404, {"error": "no such image"})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/decide":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "bad request"})
        name, dec, serial = body.get("image_name", ""), body.get("decision", ""), (body.get("serial") or "").strip()
        valid_names = {it["name"] for it in DATA["items"]} | {OVERLAP_KEY}
        if name not in valid_names:
            return self._json(400, {"error": "unknown image"})
        if name == OVERLAP_KEY and dec not in ("new_only", "all_flagged"):
            return self._json(400, {"error": "bad overlap choice"})
        if name != OVERLAP_KEY and dec not in ("keep", "remove", ""):
            return self._json(400, {"error": "bad decision"})
        if dec == "keep" and (not serial or len(serial) > 20 or not serial.replace(" ", "").isalnum()):
            return self._json(400, {"error": "serial must be 1-20 letters/digits"})
        with LOCK:
            d = load_decisions()
            if dec == "":
                d.pop(name, None)
            else:
                d[name] = {"image_name": name, "decision": dec, "serial": serial.upper() if dec == "keep" else "",
                           "decided_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_decisions(d)
            return self._json(200, {"ok": True, "decisions": d})


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review Round 2</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1b1f24;--muted:#5f6b7a;--line:#e2e6eb;--accent:#2457c5;--accent-soft:#e8eefb;
--ok:#17803d;--ok-bg:#e7f6ec;--bad:#b42318;--bad-bg:#fdecea;--warn:#9a6700;--warn-bg:#fff4d6;--chip:#eef1f5}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f1216;--panel:#171b21;--ink:#e6e9ee;--muted:#9aa5b4;--line:#2a3039;
--accent:#7aa2ff;--accent-soft:#1d2840;--ok:#5fd28a;--ok-bg:#11291b;--bad:#ff8a80;--bad-bg:#341716;--warn:#f2c14e;--warn-bg:#33290f;--chip:#222832}}
:root[data-theme="dark"]{--bg:#0f1216;--panel:#171b21;--ink:#e6e9ee;--muted:#9aa5b4;--line:#2a3039;
--accent:#7aa2ff;--accent-soft:#1d2840;--ok:#5fd28a;--ok-bg:#11291b;--bad:#ff8a80;--bad-bg:#341716;--warn:#f2c14e;--warn-bg:#33290f;--chip:#222832}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:20px 16px 80px}
header{position:sticky;top:0;z-index:5;background:var(--bg);padding:12px 0;border-bottom:1px solid var(--line);margin-bottom:16px;display:flex;gap:12px;justify-content:space-between;align-items:center;flex-wrap:wrap}
h1{font-size:20px;margin:0}.sub{color:var(--muted);font-size:13px}
.prog{display:flex;align-items:center;gap:10px;font-size:14px}.bar{width:180px;height:8px;background:var(--chip);border-radius:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok)}
h2{font-size:17px;margin:28px 0 4px}.gdesc{color:var(--muted);font-size:13px;margin-bottom:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px;display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:18px}
.card.done{border-color:color-mix(in srgb,var(--ok) 45%,transparent)}
.imgbox{background:var(--chip);border-radius:8px;padding:10px;display:flex;align-items:center;justify-content:center;min-height:140px}
.imgbox img{width:100%;max-width:560px;height:auto}
.fname{font:12px ui-monospace,Menlo,Consolas,monospace;color:var(--muted);margin-top:6px;word-break:break-all}
.note{background:var(--warn-bg);color:var(--ink);border-radius:8px;padding:8px 10px;font-size:13px;margin-top:10px}
.status{font-size:12px;color:var(--muted);margin-top:6px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:10px}td{padding:3px 6px;border-bottom:1px solid var(--line)}td:first-child{color:var(--muted);width:42%}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-weight:600}
.opts{display:flex;flex-direction:column;gap:6px}
.opt{display:flex;gap:10px;align-items:flex-start;border:1px solid var(--line);border-radius:8px;padding:8px 10px;cursor:pointer}
.opt:hover{border-color:var(--accent)}.opt.sel{border-color:var(--accent);background:var(--accent-soft)}
.opt.rm.sel{border-color:var(--bad);background:var(--bad-bg)}
.opt input[type=radio]{margin-top:3px}.opt .big{font-size:17px}.opt .src{font-size:12px;color:var(--muted)}
.opt input[type=text]{font:inherit;font-family:ui-monospace,Menlo,Consolas,monospace;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);width:170px}
.saved{font-size:12px;color:var(--ok);min-height:16px;margin-top:4px}.err{color:var(--bad)}
.ov{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.ov .opt{margin-top:8px}
@media (max-width:760px){.card{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<header><div><h1>Review round 2: uncertain crops for v4 retraining</h1>
<div class="sub">Every choice is saved immediately to new testing data 4 nov/review_round2_decisions.csv</div></div>
<div class="prog"><span id="progTxt"></span><span class="bar"><i id="progBar"></i></span></div></header>
<div id="groups"></div>
<h2>Images already in v4's training data</h2>
<div class="gdesc">These have the same serial in datasets/v4_train, so adding them again would duplicate them.</div>
<div class="ov" id="ov"></div>
</div>
<script>
let S = null;
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const norm = s => (s||"").toUpperCase().replace(/\s+/g,"");
async function load(){ S = await (await fetch("/api/state")).json(); render(); }
async function decide(name, decision, serial, msgEl){
  msgEl.textContent = "saving..."; msgEl.classList.remove("err");
  try{
    const r = await fetch("/api/decide",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image_name:name,decision,serial})});
    const j = await r.json();
    if(!r.ok) throw new Error(j.error||"failed");
    S.decisions = j.decisions; msgEl.textContent = "saved " + new Date().toLocaleTimeString(); progress(); mark();
  }catch(e){ msgEl.textContent = "not saved: " + e.message; msgEl.classList.add("err"); }
}
function progress(){
  const n = S.items.length + 1, d = S.items.filter(i => S.decisions[i.name]).length + (S.decisions["(overlap)"] ? 1 : 0);
  document.getElementById("progTxt").textContent = `${d} of ${n} decided`;
  document.getElementById("progBar").style.width = (100*d/n) + "%";
}
function mark(){
  document.querySelectorAll(".card").forEach(c => c.classList.toggle("done", !!S.decisions[c.dataset.name]));
  document.querySelectorAll(".opt").forEach(o => o.classList.toggle("sel", o.querySelector("input[type=radio]").checked));
}
function card(it){
  const d = S.decisions[it.name] || {};
  const reads = it.reads.map(r => `<tr><td>${esc(r.model)}</td><td class="mono">${esc(r.read)||"-"}</td></tr>`).join("");
  const known = it.candidates.map(c => norm(c.serial));
  const isOther = d.decision === "keep" && !known.includes(norm(d.serial));
  const opts = it.candidates.map((c,i) => `<label class="opt"><input type="radio" name="r_${esc(it.name)}" value="keep" data-serial="${esc(c.serial)}" ${d.decision==="keep" && norm(d.serial)===norm(c.serial) ? "checked":""}>
      <div><div class="big">Keep as <span class="mono">${esc(c.serial)}</span></div><div class="src">${esc(c.sources.join(" · "))}</div></div></label>`).join("") +
    `<label class="opt"><input type="radio" name="r_${esc(it.name)}" value="other" ${isOther?"checked":""}>
      <div><div class="big">Keep as <input type="text" placeholder="type serial" value="${isOther?esc(d.serial):""}"></div><div class="src">your own reading</div></div></label>
     <label class="opt rm"><input type="radio" name="r_${esc(it.name)}" value="remove" ${d.decision==="remove"?"checked":""}>
      <div><div class="big">Remove</div><div class="src">leave this crop out of the ground truth and the training data</div></div></label>`;
  return `<div class="card" data-name="${esc(it.name)}"><div>
      <div class="imgbox"><img src="/img/${encodeURIComponent(it.name)}" alt="${esc(it.name)}"></div>
      <div class="fname">${esc(it.name)}</div>
      <div class="note">${esc(it.note)}</div><div class="status">Now: ${esc(it.status)}</div></div>
    <div><table><tr><td>Ground truth file</td><td class="mono">${esc(it.gt)}</td></tr>
      <tr><td>v4 training label</td><td class="mono">${esc(it.v4train)||"not in v4 training data"}</td></tr>${reads}</table>
      <div class="opts">${opts}</div><div class="saved"></div></div></div>`;
}
function render(){
  document.getElementById("groups").innerHTML = S.groups.map(g => {
    const its = S.items.filter(i => i.group === g.key);
    return `<h2>${esc(g.title)} (${its.length})</h2><div class="gdesc">${esc(g.desc)}</div>` + its.map(card).join("");
  }).join("");
  document.querySelectorAll(".card").forEach(c => {
    const name = c.dataset.name, msg = c.querySelector(".saved"), txt = c.querySelector("input[type=text]");
    c.querySelectorAll("input[type=radio]").forEach(r => r.addEventListener("change", () => {
      if (r.value === "keep") decide(name, "keep", r.dataset.serial, msg);
      else if (r.value === "remove") decide(name, "remove", "", msg);
      else { txt.focus(); if (txt.value.trim()) decide(name, "keep", txt.value.trim(), msg); else msg.textContent = "type the serial, then press Enter"; }
    }));
    txt.addEventListener("focus", () => { c.querySelector("input[value=other]").checked = true; mark(); });
    txt.addEventListener("keydown", e => { if (e.key === "Enter" && txt.value.trim()) decide(name, "keep", txt.value.trim(), msg); });
    txt.addEventListener("change", () => { if (txt.value.trim()) decide(name, "keep", txt.value.trim(), msg); });
  });
  const o = S.overlap, d = (S.decisions["(overlap)"]||{}).decision;
  document.getElementById("ov").innerHTML =
    `<div>Of the ${o.total.toLocaleString()} crops kept now, <b>${o.in_v4.toLocaleString()}</b> are already in v4's training data and <b>${o.new.toLocaleString()}</b> are new to v4 (counts before this round's choices).</div>
     <label class="opt"><input type="radio" name="ov" value="new_only" ${d==="new_only"?"checked":""}><div><div class="big">Only the new crops</div><div class="src">the retraining folder holds only crops v4 has never seen; no duplicates</div></div></label>
     <label class="opt"><input type="radio" name="ov" value="all_flagged" ${d==="all_flagged"?"checked":""}><div><div class="big">All crops, flagged</div><div class="src">every kept crop, with a column saying which are already in v4_train</div></div></label>
     <div class="saved" id="ovMsg"></div>`;
  document.querySelectorAll("input[name=ov]").forEach(r => r.addEventListener("change", () => decide("(overlap)", r.value, "", document.getElementById("ovMsg"))));
  progress(); mark();
}
load();
</script></body></html>"""


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "this-machine-ip"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8096)
    a = ap.parse_args()
    build()
    print(f"Review round 2: {len(DATA['items'])} crops + 1 overlap choice   decisions -> {os.path.relpath(DECISIONS, ROOT)}")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"Open  http://127.0.0.1:{a.port}")
    if a.host == "0.0.0.0":
        print(f"LAN   http://{lan_ip()}:{a.port}")
    print("Stop with Ctrl+C")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
