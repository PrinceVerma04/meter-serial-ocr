#!/usr/bin/env python3
"""
LIVE (server-backed) review dashboard for the TPDDL July-set "All 3 agree"
filter (fp32 == ptq_v3 == ptq_v2), currently 2,312 images.

Unlike the earlier static dashboards (which only save to browser localStorage
and need a manual CSV download + merge step), this one writes to disk
IMMEDIATELY on every click:
  - Keep / Write  -> copies the image + appends `filename, ground truth` to
                     Some_analysis_part/all3_agree_eq_bot_tpddl/ground_truth.csv
                     right away, and the image disappears from the dashboard
                     for good (recomputed fresh from disk on every request).
  - Discard       -> appends the filename to Some_analysis_part/discarded_tpddl.csv
                     right away and the image disappears for good too.

No Flask needed (not installed on this box) — pure stdlib http.server.

Usage:
    python3 scripts/all3agree_live_server.py [--port 8812] [--host 0.0.0.0]
Then open http://<this-machine-ip>:8812/
"""
import argparse
import csv
import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "Some_analysis_part")
SRC_CSV = os.path.join(BASE, "merged_3models_vs_bot_tpddl.csv")
IMG_DIR = os.path.join(BASE, "1stjuly_4thjuly_2026_tpddl")
TAKEN_OUT_DIR = os.path.join(BASE, "all3_agree_eq_bot_tpddl")
TAKEN_OUT_IMG_DIR = os.path.join(TAKEN_OUT_DIR, "images")
TAKEN_OUT_CSV = os.path.join(TAKEN_OUT_DIR, "ground_truth.csv")
DISCARDED_CSV = os.path.join(BASE, "discarded_tpddl.csv")

LOCK = threading.Lock()


def load_taken_out():
    with open(TAKEN_OUT_CSV, newline="", encoding="utf-8") as f:
        return {r["filename"] for r in csv.DictReader(f)}


def load_discarded():
    if not os.path.exists(DISCARDED_CSV):
        return set()
    with open(DISCARDED_CSV, newline="", encoding="utf-8") as f:
        return {r["filename"] for r in csv.DictReader(f)}


def load_remaining():
    """Recomputed fresh from disk every time -> always reflects reality."""
    with open(SRC_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    taken_out = load_taken_out()
    discarded = load_discarded()
    out = []
    for r in rows:
        if r["all_agree"] != "yes":
            continue
        name = r["image_name"]
        if name in taken_out or name in discarded:
            continue
        out.append({
            "img": name,
            "fp32": r["fp32"], "ptq_v3": r["ptq_v3"], "ptq_v2": r["ptq_v2"],
            "bot": r["bot_serial"],
        })
    return out


def do_keep(image_name, label):
    with LOCK:
        taken_out = load_taken_out()
        if image_name in taken_out:
            return  # already handled (double click / race) - no-op
        src = os.path.join(IMG_DIR, image_name)
        if not os.path.exists(src):
            raise FileNotFoundError(image_name)
        shutil.copy(src, os.path.join(TAKEN_OUT_IMG_DIR, image_name))
        with open(TAKEN_OUT_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([image_name, label])


def do_discard(image_name):
    with LOCK:
        discarded = load_discarded()
        if image_name in discarded:
            return
        write_header = not os.path.exists(DISCARDED_CSV)
        with open(DISCARDED_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["filename"])
            w.writerow([image_name])


PAGE_TPL = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>TPDDL — LIVE review: All 3 agree</title>
<style>
:root{color-scheme:light dark;
 --bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--mut:#52514e;--line:#e6e5e2;
 --good:#0ca30c;--goodink:#006300;--bad:#d03b3b;--warn:#fab219;--warnink:#7a5200;--accent:#2f6fed;}
@media(prefers-color-scheme:dark){:root{
 --bg:#141413;--card:#1f1f1e;--ink:#f5f5f2;--mut:#b4b3ab;--line:#33322f;
 --good:#3ec13e;--goodink:#8ff08f;--bad:#ef6a6a;--warn:#fab219;--warnink:#ffd98a;--accent:#6c9bff;}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
 background:var(--bg);color:var(--ink);padding:28px clamp(14px,4vw,48px)}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 18px;font-size:13.5px}
.live{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;
 background:color-mix(in srgb,var(--good) 22%,transparent);color:var(--goodink);
 padding:2px 9px;border-radius:20px;margin-bottom:10px}
.reviewbar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;background:var(--card);
 border:1px solid var(--line);border-radius:14px;padding:14px 18px;margin-bottom:18px}
.reviewbar .n{font-size:22px;font-weight:700}
.reviewbar .l{font-size:12px;color:var(--mut)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px}
.card{background:var(--card);border:2px solid var(--line);border-radius:12px;overflow:hidden;transition:opacity .2s,transform .2s}
.card.removing{opacity:0;transform:scale(.9)}
.card img{width:100%;height:160px;object-fit:contain;background:#000;display:block;cursor:zoom-in}
.card .b{padding:10px 12px;font-size:12.5px}
.namerow{display:flex;align-items:center;gap:6px;margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.namerow code{flex:1;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:10.5px;color:var(--mut);
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cp{flex:none;font-size:11px;padding:3px 9px;border-radius:6px;border:1px solid var(--line);
 background:var(--bg);color:var(--ink);cursor:pointer}
.cp.done{background:var(--good);color:#fff;border-color:var(--good)}
.row{display:flex;justify-content:space-between;gap:8px;margin:2px 0}
.row .k{color:var(--mut)}.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.row.agreed{background:color-mix(in srgb,var(--good) 12%,transparent);border-radius:6px;padding:4px 6px;margin:4px -6px}
.row.agreed .k{color:var(--goodink);font-weight:600}
.revrow{display:flex;gap:6px;margin-top:9px;padding-top:9px;border-top:1px solid var(--line)}
.revrow button{flex:1;font:inherit;font-size:11px;padding:6px 6px;border-radius:6px;cursor:pointer;
 background:var(--bg);color:var(--ink);border:1px solid var(--line)}
.revrow button.keep{border-color:var(--good);color:var(--goodink)}
.revrow button.keep:hover{background:var(--good);color:#fff}
.revrow button.discard{border-color:var(--bad);color:var(--bad)}
.revrow button.discard:hover{background:var(--bad);color:#fff}
.revrow button:disabled{opacity:.5;cursor:default}
.writerow{display:flex;gap:6px;margin-top:6px}
.writerow input{flex:1;font:inherit;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;
 padding:5px 8px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.writerow button{font-size:11px;padding:5px 9px;border-radius:6px;border:1px solid var(--line);
 background:var(--bg);color:var(--ink);cursor:pointer}
.lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:99;
 align-items:center;justify-content:center;cursor:zoom-out;padding:30px}
.lightbox.on{display:flex}
.lightbox img{max-width:100%;max-height:100%;border-radius:8px}
.empty{padding:60px 20px;text-align:center;color:var(--mut)}
</style></head><body>
<div class=lightbox id=lightbox><img id=lightboximg src="" alt=""></div>

<div class=live>live &mdash; writes to disk instantly</div>
<h1>TPDDL July set &mdash; All 3 agree (fp32 == ptq_v3 == ptq_v2)</h1>
<p class=sub>Keep/Write saves this image + label straight into <span class=mono>all3_agree_eq_bot_tpddl/</span> immediately. Discard removes it from this dashboard permanently, immediately. No download/merge step needed.</p>

<div class=reviewbar>
 <div><div class=n id=remainingN>__N__</div><div class=l>remaining</div></div>
</div>

<div class=grid id=grid></div>
<div class=empty id=emptyMsg style="display:none">All reviewed — nothing left in this filter.</div>

<script>
let DATA = __DATA__;
const grid = document.getElementById('grid'), remainingN = document.getElementById('remainingN'),
      emptyMsg = document.getElementById('emptyMsg');
const lightbox = document.getElementById('lightbox'), lightboximg = document.getElementById('lightboximg');
lightbox.onclick = ()=>{ lightbox.classList.remove('on'); lightboximg.src=''; };

function esc(s){return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))||"<em>&empty;</em>";}

async function callAction(image, action, label){
  const res = await fetch('/action', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({image, action, label: label||''})
  });
  if(!res.ok) throw new Error(await res.text());
  return res.json();
}

function renderAll(){
  grid.innerHTML = '';
  remainingN.textContent = DATA.length;
  emptyMsg.style.display = DATA.length ? 'none' : 'block';
  for(const d of DATA) grid.appendChild(makeCard(d));
}

function makeCard(d){
  const src = '/images/' + encodeURIComponent(d.img);
  const c = document.createElement('div'); c.className = 'card';
  c.innerHTML = `<img loading=lazy src="${src}" alt="" title="click to zoom">
   <div class=b>
    <div class=namerow><code title="${d.img}">${d.img}</code><button class=cp>copy</button></div>
    <div class=row><span class=k>bot</span><span class=mono>${esc(d.bot)}</span></div>
    <div class="row agreed"><span class=k>fp32/ptq_v3/ptq_v2</span><span class=mono>${esc(d.fp32)}</span></div>
    <div class=revrow>
     <button class=keep>&#10003; Keep (use ${esc(d.fp32)})</button>
     <button class=discard>&#10007; Discard</button>
    </div>
    <div class=writerow>
     <input type=text placeholder="write true label">
     <button class=save>Save</button>
    </div>
   </div>`;
  c.querySelector('img').onclick = ()=>{ lightboximg.src = src; lightbox.classList.add('on'); };
  const cpBtn = c.querySelector('.cp');
  cpBtn.onclick = ()=>{ navigator.clipboard.writeText(d.img).then(()=>{
    cpBtn.textContent='✓'; cpBtn.classList.add('done');
    setTimeout(()=>{cpBtn.textContent='copy'; cpBtn.classList.remove('done');},1200); }); };

  const keepBtn = c.querySelector('.keep'), discardBtn = c.querySelector('.discard');
  const input = c.querySelector('.writerow input'), saveBtn = c.querySelector('.writerow .save');
  const allBtns = [keepBtn, discardBtn, saveBtn];

  async function resolve(action, label){
    allBtns.forEach(b=>b.disabled = true);
    try{
      await callAction(d.img, action, label);
      c.classList.add('removing');
      setTimeout(()=>{
        c.remove();
        DATA = DATA.filter(x=>x.img !== d.img);
        remainingN.textContent = DATA.length;
        emptyMsg.style.display = DATA.length ? 'none' : 'block';
      }, 200);
    }catch(e){
      allBtns.forEach(b=>b.disabled = false);
      alert('Failed: ' + e.message);
    }
  }
  keepBtn.onclick = ()=>resolve('keep', d.fp32); // all 3 models agree in this filter -> use their shared value, not the bot's
  discardBtn.onclick = ()=>resolve('discard', '');
  saveBtn.onclick = ()=>{ const v=input.value.trim(); if(!v) return; resolve('write', v); };
  input.addEventListener('keydown', e=>{ if(e.key==='Enter'){ e.preventDefault(); saveBtn.click(); } });
  return c;
}

renderAll();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            data = load_remaining()
            html = PAGE_TPL.replace("__DATA__", json.dumps(data)).replace("__N__", str(len(data)))
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/images/"):
            name = path[len("/images/"):]
            fp = os.path.join(IMG_DIR, name)
            if not os.path.abspath(fp).startswith(os.path.abspath(IMG_DIR)) or not os.path.exists(fp):
                self.send_response(404); self.end_headers(); return
            with open(fp, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path != "/action":
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image = payload["image"]
            action = payload["action"]
            label = payload.get("label", "")
            if action in ("keep", "write"):
                if not label:
                    raise ValueError("empty label")
                do_keep(image, label)
            elif action == "discard":
                do_discard(image)
            else:
                raise ValueError("bad action")
            self._send_json({"ok": True})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8812)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    os.makedirs(TAKEN_OUT_IMG_DIR, exist_ok=True)
    n = len(load_remaining())
    print(f"All-3-agree remaining: {n}")
    print(f"Serving on http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
