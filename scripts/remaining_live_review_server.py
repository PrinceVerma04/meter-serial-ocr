#!/usr/bin/env python3
"""
LIVE (server-backed), single-image, KEYBOARD-driven review tool for whatever
is still left on the main TPDDL dashboard (everything not yet taken out to
all3_agree_eq_bot_tpddl/ and not yet discarded) -> currently 828 images,
almost entirely the hardest bucket where all 3 models disagree with the bot.

Every action writes to disk IMMEDIATELY (no download/merge step):
  - Keep     -> saves the BOT's value as the label, copies the image + appends
                `filename, ground truth` to
                Some_analysis_part/all3_agree_eq_bot_tpddl/ground_truth.csv
  - Write    -> same as Keep but with whatever true label you typed
  - Discard  -> appends the filename to Some_analysis_part/discarded_tpddl.csv
                and does NOT add it to all3_agree_eq_bot_tpddl/ at all.
In every case the image is gone from this tool for good, immediately.

Keyboard: K = keep & advance, D = discard & advance, W = focus the write box
(Enter in the box saves & advances), LeftArrow/RightArrow or P/N = navigate
without judging, S = skip.

No Flask needed (not installed on this box) — pure stdlib http.server.

Usage:
    python3 scripts/remaining_live_review_server.py [--port 8813] [--host 0.0.0.0]
Then open http://<this-machine-ip>:8813/
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


def taken_out_count():
    return len(load_taken_out())


def load_remaining():
    """Recomputed fresh from disk every time -> always reflects reality."""
    with open(SRC_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    taken_out = load_taken_out()
    discarded = load_discarded()
    out = []
    for r in rows:
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
            return
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
<title>TPDDL — LIVE review: remaining</title>
<style>
:root{color-scheme:light dark;
 --bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--mut:#52514e;--line:#e6e5e2;
 --good:#0ca30c;--goodink:#006300;--bad:#d03b3b;--badink:#8a1f1f;--warn:#fab219;--warnink:#7a5200;
 --accent:#2f6fed;}
@media(prefers-color-scheme:dark){:root{
 --bg:#141413;--card:#1f1f1e;--ink:#f5f5f2;--mut:#b4b3ab;--line:#33322f;
 --good:#3ec13e;--goodink:#8ff08f;--bad:#ef6a6a;--badink:#ffb0b0;--warn:#fab219;--warnink:#ffd98a;
 --accent:#6c9bff;}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
 background:var(--bg);color:var(--ink);padding:20px clamp(14px,4vw,48px) 60px}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 18px;font-size:13.5px}
.live{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;
 background:color-mix(in srgb,var(--good) 22%,transparent);color:var(--goodink);
 padding:2px 9px;border-radius:20px;margin-bottom:10px}

.topbar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:18px}
.pbar{flex:1;min-width:220px;height:10px;border-radius:6px;background:var(--line);overflow:hidden}
.pbar i{display:block;height:100%;background:var(--accent)}
.stats{display:flex;gap:16px;font-size:13px;color:var(--mut);flex-wrap:wrap}
.stats b{color:var(--ink)}
button{font:inherit;background:var(--card);color:var(--ink);
 border:1px solid var(--line);border-radius:8px;padding:8px 16px;cursor:pointer}
button.good{background:var(--good);color:#fff;border-color:var(--good)}
button.bad{background:var(--bad);color:#fff;border-color:var(--bad)}
button:disabled{opacity:.4;cursor:default}

.wrap{display:grid;grid-template-columns:minmax(280px,460px) 1fr;gap:26px;align-items:start}
@media(max-width:820px){.wrap{grid-template-columns:1fr}}
.imgcard{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;text-align:center}
.imgcard img{max-width:100%;max-height:320px;background:#000;border-radius:8px;cursor:zoom-in;object-fit:contain}
.namerow{display:flex;align-items:center;gap:8px;margin-top:12px}
.namerow code{flex:1;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--mut);
 white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:left}
.cp{flex:none;font-size:11px;padding:4px 10px;border-radius:6px;border:1px solid var(--line);
 background:var(--bg);color:var(--ink);cursor:pointer}
.cp.done{background:var(--good);color:#fff;border-color:var(--good)}

.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
.kv{display:flex;justify-content:space-between;gap:10px;padding:9px 0;border-bottom:1px solid var(--line)}
.kv:last-of-type{border-bottom:none}
.kv .k{color:var(--mut);font-size:13px}
.kv .v{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:15px;font-weight:600}
.kv.bot .v{color:var(--warnink)}

.actions{display:flex;gap:10px;margin:18px 0 12px;flex-wrap:wrap}
.correction{display:flex;gap:8px;margin-bottom:14px}
.correction input{flex:1;font:inherit;font-family:ui-monospace,Menlo,Consolas,monospace;
 padding:10px 12px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.status{font-size:13px;padding:8px 12px;border-radius:8px;margin-bottom:14px;display:none}
.status.keep{display:block;background:color-mix(in srgb,var(--good) 16%,transparent);color:var(--goodink)}
.status.discard{display:block;background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--badink)}
.status.written{display:block;background:color-mix(in srgb,var(--accent) 16%,transparent);color:var(--accent)}
.nav{display:flex;gap:10px;justify-content:space-between;margin-top:16px}
.keys{margin-top:20px;font-size:12px;color:var(--mut);line-height:1.9}
.keys kbd, .kv .k kbd{background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:1px 7px;
 font-family:ui-monospace,Menlo,Consolas,monospace}
.kv .k kbd{margin-right:6px}
.kv.pickable{cursor:pointer}
.kv.pickable:hover{background:color-mix(in srgb,var(--accent) 10%,transparent);border-radius:6px}

.lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:99;
 align-items:center;justify-content:center;cursor:zoom-out;padding:30px}
.lightbox.on{display:flex}
.lightbox img{max-width:100%;max-height:100%;border-radius:8px}
.empty{padding:60px 20px;text-align:center;color:var(--mut)}
</style></head><body>
<div class=lightbox id=lightbox><img id=lightboximg src="" alt=""></div>

<div class=live>live &mdash; writes to disk instantly, no download/merge step</div>
<h1>TPDDL July set &mdash; review: remaining images</h1>
<p class=sub>Everything not yet taken out or discarded. Keep = accept the bot's value. Write = type the true value if the bot is wrong. Discard = drop it (not saved anywhere).</p>

<div class=topbar>
 <div class=pbar><i id=pbarfill style="width:0%"></i></div>
 <div class=stats>
  <span><b id=stRemaining>__N__</b> remaining here</span>
  <span class=good><b id=stKeep>0</b> kept this session</span>
  <span class=bad><b id=stDiscard>0</b> discarded this session</span>
  <span><b id=stWritten>0</b> written this session</span>
  <span><b id=stTotal>__TOTAL__</b> total labeled in all3_agree_eq_bot_tpddl/</span>
 </div>
</div>

<div class=wrap>
 <div class=imgcard>
  <img id=img src="" alt="" title="click to zoom">
  <div class=namerow><code id=fname></code><button class=cp id=cpBtn>copy</button></div>
 </div>

 <div class=panel>
  <div class="kv bot pickable" id=kvBot title="click, or press 1, to save this as the label"><span class=k><kbd>1</kbd> bot reading</span><span class=v id=vBot></span></div>
  <div class="kv pickable" id=kvFp32 title="click, or press 2, to save this as the label"><span class=k><kbd>2</kbd> fp32</span><span class=v id=vFp32></span></div>
  <div class="kv pickable" id=kvPtq3 title="click, or press 3, to save this as the label"><span class=k><kbd>3</kbd> ptq_v3</span><span class=v id=vPtq3></span></div>
  <div class="kv pickable" id=kvPtq2 title="click, or press 4, to save this as the label"><span class=k><kbd>4</kbd> ptq_v2</span><span class=v id=vPtq2></span></div>

  <div class=status id=statusBox></div>

  <div class=actions>
   <button class=good id=btnKeep>&#10003; Keep (=bot) <kbd>K</kbd></button>
   <button class=bad id=btnDiscard>&#10007; Discard <kbd>D</kbd></button>
  </div>
  <div class=correction>
   <input id=writeInput type=text placeholder="write the true label, then Enter" autocomplete=off>
   <button id=saveWrite>Save <kbd>&crarr;</kbd></button>
  </div>

  <div class=nav>
   <button id=prevBtn>&larr; Prev <kbd>P</kbd></button>
   <button id=skipBtn>Skip <kbd>S</kbd></button>
   <button id=nextBtn>Next &rarr; <kbd>N</kbd></button>
  </div>

  <div class=keys>
   <div><kbd>1</kbd> save the <b>bot</b> value as the label &amp; advance</div>
   <div><kbd>2</kbd> save the <b>fp32</b> value as the label &amp; advance</div>
   <div><kbd>3</kbd> save the <b>ptq_v3</b> value as the label &amp; advance</div>
   <div><kbd>4</kbd> save the <b>ptq_v2</b> value as the label &amp; advance</div>
   <div><kbd>K</kbd> same as <kbd>1</kbd> (keep = bot value) &amp; advance</div>
   <div><kbd>D</kbd> discard (excluded permanently, not saved anywhere) &amp; advance</div>
   <div><kbd>W</kbd> focus the write box to type a different true label</div>
   <div><kbd>Enter</kbd> (in box) save written label to disk immediately &amp; advance</div>
   <div><kbd>&larr;</kbd>/<kbd>&rarr;</kbd> or <kbd>P</kbd>/<kbd>N</kbd> navigate (view only, no write)</div>
   <div><kbd>S</kbd> skip without judging</div>
  </div>
 </div>
</div>

<script>
let DATA = __DATA__, sessTotal = __TOTAL__;
const DIR = "/images/";
let idx = 0;
let sessKeep = 0, sessDiscard = 0, sessWritten = 0;

const els = {
  img: document.getElementById('img'), fname: document.getElementById('fname'),
  vBot: document.getElementById('vBot'), vFp32: document.getElementById('vFp32'),
  vPtq3: document.getElementById('vPtq3'), vPtq2: document.getElementById('vPtq2'),
  statusBox: document.getElementById('statusBox'),
  writeInput: document.getElementById('writeInput'),
  pbarfill: document.getElementById('pbarfill'),
  stRemaining: document.getElementById('stRemaining'), stKeep: document.getElementById('stKeep'),
  stDiscard: document.getElementById('stDiscard'), stWritten: document.getElementById('stWritten'),
  stTotal: document.getElementById('stTotal'),
  lightbox: document.getElementById('lightbox'), lightboximg: document.getElementById('lightboximg'),
  btnKeep: document.getElementById('btnKeep'), btnDiscard: document.getElementById('btnDiscard'),
  saveWrite: document.getElementById('saveWrite'),
};

function updateStats(startN){
  els.stRemaining.textContent = DATA.length;
  els.stKeep.textContent = sessKeep;
  els.stDiscard.textContent = sessDiscard;
  els.stWritten.textContent = sessWritten;
  els.stTotal.textContent = sessTotal;
  els.pbarfill.style.width = (startN ? 100*(startN-DATA.length)/startN : 0)+'%';
}
const START_N = DATA.length;

function render(){
  if(DATA.length === 0){
    document.querySelector('.wrap').innerHTML = '<div class=empty>All reviewed &mdash; nothing left.</div>';
    return;
  }
  if(idx >= DATA.length) idx = DATA.length - 1;
  const d = DATA[idx];
  els.img.src = DIR + encodeURIComponent(d.img);
  els.fname.textContent = d.img; els.fname.title = d.img;
  els.vBot.textContent = d.bot || '∅';
  els.vFp32.textContent = d.fp32 || '∅';
  els.vPtq3.textContent = d.ptq_v3 || '∅';
  els.vPtq2.textContent = d.ptq_v2 || '∅';
  els.writeInput.value = '';
  els.statusBox.className = 'status';
  els.statusBox.textContent = '';
  document.title = `(${DATA.length} left) TPDDL live review`;
}

function goto(i){ idx = Math.max(0, Math.min(DATA.length-1, i)); render(); }
function next(){ goto(idx); } // after removal, idx already points to the next item
function prev(){ goto(idx-1); }

async function callAction(image, action, label){
  const res = await fetch('/action', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({image, action, label: label||''})
  });
  if(!res.ok) throw new Error(await res.text());
  return res.json();
}

async function resolve(action, label){
  const d = DATA[idx];
  const btns = [els.btnKeep, els.btnDiscard, els.saveWrite];
  btns.forEach(b=>b.disabled = true);
  try{
    const r = await callAction(d.img, action, label);
    if(action==='keep') sessKeep++;
    else if(action==='write'){ sessWritten++; }
    else if(action==='discard') sessDiscard++;
    if(action!=='discard') sessTotal = r.total_labeled ?? (sessTotal+1);
    DATA.splice(idx, 1);
    updateStats(START_N);
    render();
  }catch(e){
    alert('Failed: ' + e.message);
  }finally{
    btns.forEach(b=>b.disabled = false);
  }
}

els.btnKeep.onclick = ()=>{ const d=DATA[idx]; resolve('keep', d.bot); };
els.btnDiscard.onclick = ()=>resolve('discard', '');
els.saveWrite.onclick = ()=>{ const v=els.writeInput.value.trim(); if(!v) return; resolve('write', v); };

// pick any of the 4 shown values directly (click or number key 1-4)
function pick(field){
  const d = DATA[idx];
  const val = d[field];
  if(!val){ alert('That field is blank for this image.'); return; }
  resolve(field === 'bot' ? 'keep' : 'write', val);
}
document.getElementById('kvBot').onclick = ()=>pick('bot');
document.getElementById('kvFp32').onclick = ()=>pick('fp32');
document.getElementById('kvPtq3').onclick = ()=>pick('ptq_v3');
document.getElementById('kvPtq2').onclick = ()=>pick('ptq_v2');

document.getElementById('prevBtn').onclick = prev;
document.getElementById('nextBtn').onclick = ()=>goto(idx+1);
document.getElementById('skipBtn').onclick = ()=>goto(idx+1);
document.getElementById('cpBtn').onclick = ()=>{
  const btn = document.getElementById('cpBtn');
  navigator.clipboard.writeText(DATA[idx].img).then(()=>{
    btn.textContent='✓'; btn.classList.add('done');
    setTimeout(()=>{btn.textContent='copy'; btn.classList.remove('done');},1200);
  });
};
els.img.onclick = ()=>{ els.lightboximg.src = els.img.src; els.lightbox.classList.add('on'); };
els.lightbox.onclick = ()=>{ els.lightbox.classList.remove('on'); els.lightboximg.src=''; };

els.writeInput.addEventListener('keydown', e=>{
  if(e.key==='Enter'){ e.preventDefault(); els.saveWrite.click(); }
});

document.addEventListener('keydown', e=>{
  if(document.activeElement === els.writeInput) return;
  switch(e.key){
    case '1': pick('bot'); break;
    case '2': pick('fp32'); break;
    case '3': pick('ptq_v3'); break;
    case '4': pick('ptq_v2'); break;
    case 'k': case 'K': els.btnKeep.click(); break;
    case 'd': case 'D': els.btnDiscard.click(); break;
    case 'w': case 'W': els.writeInput.focus(); break;
    case 'ArrowRight': case 'n': case 'N': goto(idx+1); break;
    case 'ArrowLeft': case 'p': case 'P': prev(); break;
    case 's': case 'S': goto(idx+1); break;
  }
});

updateStats(START_N);
render();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

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
            total = taken_out_count()
            html = (PAGE_TPL.replace("__DATA__", json.dumps(data))
                    .replace("__N__", str(len(data)))
                    .replace("__TOTAL__", str(total)))
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
            self._send_json({"ok": True, "total_labeled": taken_out_count()})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, status=400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8813)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    os.makedirs(TAKEN_OUT_IMG_DIR, exist_ok=True)
    n = len(load_remaining())
    print(f"Remaining: {n}")
    print(f"Serving on http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
