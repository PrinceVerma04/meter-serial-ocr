#!/usr/bin/env python3
"""
Meter OCR dashboard — upload one image or a whole folder and see the read of every model.

    .venv/bin/python dashboard/app.py                 # http://127.0.0.1:8095
    .venv/bin/python dashboard/app.py --host 0.0.0.0  # also reachable from the LAN
    .venv/bin/python dashboard/app.py --port 9000

Models (whichever files exist): v3 FP32, v3 PTQ (deployed), v4 FP32, v4 PTQ, v4.2 FP32, v4.2 PTQ.
All run as ONNX on CPU through the deploy code (deploy/ptq_int8/meterocr.py): same preprocessing,
CTC decode and 0.95 accept gate as production. No extra packages needed beyond the project .venv
(onnxruntime, opencv, numpy). Images are processed in memory and never written to disk.
"""
import argparse
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "deploy", "ptq_int8"))
from meterocr import ACCEPT_THRESHOLD, DICT, MeterOCR  # noqa: E402

# key, label, file, note
MODELS = [
    ("v3_fp32", "v3 FP32", "models/v3/meterocr_v3_fp32.onnx", ""),
    ("v3_ptq", "v3 PTQ", "models/v3/meterocr_v3_int8_ptq.onnx", "deployed"),
    ("v4_fp32", "v4 FP32", "models/v4/meterocr_v4_fp32.onnx", ""),
    ("v4_ptq", "v4 PTQ", "models/v4/meterocr_v4_int8_ptq.onnx", ""),
    ("v4.2_fp32", "v4.2 FP32", "models/v4.2/meterocr_v4.2_fp32.onnx", ""),
    ("v4.2_ptq", "v4.2 PTQ", "models/v4.2/meterocr_v4.2_int8_ptq.onnx", "candidate"),
]
MAX_BYTES = 25 * 1024 * 1024  # per image


class Reader(MeterOCR):
    """MeterOCR with a thread-limited ONNX session (decode/preprocess/gate inherited unchanged)."""

    def __init__(self, model_path, threads):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        chars = [l.rstrip("\n") for l in open(DICT, encoding="utf-8")]
        self.character = ["blank"] + chars + [" "]


READERS = {}
MODEL_INFO = []
GATE = None  # limits how many images are inferred at once


def load_models(workers):
    threads = max(1, (os.cpu_count() or 4) // workers)
    dummy = np.full((48, 200, 3), 255, np.uint8)
    for key, label, rel, note in MODELS:
        path = os.path.join(ROOT, rel)
        ok = os.path.isfile(path)
        if ok:
            t = time.perf_counter()
            READERS[key] = Reader(path, threads)
            READERS[key].predict(dummy)  # warm-up
            print(f"  loaded {label:<12} {os.path.getsize(path) / 1e6:5.1f} MB  "
                  f"({time.perf_counter() - t:.1f}s)  {rel}")
        else:
            print(f"  missing {label:<11} {rel}  (skipped)")
        MODEL_INFO.append({"key": key, "label": label, "file": rel, "note": note, "available": ok})
    if not READERS:
        sys.exit("No model files found under models/ — nothing to serve.")
    print(f"  {workers} images at a time x {threads} CPU threads per model")


class Handler(BaseHTTPRequestHandler):
    server_version = "MeterOCRDashboard/1.0"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/models":
            self._json(200, {"models": MODEL_INFO, "threshold": ACCEPT_THRESHOLD})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/read":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            return self._json(400, {"error": "empty upload"})
        if n > MAX_BYTES:
            return self._json(413, {"error": f"image larger than {MAX_BYTES // 2**20} MB"})
        data = self.rfile.read(n)
        wanted = parse_qs(url.query).get("models", [""])[0].split(",")
        keys = [k for k in wanted if k in READERS] or list(READERS)

        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return self._json(200, {"error": "not a readable image"})
        reads = {}
        with GATE:
            for k in keys:
                t = time.perf_counter()
                text, score, status = READERS[k].predict(bgr)
                reads[k] = {"text": text, "conf": score, "status": status,
                            "ms": round((time.perf_counter() - t) * 1000, 1)}
        self._json(200, {"w": int(bgr.shape[1]), "h": int(bgr.shape[0]), "reads": reads})


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
    global GATE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other machines on the LAN")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--workers", type=int, default=8, help="images inferred at the same time")
    a = ap.parse_args()

    print("Meter OCR dashboard — loading models")
    load_models(a.workers)
    GATE = threading.BoundedSemaphore(a.workers)

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"\nOpen  http://127.0.0.1:{a.port}")
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
