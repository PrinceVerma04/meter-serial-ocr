#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meter serial OCR — HTTP API. PYTHON 3.8 COMPATIBLE.

Endpoints
    GET  /health              -> {"status":"ok", ...}          (ALB/ELB health check)
    POST /ocr                 -> single image, multipart field "image" OR raw body
    POST /ocr/batch           -> many images, repeated multipart field "images"

    curl -F image=@crop.jpg http://localhost:8080/ocr

Run (production):
    python3 app.py --host 0.0.0.0 --port 8080

Serving: uses gevent's WSGIServer if gevent is installed, else waitress, else
falls back to the Flask dev server with a warning. All three work on 3.8.

The model is loaded ONCE at import, so every request reuses the warm session
rather than paying ~1 s of setup.
"""
from __future__ import absolute_import, division, print_function

import os
import argparse
import time

from flask import Flask, request, jsonify

from meterocr import MeterOCR, ACCEPT_THRESHOLD

try:                                  # optional — only needed for browser callers
    from flask_cors import CORS
except ImportError:
    CORS = None

MAX_MB = int(os.environ.get("METEROCR_MAX_MB", "10"))
THREADS = int(os.environ.get("METEROCR_THREADS", "2"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
if CORS is not None:
    CORS(app)

# One session for the whole process. ORT run() is thread-safe.
OCR = MeterOCR(threads=THREADS)


def _read_one(blob, name=""):
    t0 = time.time()
    serial, score, status = OCR.predict(blob)
    return {
        "filename": name,
        "serial": serial,
        "confidence": score,
        "status": status,                     # accepted | review | blank
        "ms": round((time.time() - t0) * 1000, 1),
    }


# Flask 1.x has no @app.get/@app.post shortcuts, so use the explicit form.
# This works on every Flask version from 1.0 through 3.x.
@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", model="meterocr_ptq_int8",
                   threshold=ACCEPT_THRESHOLD, threads=THREADS)


@app.route("/ocr", methods=["POST"])
def ocr_one():
    f = request.files.get("image")
    blob = f.read() if f else request.get_data()
    if not blob:
        return jsonify(error="no image: send multipart field 'image' or a raw image body"), 400
    return jsonify(_read_one(blob, f.filename if f else ""))


@app.route("/ocr/batch", methods=["POST"])
def ocr_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify(error="no images: send repeated multipart field 'images'"), 400
    results = [_read_one(f.read(), f.filename) for f in files]
    accepted = sum(1 for r in results if r["status"] == "accepted")
    return jsonify(count=len(results), accepted=accepted, results=results)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="image too large (limit %d MB)" % MAX_MB), 413


def serve(host, port):
    """Pick the best available production server; all options support 3.8."""
    try:
        from gevent.pywsgi import WSGIServer
        print("meterocr serving on http://%s:%d  (gevent, %d ORT threads)"
              % (host, port, THREADS), flush=True)
        WSGIServer((host, port), app).serve_forever()
        return
    except ImportError:
        pass

    try:
        from waitress import serve as waitress_serve
        print("meterocr serving on http://%s:%d  (waitress, %d ORT threads)"
              % (host, port, THREADS), flush=True)
        waitress_serve(app, host=host, port=port, threads=4)
        return
    except ImportError:
        pass

    print("WARNING: neither gevent nor waitress found — using the Flask dev "
          "server. Fine for a smoke test, NOT for production traffic.", flush=True)
    app.run(host=host, port=port, debug=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--dev", action="store_true", help="force the Flask dev server")
    args = ap.parse_args()

    if args.dev:
        app.run(host=args.host, port=args.port, debug=False)
    else:
        serve(args.host, args.port)


if __name__ == "__main__":
    main()
