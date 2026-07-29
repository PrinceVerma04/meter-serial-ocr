#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drop-in replacement for meterocr.py that talks to the OCR service over HTTP.

Why this exists
---------------
The OCR model needs numpy>=1.21.6 (onnxruntime). TensorFlow 2.4.1 needs
numpy~=1.19.2. They cannot share one interpreter. So the model runs in its own
venv as a local service, and this shim — which imports ONLY requests + cv2,
both already present in system Python — forwards calls to it.

The API is identical to the real meterocr.MeterOCR, so existing code keeps
working unchanged:

    from meterocr import MeterOCR
    ocr = MeterOCR()
    serial, conf, status = ocr.predict("crop.jpg")

Install as:  ~/flaskML/meterocr.py
Service at:  METEROCR_URL (default http://127.0.0.1:8080)
"""
from __future__ import absolute_import, division, print_function

import os

import requests

URL = os.environ.get("METEROCR_URL", "http://127.0.0.1:8080").rstrip("/")
TIMEOUT = float(os.environ.get("METEROCR_TIMEOUT", "30"))
ACCEPT_THRESHOLD = float(os.environ.get("METEROCR_THRESHOLD", "0.95"))


class MeterOCRError(RuntimeError):
    pass


class MeterOCR(object):
    """Same surface as the real MeterOCR; the work happens in the service."""

    def __init__(self, model_path=None, dict_path=None, providers=None,
                 threads=None, url=URL, timeout=TIMEOUT):
        # model_path/dict_path/providers/threads are accepted and ignored so
        # this is a literal drop-in for the local class.
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health(self):
        r = requests.get(self.url + "/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, blob, filename="image.jpg"):
        try:
            r = requests.post(self.url + "/ocr",
                              files={"image": (filename, blob, "image/jpeg")},
                              timeout=self.timeout)
        except requests.RequestException as e:
            raise MeterOCRError(
                "OCR service unreachable at %s (%s). Is it running? "
                "Check: pm2 logs meterocr" % (self.url, e))
        if r.status_code != 200:
            raise MeterOCRError("OCR service returned %d: %s"
                                % (r.status_code, r.text[:200]))
        return r.json()

    def predict(self, image, threshold=None, clean=True):
        """image: path, bytes, or BGR ndarray -> (serial, confidence, status)."""
        name = "image.jpg"

        if isinstance(image, str):
            if not os.path.exists(image):
                return "", 0.0, "blank"
            name = os.path.basename(image)
            with open(image, "rb") as fh:
                blob = fh.read()
        elif isinstance(image, (bytes, bytearray)):
            blob = bytes(image)
        else:
            # assume a BGR ndarray from cv2
            import cv2
            ok, buf = cv2.imencode(".jpg", image)
            if not ok:
                return "", 0.0, "blank"
            blob = buf.tobytes()

        d = self._post(blob, name)
        serial = d.get("serial", "")
        score = float(d.get("confidence", 0.0) or 0.0)
        status = d.get("status", "blank")

        # honour a caller-supplied threshold locally, so behaviour matches the
        # local class even when the service runs a different default
        if threshold is not None and serial:
            status = "accepted" if score >= threshold else "review"
        return serial, score, status

    def read(self, bgr):
        """Return (text, confidence) — matches the local class's read()."""
        serial, score, _ = self.predict(bgr)
        return serial, score


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python3 meterocr.py <image.jpg> [more.jpg ...]")
        raise SystemExit(1)
    ocr = MeterOCR()
    print("service:", ocr.health())
    for p in sys.argv[1:]:
        t, s, st = ocr.predict(p)
        print("%s\t%s\t%s\t%s" % (p, t, s, st))
