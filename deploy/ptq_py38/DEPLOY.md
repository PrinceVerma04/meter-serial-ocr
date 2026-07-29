# Meter-OCR PTQ INT8 — deployment for **Python 3.8**

Reads the serial number off a cropped electricity-meter nameplate.
**21 MB INT8 ONNX model · CPU-only · no PaddlePaddle, torch, or TensorFlow.**

Measured **97.01%** exact-match on 1,004 human-verified crops.

> This package replaces the earlier Python 3.10 build, which has been deleted.
> Only the dependency pins and a few syntax details changed — **the model file
> and the maths are byte-identical**, so accuracy is unchanged.

---

## 0. FIRST — repair the broken TensorFlow, if you already ran the bad install

If you ran `pip3 install onnxruntime==1.16.3` **without a venv**, pip printed:

```
ERROR: tensorflow 2.4.1 has requirement numpy~=1.19.2, but you'll have numpy 1.24.4
  Uninstalling numpy-1.19.5: Successfully uninstalled numpy-1.19.5
```

**Your TensorFlow 2.4.1 is now broken.** Repair it, then start over with a venv:

```bash
# 1. restore the numpy TensorFlow needs, and remove the OCR packages from system Python
pip3 uninstall -y onnxruntime
pip3 install "numpy==1.19.5"

# 2. confirm TensorFlow is healthy again
python3 -c "import tensorflow as tf, numpy; print(tf.__version__, numpy.__version__)"
#    expect: 2.4.1 1.19.5
```

Then install the OCR into its own venv (section 2.3, Option A) — never system-wide.

### Why they can never share one environment

This is not a preference, it is arithmetic:

| | needs |
|---|---|
| TensorFlow 2.4.1 | `numpy~=1.19.2` |
| this model (onnxruntime ≥1.13) | `numpy>=1.21.6` |

**No numpy version satisfies both.** There is no pin, no flag, no ordering of
installs that fixes it. The OCR must run in a separate venv, and if your Flask
app needs both, the app calls the OCR over HTTP (`app.py` exists for exactly
this) rather than importing it.

---

## 0b. Two bugs found and fixed since the last zip

**Bug 1 — the model would not load on your onnxruntime.** The exported ONNX
declared six operator domains it never actually uses, including `ai.onnx.ml`
**opset 5**. onnxruntime ≤1.16 supports that domain only to opset 3 and refuses
the model outright:

```
FAIL : Load model ... failed: ... Opset 5 is under development ...
       Current official support for domain ai.onnx.ml is till opset 3.
```

The bundled model has had those dead declarations stripped. Verified
**bit-identical output** to the original (max abs difference 0.0 across random
inputs) — this is a metadata fix, not a re-quantization, so accuracy is
unchanged at 97.01%.

**Bug 2 — the pinned onnxruntime version does not exist for your pip.** Your pip
offers only up to 1.16.3 because onnxruntime 1.17+ ships `manylinux_2_27/2_28`
wheels whose tags an older pip cannot parse. The pin is now **1.16.3**, verified
working.

Tested version floor and ceiling for this model:

| onnxruntime | Result |
|---|---|
| 1.12.0 | ❌ `com.microsoft:QLinearSoftmax is not a registered function/op` |
| **1.13.1 – 1.16.3** | ✅ reads `WHL059419` @ 0.9913 |
| 1.17+ | ✅ works, but your pip cannot see it |

---

## 1. What changed for Python 3.8

Every pin below is the newest version that still ships a **cp38 manylinux
x86_64 wheel** — verified against PyPI, not assumed.

| Package | 3.8 pin | Why |
|---|---|---|
| `onnxruntime` | **1.16.3** | floor 1.13 (QLinearSoftmax); ceiling = newest your pip can see |
| `numpy` | **1.24.4** | **hard ceiling** — 1.25+ requires Python ≥3.9 |
| `opencv-python-headless` | **4.9.0.80** | `cp37-abi3` wheel, installs on 3.8; headless avoids needing libGL |
| `Flask` | 3.0.3 | last 3.x supporting 3.8 (3.1 requires ≥3.9) |
| `Werkzeug` | 3.0.6 | matched to Flask 3.0.x |
| `Flask-Cors` | 5.0.0 | optional |
| `gevent` | 24.2.1 | optional; production WSGI server |

Code changes: `from __future__` imports, `%`-formatting instead of newer
f-string features, explicit `@app.route(...)` instead of `@app.get`, and no
`list[str]`-style annotations. **All four `.py` files were verified to parse
under 3.8 grammar** (`ast.parse(..., feature_version=(3,8))`).

### ⚠️ The requirements.txt you sent me is NOT from this server

You gave me a list containing `tensorflow==2.15.0.post1` and `numpy==1.26.4`.
**Neither can install on Python 3.8** — both require ≥3.9:

```
tensorflow 2.15.0.post1   requires_python >=3.9    cp38 wheel: False
numpy      1.26.4         requires_python >=3.9    cp38 wheel: False
```

So that file describes a *different* (3.10) machine. Two consequences:

1. The real server runs **TensorFlow 2.4.1 with numpy 1.19.5** — much older than
   that file suggests. The numpy conflict is therefore *worse*, not absent: the
   gap is 1.19 vs ≥1.21, and no version bridges it (see section 0).
2. A constraints file cannot help here. Isolation is the only fix — hence the
   venv is mandatory rather than merely recommended.

---

## 2. Sequence — follow in order

### 2.1 Confirm the interpreter

```bash
python3.8 --version          # expect: Python 3.8.x
which python3.8
```

If `python3` is *not* 3.8 on that box, use `python3.8` explicitly everywhere below.

### 2.2 Upload and unpack

```bash
scp -i your-key.pem meterocr_ptq_py38.zip ubuntu@<SERVER>:/tmp/
ssh -i your-key.pem ubuntu@<SERVER>
```

```bash
sudo mkdir -p /opt/meterocr && sudo chown $USER:$USER /opt/meterocr
cd /opt/meterocr
unzip /tmp/meterocr_ptq_py38.zip -d /tmp/mo && mv /tmp/mo/meterocr_ptq_py38/* .
ls model/          # meterocr_ptq_int8.onnx (~21 MB) + ppocrv5_dict.txt
```

### 2.3 Install

**Option A — venv. USE THIS.** It is the only option that leaves TensorFlow 2.4.1 working:

```bash
cd /opt/meterocr
sudo apt-get update && sudo apt-get install -y python3.8-venv
python3.8 -m venv venv
./venv/bin/pip install --upgrade "pip<25"        # last pip line that supports 3.8
./venv/bin/pip install -r requirements-py38.txt
```

Use `./venv/bin/python` for every command below.

**Option B — inference only, no web server** (3 packages, ~200 MB) — still in a venv:

```bash
python3.8 -m venv venv
./venv/bin/pip install -r requirements-minimal.txt
```

Choose this if you will `from meterocr import MeterOCR` inside your own script
rather than run the HTTP API. Note this venv still cannot contain TensorFlow —
if your Flask app imports TF, it must reach the OCR over HTTP instead.

**Option C — system-wide:** ~~`python3.8 -m pip install -r requirements-py38.txt`~~

**Do not do this on the flaskML server.** It is what broke TensorFlow 2.4.1
(see section 0). Only valid on a box with no other Python workloads.

### 2.4 Verify before wiring anything up

```bash
cd /opt/meterocr && ./venv/bin/python smoke_test.py     # or python3.8 smoke_test.py
```

Checks the interpreter version, imports, that numpy is valid for 3.8, model file
size, session load, a known-answer prediction (`sample.jpg` → `WHL059419`), and
latency. **Exit 0 = safe to proceed.**

### 2.5 Run it once in the foreground

```bash
./venv/bin/python app.py --host 0.0.0.0 --port 8080
```

From a second shell:

```bash
curl http://localhost:8080/health
curl -F image=@sample.jpg http://localhost:8080/ocr
# -> {"serial":"WHL059419","confidence":0.9913,"status":"accepted","ms":...}
```

`Ctrl-C` when satisfied.

### 2.6 Install as a service

```bash
sudo cp /opt/meterocr/meterocr.service /etc/systemd/system/
# Not using the venv? Edit ExecStart to your python3.8 path first.
sudo systemctl daemon-reload
sudo systemctl enable --now meterocr
systemctl status meterocr
journalctl -u meterocr -f
```

### 2.7 Load balancer

- Target group **HTTP : 8080**, health check path **`/health`**.
- Health check grace period **≥ 30 s** (model load ~1 s + boot).

### 2.8 Security — before real traffic

The API is **unauthenticated by design** (meant to sit behind your gateway):

- keep the security group scoped to your VPC/ALB, **never** `0.0.0.0/0`;
- or front it with API Gateway / an ALB with auth;
- or add an API-key check in `app.py` (~5 lines in a `@app.before_request`).

`METEROCR_MAX_MB` (default 10) caps upload size.

---

## Using it

```bash
# single image
curl -F image=@crop.jpg http://<HOST>:8080/ocr

# raw body
curl --data-binary @crop.jpg -H 'Content-Type: image/jpeg' http://<HOST>:8080/ocr

# batch
curl -F images=@a.jpg -F images=@b.jpg http://<HOST>:8080/ocr/batch
```

```json
{"filename":"crop.jpg","serial":"WHL059419","confidence":0.9913,"status":"accepted","ms":142.3}
```

**Bulk / offline** (skips HTTP entirely):

```bash
aws s3 sync s3://your-bucket/crops/ ./crops/
./venv/bin/python batch_predict.py --images ./crops --out results.csv --workers 4
```

**From your own Python:**

```python
from meterocr import MeterOCR
ocr = MeterOCR(threads=2)          # build ONCE at process start, never per request
serial, conf, status = ocr.predict("crop.jpg")
```

---

## `status` — the part that matters in production

| status | meaning | action |
|---|---|---|
| `accepted` | confidence ≥ 0.95 | **trust it, no human review** |
| `review` | below 0.95 | send to a human queue |
| `blank` | nothing read | bad crop — re-capture |

Across every dataset tested, `accepted` reads are correct **95–98%** of the time.
Do not collapse the three statuses into one string — that gate is the whole
reason the system is safe to automate.

---

## Tuning

| Env var | Default | Notes |
|---|---|---|
| `METEROCR_THREADS` | 2 | ORT intra-op threads **per process** |
| `METEROCR_THRESHOLD` | 0.95 | auto-accept cutoff |
| `METEROCR_MAX_MB` | 10 | max upload size |
| `OMP_NUM_THREADS` | 2 | keep equal to `METEROCR_THREADS` |

Expect **~130–215 ms/image** on a t3-class CPU. Scale out with more instances
rather than more threads — this model stops gaining past ~2 threads.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Could not find a version that satisfies onnxruntime==1.18.1` | your pip cannot read manylinux_2_27 tags; use the pinned 1.16.3 |
| `Opset 5 is under development ... ai.onnx.ml is till opset 3` | old model file — re-unzip, the bundled one has this stripped |
| `com.microsoft:QLinearSoftmax is not a registered function/op` | onnxruntime <1.13; install `onnxruntime==1.16.3` |
| `tensorflow 2.4.1 has requirement numpy~=1.19.2` | you installed system-wide instead of into a venv — see section 0 |
| `ERROR: Could not find a version that satisfies numpy==1.26.4` | you are on 3.8 — use `requirements-py38.txt`, which pins 1.24.4 |
| `onnxruntime` has no matching distribution | pip too old or wrong interpreter: `python3.8 -m pip install --upgrade "pip<25"` |
| `ImportError: libGL.so.1` | you installed `opencv-python` instead of `-headless`; switch, or `apt-get install -y libgl1` |
| `SyntaxError` on start | you ran an old 3.10-only file — this package is the 3.8 build, re-unzip |
| ~1 s per image | `METEROCR_THREADS` fighting `OMP_NUM_THREADS`; set both to 2 |
| Model reloads per request | you built `MeterOCR()` inside the handler — build once at module level |
| Serial right but spacing differs | output is normalized to match the benchmark; pass `clean=False` for raw |
| 413 error | image exceeds `METEROCR_MAX_MB` |

---

## One note on Python 3.8 itself

Python 3.8 reached **end of life in October 2024** — it receives no further
security patches, and the pins above are frozen at their final 3.8-compatible
releases. That is fine for running this model today, but it will slowly drift
further from supported versions.

Nothing in this package requires 3.8 specifically; it runs unchanged on 3.9–3.12
if you drop the version pins. **When the server is eventually upgraded, reinstall
from unpinned versions** — you get newer onnxruntime (faster INT8 kernels) for
free, with no code changes.
