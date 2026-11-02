# Paddle 3.0.0 compat env (needed for the ONNX export)

**Why:** Paddle 3.3.1 (used for training) exports PIR **v4**, which `paddle2onnx 2.1.0`
can't convert. Paddle **3.0.0** exports PIR **v1**, which converts. Only the export step
before quantization needs this env.

**Where it is:** `/home/coral/Desktop/Prince/Prince number detection in meter/size_reduction/compat_env/venv`
(outside this project; Python 3.10, paddle 3.0.0, paddle2onnx 2.1.0, onnx 1.17, onnxruntime 1.23).
`quantization/quantize_v4_ptq.py` uses it automatically.

**Rebuild it** (only if that folder is lost): `bash quantization/compat_env/1_setup_venv.sh`

Known quirk: the venv was moved after it was created, so its `bin/paddle2onnx` launcher
points at a path that no longer exists. `quantize_v4_ptq.py` calls the converter through
the venv's python instead (`python -c "from paddle2onnx.command import main"`).
