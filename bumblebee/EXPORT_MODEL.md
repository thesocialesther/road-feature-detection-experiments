# Export the trained YOLO11 road model

`export_yolo11_onnx.py` converts the uploaded `yolo11n.pt` to `yolo11n.onnx`.
It uses the checkpoint's training version of Ultralytics, preserves the five
road classes, and exports a fixed 640×640 float32 CPU model without embedded NMS.

Use a separate export environment so installing Ultralytics does not replace
the GUI-enabled OpenCV in your camera environment:

```bash
python3 -m venv .venv-export
source .venv-export/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-export.txt
python export_yolo11_onnx.py
```

The script refuses to overwrite an existing output. To re-export for comparison:

```bash
python export_yolo11_onnx.py --output yolo11n_reexport.onnx
```

Before publishing the output file, it validates the ONNX graph and compares raw
PyTorch/ONNX predictions on two deterministic synthetic inputs. The resulting
`.export.json` contains the shapes, classes, exporter version and numerical
errors. Numerical parity is an export check, not a measurement of road detection
accuracy. Test accuracy separately on known-positive road images.

Export options follow the [Ultralytics ONNX export documentation](https://docs.ultralytics.com/modes/export).
The checkpoint and previous YOLO26 ONNX file are retained for comparison.

All detection entry points now default to `yolo11n.onnx`: `main.py`,
`test_onnx.py`, `autonom.py`, `auton.py`, `autonomous.py`, and `Detector.py`.
The Pi-camera variant reads class names from ONNX metadata rather than its old
four-label override. Export validation results are in `yolo11n.export.json`.

The generated model also completed inference on a recorded video frame through
`test_onnx.py`; results are in `model_inspection/yolo11_smoke/`. That frame's
maximum score was about 0.0564, so successful conversion alone does not establish
road detection accuracy. Use known-positive road images for that check.
