# YOLO26n desktop checkpoint smoke test — 2026-11-09

A single-image inference smoke test was executed against the *recovered final YOLO26n best checkpoint* on the author's Windows computer.

- Checkpoint SHA-256: `3f00eebfd2f895eb9909429939905c1525318f67262caff351c7cf88fe1fc9c2`
- Source image: `split_dataset2/val/images/C (103).jpg`
- Execution: Ultralytics Python API, CPU inference, image size 640, confidence threshold 0.25
- Model class metadata: `cracks`, `speed_bumps`, `potholes`, `left_road_boundaries`, `right_road_boundaries`
- Recorded output: **1 bounding-box detection**
- Wall-clock including checkpoint load: **31.526 seconds**
- Local JSON evidence: `D:/personal_projects/autonomous_vehicle/verification_2026-11-09/yolo26n_smoke_test.json`

**Interpretation:** The local checkpoint loaded successfully, carried the expected five-class mapping, and completed inference on one image. This does *not* measure detection accuracy, classification correctness, throughput/FPS, ONNX parity, Raspberry Pi runtime, serial communication, or the physical prototype's control performance. Timing includes checkpoint load and should not be treated as per-frame inference latency.

Original image, checkpoint and repository contents were not altered by this test.
