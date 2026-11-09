# Road feature detection: final YOLO26n verification (9 November 2026)

Final selected model (author-confirmed): **YOLO26n**. The author reports a full physical prototype, but independent Raspberry Pi/ESP32 validation records have not yet been verified.

## Recovered training logs

Original training records were recovered from D: drive, in personal_projects/autonomous_vehicle/road_data_4/yolo_output/.

| Metric | YOLO11n comparison | YOLO26n selected |
| --- | ---: | ---: |
| Epochs logged | 100 | 100 |
| Image size | 640 | 640 |
| Batch size | 16 | 16 |
| Best logged mAP50-95 epoch | 79 | 89 |
| Precision at that epoch | 0.81113 | 0.76093 |
| Recall at that epoch | 0.71901 | 0.80029 |
| mAP50 at that epoch | 0.78303 | 0.78680 |
| mAP50-95 at that epoch | 0.48353 | 0.47324 |
| Epoch-100 precision | 0.81691 | 0.76630 |
| Epoch-100 recall | 0.72314 | 0.75284 |
| Epoch-100 mAP50 | 0.75200 | 0.76731 |
| Epoch-100 mAP50-95 | 0.46632 | 0.45209 |

These are **saved validation-log metrics**, NOT independently reproduced test metrics. Both runs reference the five-class split_dataset2 YAML. The numbers are not evidence of a separately evaluated final held-out test set.

## Checkpoints (SHA-256)

- YOLO26n best.pt: 3f00eebfd2f895eb9909429939905c1525318f67262caff351c7cf88fe1fc9c2
- YOLO26n last.pt: b051ab18cdeae622f6dd33fe9b7874c1ead701c4a606814d01a86f2cda814ded
- YOLO11n best.pt: 29d0b2fb666317abcdcbab19d445350a970f6ba3cfb771671cda68e046331437

The D:/Final Year Project/yolo11n.pt file exactly matches the historical YOLO11n best checkpoint. The existing Bumblebee YOLO11n ONNX artifact is NOT the final YOLO26n model, and must not be relabeled. Model binaries are not in this repository.

## Known limitations and release blockers

1. Original dataset: a prior audit reported exact-image overlap across splits, contradictory bounding boxes, unmatched image/label pairs and an invalid zero-width bounding box. Original scores may overstate generalization.
2. Both training configurations used fliplr: 0.5, which can invalidate left/right boundary supervision without an explicit class swap. Correct in a new experiment; preserve original run.
3. Final YOLO26n ONNX export and inference-parity test need completion; historical Bumblebee runner depends on missing autonom.py code in this repository.
4. Full physical prototype is author-reported. Model hash used on the Pi, hardware configuration, firmware behavior, FPS/latency and a recorded repeatable demonstration remain unverified.
5. No clean source-sequence-disjoint test evaluation has been independently reproduced.

## Reproduction priorities

- Retain original models and logged results unchanged.
- Audit and manually resolve ambiguous labels; construct source-sequence-disjoint train/val/test splits.
- Train a *new* YOLO26n run with seed, environment versions, fixed five-class mapping and fliplr=0 unless left/right IDs are swapped correctly.
- Run independent evaluation and report per-class precision, recall, mAP50, mAP50-95 and failure examples.
- Export verified YOLO26n best.pt to ONNX, check numerical parity, then benchmark on the actual Raspberry Pi.
- Record and verify ESP32 serial interface, safe handling of model outputs and hardware actions.

Documentation of software source, saved training results, ONNX export and physical system testing must remain distinct. No unperformed hardware or retraining test is claimed here.
