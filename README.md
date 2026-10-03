# Road-feature detection experiments

**Author: Oluwaferanmi Esther Onifade**

Computer-vision experiments for detecting five road features: cracks, speed bumps, potholes, left road boundaries, and right road boundaries. The repository preserves training and video-inference work plus an experimental ONNX/ESP actuation prototype.

## Components

- Training notebook: road-dataset preparation and model experiments.
- `run_inference.py`: YOLO11 video inference, annotated output, and detection records.
- Display experiments: temporal box smoothing, brief detection holds, and optional box insets.
- Classical road-edge estimator: Canny/Hough-based visualization.
- `bumblebee/`: ONNX inference/export and a headless USB/ESP prototype.
- Annotation archives and image inventories for two local dataset splits.

YOLO detections, smoothed display boxes, and classical estimated road lines are different outputs. Display adjustments do not change the underlying detector's measured accuracy.

## Setup and video inference

Create and activate a Python virtual environment, then install the inference dependencies:

~~~sh
python -m pip install -r requirements-inference.txt
python run_inference.py --help
~~~

Supply trained weights compatible with the five road classes and your input footage. An example using unmodified display boxes is:

~~~sh
python run_inference.py --model path/to/trained_road_model.pt --source path/to/road_video.mp4 --output outputs/road_demo.mp4 --device cpu --conf 0.25 --boundary-conf 0.25 --box-scale 1 --smooth-alpha 1 --hold-frames 0
~~~

Choose confidence thresholds using validation results. See [INFERENCE_README.md](INFERENCE_README.md), [bumblebee/HEADLESS.md](bumblebee/HEADLESS.md), and [bumblebee/EXPORT_MODEL.md](bumblebee/EXPORT_MODEL.md) for component-specific instructions.

## Dataset status

[DATASETS.md](DATASETS.md) documents the two exported annotation sets and compressed image manifests. Full image files are not included. The local split inventories contain 4,298 and 5,704 files respectively; neither has been confirmed as the final 8,998-image project dataset. Preserve the matching image/label structure and verify YAML paths when restoring a dataset.

## Project status and missing artifacts

This is an experimental source snapshot. The video-inference code is YOLO11-based; notebook experiments also include later YOLO26 work. The final YOLO26 Raspberry Pi deployment has not yet been identified and is not represented as complete here.

To complete the final-project release, add the confirmed deployment source, matching trained `.pt`/`.onnx` models, final dataset/splits, class configuration, evaluation reports, and sample footage or outputs. Notebook outputs were cleared for export. Physical autonomous-driving performance has not been established by this snapshot.
