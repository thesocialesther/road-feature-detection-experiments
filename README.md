# Road-feature detection experiments
Author: Oluwaferanmi Esther Onifade

Five-class road-feature experiments: cracks, speed bumps, potholes, left road boundary, and right road boundary. Contains the notebook, YOLO11 video inference, temporal display smoothing, an experimental classical road-edge estimator, and an ONNX/ESP actuation prototype.

This is an experimental snapshot. It does not establish that this is the final YOLO26 Raspberry Pi implementation used in the final-year evaluation. Trained weights and video inputs are excluded. Complete annotation archives and image inventories are included; see DATASETS.md for the pending image dataset.

## Setup
~~~sh
python -m venv .venv
python -m pip install -r requirements-inference.txt
python run_inference.py --help
~~~
Read [INFERENCE_README.md](INFERENCE_README.md) for the original commands and working-directory assumptions. Supply compatible five-class weights and footage. See bumblebee/HEADLESS.md and bumblebee/EXPORT_MODEL.md for the ONNX prototype.

Estimated road-edge lines and smoothed display boxes are visualization experiments, not validated driving signals. Notebook outputs were cleared for export.

## Dataset access
See [DATASETS.md](DATASETS.md) for included data, exact file manifests, and setup instructions.
