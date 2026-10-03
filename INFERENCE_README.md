# Road-feature video inference

## Experimental estimated road edges (no retraining)

```powershell
autonomous_vehicle/.venv/Scripts/python.exe autonomous_vehicle/run_boundary_overlay.py
```

Output: `outputs/vid3_estimated_boundary_boxes.mp4`. This separate renderer reuses the
saved vid3_focused YOLO damage boxes, large scores and smoothing. It replaces
YOLO boundary boxes with cyan/green **estimated** left/right road-edge boxes.
Each rectangle encloses the visible estimated line segment with a small margin;
it is a display conversion, not a box predicted by YOLO. The previous line-only
video remains available as `outputs/vid3_estimated_edges.mp4`.
It does not change YOLO weights or claim those lines are model detections.
The source video path, dimensions, frame count, and FPS must match the cache.

The estimator runs Canny/Hough line detection on each source frame, selects
outer-side line candidates with road-perspective constraints, and smooths them
temporally. It is tuned to this straight-road clip, not a general road segmentation
model. It can confuse curbs, markings, shadows and cracks, and is unsuitable as
a validated driving signal. Five sample frames were visually checked. Estimates
may be held for up to 0.3 seconds; labels identify held estimates or unavailable
edges. No made-up confidence percentages are assigned to these lines. Coverage
counts in the summary describe availability, not accuracy. The edges JSON records
per-frame status and normalized line parameters. Output retains full resolution
and FPS but has no audio.

## Current focused vid3 output

Running `run_inference.py` now produces `outputs/vid3_focused.mp4` with confidence
0.25 for cracks, speed bumps, and potholes; boundaries remain at 0.02.
NMS IoU is 0.40 to suppress more overlapping same-class detections.
The large percentage labels and smoothing remain, but the `(tracked)` caption
is removed. Two-frame holds still occur and remain identified by `held_frames`
in the display-box JSON; their scores are from the most recent detection.
Damage boxes are rendered at 90% of smoothed width and height (`--box-scale 0.90`),
centered on the original box. Boundary extents are unchanged. This is a cosmetic
inset, not a localization improvement, and may omit the outer part of a feature.
Raw prediction coordinates are preserved. Set `--box-scale 1` to disable the inset.

## Stabilized vid3 output

The earlier stabilized output is `outputs/vid3_stable.mp4`. Labels and box thickness
scale with video resolution, making confidence scores readable at 3816x2160.
Class-aware one-to-one IoU matching smooths coordinates with an exponential
moving average (`--smooth-alpha 0.35`). Two missing frames can be bridged
(`--hold-frames 2`); these boxes say `(tracked)` and retain the last observed
confidence. At 30 FPS this hold lasts at most about 67 ms. Smoothing introduces
a small position lag and cannot correct wrong model predictions or long gaps.
Raw detections remain unchanged in the JSON; `display_boxes` records the smoothed
coordinates, track IDs, and held-frame age separately. Summary counts are raw
detections. Use `--smooth-alpha 1 --hold-frames 0` to disable stabilization.

## Current vid3 run: restored last vid1 settings

Run from the workspace root:

```powershell
autonomous_vehicle/.venv/Scripts/python.exe autonomous_vehicle/run_inference.py
```

The original unsmoothed vid3 run is `outputs/vid3_sensitive_clean_labels.mp4`.
The source remains `vid3.mp4`.
Settings match the final vid1 run: image size 960, general and pothole confidence
0.05, boundary confidence 0.02, NMS IoU 0.70, percentage captions without `?`,
no pothole area exclusion, and no Gemini removal. All source frames, resolution,
and frame rate are preserved; the output has no audio. Low thresholds expose
weak predictions and do not guarantee correct detections or continuous boundaries.
Use `--confidence-format decimal` if decimal captions are wanted instead.
The matching `.summary.json`, `.detections.json`, and `.preview.jpg` provide
coverage statistics, per-frame detections, and a preview. For another run, select
a new `--output` filename to avoid overwriting an existing video.

## Previous vid2 run (historical settings)

The previous vid2 run used `vid 2.mp4`, input size 960, confidence 0.60,
pothole confidence 0.60, and NMS IoU 0.40. Captions use decimal scores such as
`potholes 0.72`. These conservative thresholds discard weak predictions but
can miss real features; they do not establish measured accuracy.

```powershell
autonomous_vehicle/.venv/Scripts/python.exe autonomous_vehicle/run_inference.py --source "autonomous_vehicle/vid 2.mp4" --output autonomous_vehicle/outputs/vid2_precision_new.mp4 --conf 0.60 --pothole-conf 0.60 --boundary-conf 0.60 --iou 0.40 --max-pothole-area 0.20 --confidence-format decimal --remove-gemini
```

The final output is `outputs/vid2_precision.mp4`. Use a different `--output` for a
repeat run. `--remove-gemini` cosmetically inpaints the small diamond at the
inspected logo location in this 848x480 video. Detection always uses original
frames; reconstructed pixels are only used for display. Inpainting approximates
the obscured patch and does not recover its true content. This option is specific
to this clip's logo position. The original video remains intact. Output is silent.

The summary and per-frame detection JSON files record thresholds and predictions.
Confidence scores are model outputs, not verified probabilities of correctness.

Review of the initial `vid2_annotated.mp4` found oversized pothole boxes on broad
cracked areas. That run excluded pothole boxes larger than 20% of the
frame (`--max-pothole-area 0.2`; use 1 to disable). This footage-specific heuristic
can reject real large nearby potholes and is not a trained correctness test.
The final video was rendered from the initial cached detections with these stricter
filters, without rerunning the network. Only one frame retains a pothole prediction
(score about 0.69); most visible features remain undetected. Thus the requested
reliable coverage has not been achieved by this checkpoint. Further model
improvement requires representative labeled data and validation.

## Earlier vid1 runs (historical settings)

The completed run with improved label placement is `outputs/vid1_road_features.mp4`.
Its matching preview and summary share the same filename stem.

Earlier runs used the `yolo11n.pt` and `vid1.mp4` next to the script.
It uses the class names stored in the checkpoint, draws colored bounding boxes
with class labels and confidence scores, and processes every frame at the
original resolution and frame rate. The MP4 output has no audio.

From the workspace root in PowerShell:

```powershell
.\autonomous_vehicle\.venv\Scripts\python.exe .\autonomous_vehicle\run_inference.py
```

For a fresh environment:

```powershell
python -m venv autonomous_vehicle/.venv
autonomous_vehicle/.venv/Scripts/python.exe -m pip install -r autonomous_vehicle/requirements-inference.txt
autonomous_vehicle/.venv/Scripts/python.exe autonomous_vehicle/run_inference.py
```

Outputs go into `autonomous_vehicle/outputs/`:

- `vid1_annotated.mp4`: annotated video.
- `vid1_annotated.preview.jpg`: frame with the most detections.
- `vid1_annotated.summary.json`: settings, video properties, and detection counts.

Existing output videos are protected from overwriting. Use `--output` to choose
a new filename on subsequent runs. For example:

```powershell
autonomous_vehicle/.venv/Scripts/python.exe autonomous_vehicle/run_inference.py --source autonomous_vehicle/vid1.mp4 --conf 0.2 --pothole-conf 0.2 --imgsz 960 --output autonomous_vehicle/outputs/vid1_960.mp4
```

The original confidence threshold was 0.25 and inference image size was 640,
matching the notebook's training size. Lower confidence may show more detections
but also more false positives. Larger inference sizes require more processing.
Counts are detections across frames, not counts of unique road features.

## More sensitive inference

`outputs/vid1_sensitive.mp4` uses larger inputs and lower thresholds:

```powershell
autonomous_vehicle/.venv/Scripts/python.exe autonomous_vehicle/run_inference.py --source autonomous_vehicle/vid1.mp4 --conf 0.05 --pothole-conf 0.05 --boundary-conf 0.02 --imgsz 960 --output autonomous_vehicle/outputs/vid1_sensitive_new.mp4
```

This is an exploratory visualization, not a guarantee that every feature is
detected correctly. Those earlier videos show confidence percentages; new runs use decimals.
The 2% boundary threshold deliberately exposes weak predictions and can produce
false positives. No boundaries are invented or carried forward from old frames.
`outputs/vid1_sensitive_clean_labels.mp4` is the regenerated version without
question marks in the captions, using the same inference settings.
The summary includes frames detected per class; `.detections.json` stores boxes
and confidence scores for every frame.

Testing 20 evenly spaced frames at a 1% threshold found that, at image size 960,
left-boundary confidence peaked at only 2.87% and right-boundary confidence at
10.84%. Threshold changes cannot make these predictions reliable or continuous.
This checkpoint is a bounding-box detector: its boundary boxes are not road-edge
curves. For reliable continuous boundaries, collect and label representative
video frames and evaluate retraining; use a segmentation model with road/edge
annotations if actual road-edge outlines are required. Temporal tracking can
bridge brief gaps only after reliable detections exist.

## Notebook review

The notebook configures five classes: `cracks`, `speed_bumps`, `potholes`,
`left_road_boundaries`, and `right_road_boundaries`. YOLO11n training uses
100 epochs, 640-pixel inputs, and batches of 16. It saves training artifacts
under `/content/drive/MyDrive/road_data_4/yolo_output/anomaly_detection_yolo11n`.
The separate later YOLO26 training section is not used by this script.

Inference and annotation use the [Ultralytics prediction API](https://docs.ultralytics.com/modes/predict/).
