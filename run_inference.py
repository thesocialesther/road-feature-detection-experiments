"""Annotate a video using the supplied road-feature YOLO checkpoint.

Run: python run_inference.py
Options: python run_inference.py --conf 0.2 --imgsz 960 --device cpu
Prediction API: https://docs.ultralytics.com/modes/predict/
Output is a silent MP4; original video and weights are never modified.
"""

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from box_smoothing import BoxSmoother

BASE = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE / ".ultralytics"))


def annotate(frame, result, names, cv2, confidence_format="percent", display_boxes=None):
    """Draw high-contrast labels, moving overlapping captions to free rows."""
    colors = [(0, 200, 255), (0, 165, 255), (40, 60, 240), (255, 180, 0), (100, 220, 60)]
    occupied = []
    height, width = frame.shape[:2]
    if display_boxes is None:
        display_boxes = [dict(xyxy=r[:4], confidence=r[4], class_id=int(r[5]), held_frames=0)
                         for r in result.boxes.data.cpu().tolist()]
    scale = max(0.65, width / 1200)
    thickness = max(1, round(width / 1400))
    line_width = max(2, round(width / 650))
    padding = max(6, round(width / 250))
    for box in display_boxes:
        x1, y1, x2, y2 = box['xyxy']
        confidence, class_id = box['confidence'], box['class_id']
        class_id = int(class_id)
        color = colors[class_id % len(colors)]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)
        score = f"{confidence:.0%}" if confidence_format == "percent" else f"{confidence:.2f}"
        label = f"{names[class_id].replace('_', ' ')} {score}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        bw, bh = tw + 2 * padding, th + baseline + 2 * padding
        left = max(0, min(x1, width - bw))
        preferred = max(0, min(y1 - bh, height - bh))
        top = preferred
        candidates = sorted(range(0, height - bh + 1, bh + padding), key=lambda y: abs(y - preferred))
        for candidate in [preferred] + candidates:
            if all(left + bw <= a or left >= c or candidate + bh <= b or candidate >= d
                   for a, b, c, d in occupied):
                top = candidate
                break
        occupied.append((left, top, left + bw, top + bh))
        cv2.rectangle(frame, (left, top), (left + bw, top + bh), (20, 20, 20), -1)
        cv2.rectangle(frame, (left, top), (left + bw, top + bh), color, thickness)
        cv2.putText(frame, label, (left + padding, top + th + padding), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thickness, cv2.LINE_AA)
        if top != preferred:
            cv2.line(frame, (left, top + bh), (x1, y1), color, thickness)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BASE / "yolo11n.pt")
    parser.add_argument("--source", type=Path, default=BASE / "vid3.mp4")
    parser.add_argument("--output", type=Path, default=BASE / "outputs" / "vid3_focused.mp4")
    parser.add_argument("--smooth-alpha", type=float, default=0.35, help="Box smoothing: 1 disables averaging")
    parser.add_argument("--hold-frames", type=int, default=2, help="Bridge brief gaps for this many frames")
    parser.add_argument("--box-scale", type=float, default=0.90,
                        help="Display-only width/height scale for non-boundary boxes; 1 preserves model extent")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--pothole-conf", type=float, default=0.25,
                        help="Stricter confidence threshold for potholes")
    parser.add_argument("--iou", type=float, default=0.40, help="NMS overlap threshold")
    parser.add_argument("--confidence-format", choices=("percent", "decimal"), default="percent")
    parser.add_argument("--max-pothole-area", type=float, default=1.0,
                        help="Maximum pothole box fraction of frame area; 1 disables this heuristic")
    parser.add_argument("--remove-gemini", action="store_true",
                        help="Cosmetically inpaint the inspected vid2 logo location after inference")
    parser.add_argument("--boundary-conf", type=float, default=0.02,
                        help="Separate confidence threshold for left/right road boundaries")
    parser.add_argument("--device", default=None, help="cpu or CUDA device number, e.g. 0")
    args = parser.parse_args()
    if not 0 < args.box_scale <= 1:
        parser.error("Box scale must be in (0, 1].")
    if not 0 < args.smooth_alpha <= 1 or not 0 <= args.hold_frames <= 10:
        parser.error("Smoothing alpha must be in (0, 1]; hold frames must be from 0 to 10.")
    if not 0 <= args.conf <= 1 or args.imgsz <= 0:
        parser.error("Confidence must be between 0 and 1; image size must be positive.")
    if args.boundary_conf is not None and not 0 <= args.boundary_conf <= 1:
        parser.error("Boundary confidence must be between 0 and 1.")
    if not 0 <= args.pothole_conf <= 1 or not 0 < args.iou <= 1:
        parser.error("Pothole confidence must be in [0, 1] and IoU in (0, 1].")
    if not 0 < args.max_pothole_area <= 1:
        parser.error("Maximum pothole area must be in (0, 1].")
    for path in (args.model, args.source):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if args.output.resolve() in (args.source.resolve(), args.model.resolve()):
        parser.error("Output must not overwrite the source or model.")
    if args.output.suffix.lower() != ".mp4":
        parser.error("Output must have an .mp4 extension.")
    if args.output.exists():
        parser.error("Output already exists. Choose another --output path.")

    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    print(f"Model classes: {model.names}", flush=True)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
        cap.release()
        raise RuntimeError("Invalid video dimensions or frame rate.")
    logo_mask = None
    if args.remove_gemini:
        if (width, height) != (848, 480):
            cap.release()
            parser.error("The inspected Gemini removal region requires an 848x480 video.")
        logo_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(logo_mask, np.array([(768, 375), (790, 400),
                                              (768, 425), (746, 400)], dtype=np.int32), 255)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot open MP4 video writer.")
    count = 0
    detections = Counter()
    class_frames = Counter()
    frame_records = []
    frames_with_detections = 0
    best_count = -1
    smoother = BoxSmoother(args.smooth_alpha, args.hold_frames)
    print(f"Processing {total} frames, {width}x{height}, {fps:.3f} FPS on {device}", flush=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            boundary_conf = args.conf if args.boundary_conf is None else args.boundary_conf
            thresholds = {name: (boundary_conf if name in ("left_road_boundaries", "right_road_boundaries")
                                 else args.pothole_conf if name == "potholes" else args.conf)
                          for name in model.names.values()}
            result = model.predict(frame, conf=min(thresholds.values()), imgsz=args.imgsz,
                                   iou=args.iou, device=device, verbose=False)[0]
            keep = [i for i, box in enumerate(result.boxes.data.cpu().tolist())
                    if box[4] >= thresholds[model.names[int(box[5])]]
                    and (model.names[int(box[5])] != "potholes" or
                         (box[2] - box[0]) * (box[3] - box[1]) <= args.max_pothole_area * width * height)]
            result = result[keep]
            display_boxes = smoother.update(result.boxes.data.cpu().tolist())
            for box in display_boxes:
                if model.names[box['class_id']] not in ("left_road_boundaries", "right_road_boundaries"):
                    x1, y1, x2, y2 = box['xyxy']
                    dx = (x2 - x1) * (1 - args.box_scale) / 2
                    dy = (y2 - y1) * (1 - args.box_scale) / 2
                    box['xyxy'] = [x1 + dx, y1 + dy, x2 - dx, y2 - dy]
            classes = result.boxes.cls.int().cpu().tolist()
            detections.update(model.names[c] for c in classes)
            class_frames.update(model.names[c] for c in set(classes))
            frame_records.append({"frame": count, "seconds": count / fps, "display_boxes": display_boxes,
                                  "detections": [{"class": model.names[int(b[5])],
                                                  "confidence": b[4], "xyxy": b[:4]}
                                                 for b in result.boxes.data.cpu().tolist()]})
            frames_with_detections += bool(classes)
            display_frame = (cv2.inpaint(frame, logo_mask, 3, cv2.INPAINT_TELEA)
                             if logo_mask is not None else frame.copy())
            annotated = annotate(display_frame, result, model.names, cv2, args.confidence_format, display_boxes)
            writer.write(annotated)
            if len(classes) > best_count:
                cv2.imwrite(str(args.output.with_suffix(".preview.jpg")), annotated)
                best_count = len(classes)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count}/{total} frames", flush=True)
    finally:
        cap.release()
        writer.release()
    if count == 0 or (total > 0 and count != total):
        raise RuntimeError(f"Decoded {count}/{total} frames; output may be incomplete.")
    summary = {
        "model": str(args.model.resolve()), "source": str(args.source.resolve()),
        "output": str(args.output.resolve()), "classes": model.names,
        "frames": count, "fps": fps, "width": width, "height": height,
        "confidence_threshold": args.conf, "image_size": args.imgsz,
        "confidence_format": args.confidence_format,
        "smoothing_alpha": args.smooth_alpha, "hold_frames": args.hold_frames,
        "display_box_scale_non_boundaries": args.box_scale,
        "boundary_confidence_threshold": args.boundary_conf,
        "pothole_confidence_threshold": args.pothole_conf,
        "nms_iou": args.iou,
        "max_pothole_area_fraction": args.max_pothole_area,
        "gemini_removal": "cosmetic inpainting after inference" if args.remove_gemini else None,
        "frames_per_class": {name: class_frames[name] for name in model.names.values()},
        "frames_with_detections": frames_with_detections,
        "detections_across_frames_not_unique_objects": dict(detections),
        "audio_preserved": False,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output.with_suffix(".detections.json").write_text(json.dumps(frame_records), encoding="utf-8")
    print(f"Saved: {args.output}\nDetections across frames: {dict(detections)}", flush=True)


if __name__ == "__main__":
    main()
