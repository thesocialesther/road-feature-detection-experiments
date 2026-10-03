#!/usr/bin/env python3
"""Inspect raw YOLO ONNX predictions without ESP, TTL, confirmation or NMS."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time


def decode_raw(output, class_count, np):
    """Decode explicit output layouts instead of guessing from candidate count."""
    rows = np.asarray(output)
    if rows.ndim == 3 and rows.shape[0] == 1:
        rows = rows[0]
    if rows.ndim != 2:
        raise ValueError(f"Expected detection tensor, received {rows.shape}")
    if rows.shape[1] == class_count + 4:
        layout = "traditional"
    elif rows.shape[0] == class_count + 4:
        rows = rows.T
        layout = "traditional"
    elif rows.shape[1] == 6:
        layout = "end-to-end"
    elif rows.shape[0] == 6:
        rows = rows.T
        layout = "end-to-end"
    else:
        raise ValueError(f"Unknown output layout {rows.shape} for {class_count} classes")
    if not np.isfinite(rows).all():
        raise ValueError("Model output contains NaN or infinity")
    if layout == "traditional":
        class_scores = rows[:, 4:]
        ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(rows)), ids]
        boxes = rows[:, :4].copy()
        boxes[:, :2] = rows[:, :2] - rows[:, 2:4] / 2
        boxes[:, 2:4] = rows[:, :2] + rows[:, 2:4] / 2
        maxima = class_scores.max(axis=0) if len(rows) else np.zeros(class_count)
    else:
        boxes, scores = rows[:, :4], rows[:, 4]
        if not np.equal(rows[:, 5], np.round(rows[:, 5])).all():
            raise ValueError("Non-integer class IDs in end-to-end output")
        ids = rows[:, 5].astype(int)
        if ((ids < 0) | (ids >= class_count)).any():
            raise ValueError("Output class IDs do not match model metadata")
        maxima = np.array([scores[ids == i].max() if (ids == i).any() else 0
                           for i in range(class_count)])
    return layout, boxes, scores, ids, maxima


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("yolo11n.onnx"))
    parser.add_argument("--source", default="0", help="USB camera index, image path or video path")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold; default 0.2 (minimum)")
    parser.add_argument("--top-k", type=int, default=20, help="Maximum boxes to draw/log, ranked by score")
    parser.add_argument("--frames", type=int, default=30, help="Frames to process; 0 runs until interrupted")
    parser.add_argument("--show-window", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if not 0 <= args.conf <= 1 or args.top_k < 1 or args.frames < 0:
        parser.error("Require confidence 0–1, top-k >= 1 and frames >= 0")
    return args


def run(args):
    import autonom as vision
    cv2, np = vision.cv2, vision.np
    session = vision.create_session(args.model.expanduser().resolve(), threads=2, allow_spinning=False)
    model_input = session.get_inputs()[0]
    names = vision.parse_model_names(session)
    if len(names) == 2:
        raise ValueError("Two-class layouts are ambiguous; this diagnostic targets the five-class road model")
    width, height = vision.model_input_size(model_input, session)
    output_dir = args.output_dir or Path("onnx_tests") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model: {args.model}\nInput: {model_input.shape}, {model_input.type}", flush=True)
    print(f"Outputs: {[o.shape for o in session.get_outputs()]}\nClasses: {names}", flush=True)
    print(f"Confidence={args.conf}; no NMS, ESP or expiry. Drawing top {args.top_k} candidates.", flush=True)
    print(f"Results: {output_dir.resolve()}", flush=True)
    camera = None
    window = False
    image = None
    try:
        if args.show_window:
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                raise RuntimeError("Preview requires a desktop session; omit --show-window to save images")
            cv2.namedWindow("ONNX diagnostic - Q to quit", cv2.WINDOW_NORMAL)
            window = True
        if args.source.isdecimal():
            camera = vision.open_usb_camera(640, 480, 15, int(args.source))
        else:
            source = Path(args.source).expanduser()
            if not source.is_file():
                raise FileNotFoundError(source)
            image = cv2.imread(str(source))
            if image is None:
                camera = cv2.VideoCapture(str(source))
                if not camera.isOpened():
                    raise RuntimeError(f"Could not open {source}")
        with (output_dir / "scores.jsonl").open("w") as report:
            frame_number = 0
            while args.frames == 0 or frame_number < args.frames:
                success, frame = (True, image.copy()) if image is not None else camera.read()
                if not success or frame is None:
                    print("Capture ended or failed", flush=True)
                    break
                tensor, scale, pad_x, pad_y = vision.preprocess(frame, width, height, model_input.type)
                start = time.perf_counter()
                outputs = session.run(None, {model_input.name: tensor})
                inference_ms = (time.perf_counter() - start) * 1000
                layout, boxes, scores, ids, maxima = decode_raw(outputs[0], len(names), np)
                eligible = np.flatnonzero(scores >= args.conf)
                top = eligible[np.argsort(-scores[eligible], kind="stable")[:args.top_k]]
                detections = []
                for index in top:
                    box = vision.restore_xyxy(boxes[index], scale, pad_x, pad_y, frame.shape[1], frame.shape[0])
                    detections.append((box, float(scores[index]), int(ids[index])))
                    x1, y1, x2, y2 = box
                    colour = vision.class_colour(int(ids[index]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
                    cv2.putText(frame, f"{names[int(ids[index])]} {float(scores[index]):.3g}",
                                (x1, max(15, y1)), cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1)
                frame_number += 1
                record = {"frame": frame_number, "layout": layout, "inference_ms": inference_ms,
                          "candidates": len(scores), "above_threshold": len(eligible),
                          "class_max_scores": dict(zip(names, map(float, maxima))),
                          "top_detections": [{"class": names[c], "score": s, "box": b} for b, s, c in detections]}
                report.write(json.dumps(record) + "\n")
                report.flush()
                print(f"Frame {frame_number}: {layout}, inference={inference_ms:.0f}ms, "
                      f"candidates={len(scores)}, eligible={len(eligible)}; max scores: "
                      + ", ".join(f"{name}={score:.6g}" for name, score in zip(names, maxima)), flush=True)
                if not cv2.imwrite(str(output_dir / "latest.jpg"), frame):
                    raise RuntimeError("Could not save diagnostic image")
                if args.show_window:
                    cv2.imshow("ONNX diagnostic - Q to quit", frame)
                    if (cv2.waitKey(1) & 0xff) in (ord("q"), 27):
                        break
                if image is not None:
                    break
    finally:
        if camera is not None:
            camera.close() if hasattr(camera, "close") else camera.release()
        if window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run(arguments())
    except KeyboardInterrupt:
        print("Stopped")
