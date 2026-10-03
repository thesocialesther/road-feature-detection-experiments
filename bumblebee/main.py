#!/usr/bin/env python3
"""Headless USB road detection and acknowledged ESP class-output control."""

import argparse
import ast
import binascii
import logging
import math
import os
from pathlib import Path
import secrets
import signal
import threading
import time


CLASSES = (
    "cracks", "speed_bumps", "potholes",
    "left_road_boundaries", "right_road_boundaries",
)
LOG = logging.getLogger("road-control")


def packet(payload):
    data = payload.encode("ascii")
    return data + f"*{binascii.crc_hqx(data, 0xffff):04X}\n".encode("ascii")


def unpack(line):
    try:
        body, checksum = line.strip().rsplit(b"*", 1)
        if len(checksum) != 4 or binascii.crc_hqx(body, 0xffff) != int(checksum, 16):
            return None
        return body.decode("ascii")
    except (ValueError, UnicodeError):
        return None


def model_class_names(metadata):
    """Require an explicit, complete model mapping before controlling outputs."""
    try:
        names = ast.literal_eval(metadata["names"])
        if isinstance(names, dict):
            names = {int(k): str(v) for k, v in names.items()}
            if set(names) != set(range(len(names))):
                raise ValueError("class IDs must be contiguous from zero")
            names = [names[i] for i in range(len(names))]
        if not isinstance(names, (list, tuple)) or len(names) != len(CLASSES):
            raise ValueError("expected exactly five classes")
        if set(names) != set(CLASSES):
            raise ValueError("class names do not match the road classes")
        return list(names)
    except (KeyError, ValueError, SyntaxError, TypeError) as error:
        raise ValueError(f"Model must contain names metadata for {CLASSES}: {error}") from error


class DetectionPolicy:
    """Confirm consecutive frames; clear a class immediately when absent."""

    def __init__(self, names, confirm_frames=2):
        self.names = names
        self.confirm_frames = confirm_frames
        self.streaks = [0] * len(CLASSES)

    def update(self, detections):
        present = set()
        for box, score, class_id in detections:
            if not 0 <= class_id < len(self.names):
                raise ValueError(f"Model returned unknown class ID {class_id}")
            if not math.isfinite(score) or not all(math.isfinite(x) for x in box):
                raise ValueError("Model returned non-finite detection values")
            if box[2] > box[0] and box[3] > box[1]:
                present.add(self.names[class_id])
        mask = 0
        for index, name in enumerate(CLASSES):
            self.streaks[index] = min(self.streaks[index] + 1, self.confirm_frames) if name in present else 0
            if self.streaks[index] >= self.confirm_frames:
                mask |= 1 << index
        return mask


class ESPLink(threading.Thread):
    """One serial owner, one latest state, bounded ACK wait, no active retries."""

    def __init__(self, port, ttl_ms=1500, ack_timeout=1.0):
        super().__init__(name="esp-serial", daemon=True)
        self.port = port
        self.ttl_ms = ttl_ms
        self.ack_timeout = ack_timeout
        self.session = secrets.token_hex(8)
        self.sequence = 0
        self.lock = threading.Lock()
        self.latest = (0, 0.0)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error = None
        self.buffer = bytearray()

    def publish(self, mask, captured_at):
        if not 0 <= mask <= 31:
            raise ValueError("Invalid class mask")
        with self.lock:
            self.latest = (mask, captured_at)

    def current_state(self, now):
        with self.lock:
            mask, captured_at = self.latest
        remaining = self.ttl_ms - math.ceil(max(0, now - captured_at) * 1000)
        if remaining < 100:
            return 0, self.ttl_ms
        return mask, remaining

    def exchange(self, serial_port, payload, expected):
        wire = packet(payload)
        if serial_port.write(wire) != len(wire):
            raise RuntimeError("Incomplete serial write")
        deadline = time.monotonic() + self.ack_timeout
        while time.monotonic() < deadline:
            byte = serial_port.read(1)
            if not byte:
                continue
            if byte == b"\n":
                response = unpack(bytes(self.buffer))
                self.buffer.clear()
                if response == expected:
                    return
            elif len(self.buffer) < 160:
                self.buffer.extend(byte)
            else:
                self.buffer.clear()
        raise TimeoutError(f"ESP did not acknowledge {payload.split(',')[0]}")

    def send_state(self, serial_port, mask, ttl):
        self.sequence += 1
        self.exchange(
            serial_port,
            f"STATE,{self.session},{self.sequence},{mask},{ttl}",
            f"ACK,{self.session},{self.sequence},{mask}",
        )

    def run(self):
        serial_port = None
        serial = None
        try:
            import serial

            serial_port = serial.Serial(
                self.port, 115200, timeout=0.02, write_timeout=1.0, exclusive=True,
                xonxoff=False, rtscts=False, dsrdtr=False,
            )
            # USB-UART boards may reset when the port opens.
            if self.stop_event.wait(2.0):
                return
            serial_port.reset_input_buffer()
            for attempt in range(6):
                try:
                    self.exchange(serial_port, f"HELLO,{self.session}", f"READY,{self.session}")
                    break
                except TimeoutError:
                    if attempt == 5 or self.stop_event.is_set():
                        raise
            self.send_state(serial_port, 0, self.ttl_ms)
            self.ready.set()
            LOG.info("ESP connected on %s; outputs initially clear", self.port)
            while not self.stop_event.is_set():
                self.send_state(serial_port, *self.current_state(time.monotonic()))
                self.stop_event.wait(0.1)
        except Exception as error:
            self.error = error
            LOG.error("ESP communication failed: %s", error)
            if serial is not None and isinstance(error, serial.SerialTimeoutException):
                LOG.error("USB write stalled on %s. Check kernel USB disconnect/reset logs, "
                          "the data cable and power supply; an ESP ACK timeout is a separate error.", self.port)
        finally:
            if serial_port is not None:
                try:
                    if self.error is not None:
                        # A failed write may have left an incomplete packet.
                        serial_port.reset_output_buffer()
                        serial_port.write(b"\n")
                    self.send_state(serial_port, 0, self.ttl_ms)
                except Exception:
                    LOG.warning("Could not confirm outputs off; ESP watchdog will expire them")
                serial_port.close()
            self.ready.set()

    def close(self):
        self.stop_event.set()
        self.join(timeout=7.0)


def select_serial_port(requested=None, ports=None):
    """Select one USB serial device; never guess between multiple devices."""
    if requested:
        return requested
    if ports is None:
        try:
            from serial.tools import list_ports
        except ImportError as error:
            raise RuntimeError("USB serial requires pyserial. Run: python -m pip install pyserial") from error
        ports = list_ports.comports()
    candidates = {
        port.device: port for port in ports
        if port.vid is not None or port.device.startswith(("/dev/ttyUSB", "/dev/ttyACM"))
    }
    if not candidates:
        raise RuntimeError(
            "No USB serial device found. Connect the ESP32 using a USB data cable. "
            "List ports with: python -m serial.tools.list_ports . "
            "You can also specify --serial-port /dev/ttyUSB0 (use the actual ESP port)."
        )
    if len(candidates) > 1:
        choices = "; ".join(f"{device} ({port.description})" for device, port in sorted(candidates.items()))
        raise RuntimeError(f"Multiple USB serial devices found: {choices}. Select the ESP with --serial-port PORT.")
    selected = next(iter(candidates))
    LOG.info("Automatically selected USB serial device: %s", selected)
    return selected


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("yolo11n.onnx"))
    parser.add_argument("--serial-port", help="ESP port; automatically selects a single connected USB serial device if omitted")
    parser.add_argument("--usb-index", type=int, default=0)
    parser.add_argument("--inference-threads", type=int, default=2,
                        help="ONNX CPU threads (default: 2); benchmark 1–4 on the Pi")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--ttl-ms", type=int, default=1500, help="Maximum detection age; 100–5000 ms")
    parser.add_argument("--output", type=Path, help="Optional annotated AVI; no recording by default")
    parser.add_argument("--show-window", action="store_true",
                        help="Display detections in a desktop window; Q or Esc stops the program")
    args = parser.parse_args(argv)
    if not 0 < args.conf <= 1 or not 0 <= args.iou <= 1:
        parser.error("--conf must be in (0, 1]; --iou must be in [0, 1]")
    if args.confirm_frames < 1 or not 100 <= args.ttl_ms <= 5000 or args.usb_index < 0:
        parser.error("Require confirm-frames >= 1, ttl-ms 100–5000, usb-index >= 0")
    if not 1 <= args.inference_threads <= 64:
        parser.error("--inference-threads must be between 1 and 64")
    return args


def run(args):
    serial_port = select_serial_port(args.serial_port)
    # Reuse existing camera/inference utilities without invoking their GUI entry point.
    import autonom as vision

    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())

    session = vision.create_session(args.model.expanduser().resolve(),
                                    threads=args.inference_threads, allow_spinning=False)
    LOG.info("Inference threads: %d; ONNX worker spinning disabled", args.inference_threads)
    model_input = session.get_inputs()[0]
    metadata = session.get_modelmeta().custom_metadata_map
    names = model_class_names(metadata)
    width, height = vision.model_input_size(model_input, session)
    policy = DetectionPolicy(names, args.confirm_frames)
    LOG.info("Model loaded; classes: %s", ", ".join(names))
    LOG.info("Model input: %s; detection lifetime: %d ms; confirmation: %d frames",
             model_input.shape if hasattr(model_input, "shape") else (width, height),
             args.ttl_ms, args.confirm_frames)
    camera = writer = link = None
    window_open = False
    window_name = "Road detection - Q or Esc to quit"
    try:
        if args.show_window:
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                raise RuntimeError("--show-window requires a desktop display. Run from the Pi desktop terminal.")
            try:
                vision.cv2.namedWindow(window_name, vision.cv2.WINDOW_NORMAL)
                window_open = True
            except vision.cv2.error as error:
                raise RuntimeError("Cannot open preview. Use GUI-enabled OpenCV, not opencv-python-headless.") from error
        link = ESPLink(serial_port, args.ttl_ms)
        link.start()
        while not link.ready.wait(0.1):
            if stop.is_set():
                return
        if link.error:
            raise RuntimeError(f"ESP startup failed: {link.error}") from link.error
        camera = vision.open_usb_camera(640, 480, 15, args.usb_index)
        previous_mask = None
        stale_reported = False
        last_timing_log = None
        frame_number = 0
        while not stop.is_set():
            if link.error:
                raise RuntimeError("ESP link lost; stopping detection") from link.error
            captured_at = time.monotonic()
            success, frame = camera.read()
            capture_finished = time.monotonic()
            if not success or frame is None or not frame.size:
                link.publish(0, time.monotonic())
                raise RuntimeError("USB camera capture failed")
            tensor, scale, pad_x, pad_y = vision.preprocess(frame, width, height, model_input.type)
            inference_started = time.monotonic()
            outputs = session.run(None, {model_input.name: tensor})
            inference_finished = time.monotonic()
            if not outputs or not all(vision.np.isfinite(output).all() for output in outputs):
                raise RuntimeError("Model returned empty or non-finite outputs")
            # Preserve the detection axis when an export returns just one row.
            predictions = vision.np.asarray(outputs[0])
            if predictions.ndim == 3 and predictions.shape[0] == 1:
                predictions = predictions[0]
            if predictions.ndim != 2:
                raise RuntimeError(f"Unsupported model output shape: {predictions.shape}")
            frame_height, frame_width = frame.shape[:2]
            if vision.is_end_to_end_output(predictions, metadata):
                detections = vision.process_end_to_end(
                    predictions, args.conf, scale, pad_x, pad_y, frame_width, frame_height,
                )
            else:
                detections = vision.process_traditional(
                    predictions, args.conf, args.iou, scale, pad_x, pad_y, frame_width, frame_height,
                )
            processed_at = time.monotonic()
            age_ms = (processed_at - captured_at) * 1000
            stale = age_ms >= args.ttl_ms - 100
            mask = policy.update([] if stale else detections)
            if stale and not stale_reported:
                LOG.warning(
                    "Frame age %.0f ms exceeds usable lifetime %d ms; class signals cleared "
                    "(raw detections: %d). See timing logs before adjusting --ttl-ms.",
                    age_ms, args.ttl_ms - 100, len(detections),
                )
            stale_reported = stale
            frame_number += 1
            if last_timing_log is None or processed_at - last_timing_log >= 5:
                raw_names = sorted({names[class_id] for _, _, class_id in detections
                                    if 0 <= class_id < len(names)})
                LOG.info(
                    "Vision frame %d: capture=%.0f ms preprocess=%.0f ms inference=%.0f ms "
                    "postprocess=%.0f ms total=%.0f ms; detected=%s; state=%s; mask=%d",
                    frame_number, (capture_finished - captured_at) * 1000,
                    (inference_started - capture_finished) * 1000,
                    (inference_finished - inference_started) * 1000,
                    (processed_at - inference_finished) * 1000, age_ms,
                    ",".join(raw_names) or "none", "expired" if stale else "fresh", mask,
                )
                last_timing_log = processed_at
            if stop.is_set():
                break
            link.publish(mask, captured_at)
            if mask != previous_mask:
                LOG.info("Class outputs: %s", ", ".join(name for i, name in enumerate(CLASSES) if mask & (1 << i)) or "none")
                previous_mask = mask
            if args.output or args.show_window:
                vision.draw_detection_boxes(frame, detections, names)
            if args.output:
                if writer is None:
                    writer = vision.create_video_writer(args.output.expanduser().resolve(), frame, 15)
                writer.write(frame)
            if args.show_window:
                vision.cv2.imshow(window_name, frame)
                if (vision.cv2.waitKey(1) & 0xff) in (ord("q"), 27):
                    break
    finally:
        # Clear outputs before potentially slow camera/video cleanup.
        if link is not None:
            link.close()
        if camera is not None:
            camera.close()
        if writer is not None:
            writer.release()
        if window_open:
            vision.cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = parse_arguments()
    try:
        run(arguments)
    except Exception:
        LOG.exception("Headless control stopped")
        raise SystemExit(1)
