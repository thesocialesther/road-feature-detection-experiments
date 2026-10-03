# Headless vision → USB serial → ESP32

`main.py` runs the USB camera and ONNX detector without opening a window. It
reuses inference/camera helpers in `autonom.py`, so keep both files together.
The shared traditional-output postprocessor uses a NumPy confidence prefilter.
The ESP32 sketch receives all five class
signals; your partner implements the machine actions in `handleClassSignals()`.
There is no assumed steering, braking, motor direction, or actuator pin mapping.

## Raspberry Pi setup

Use a separate environment for this headless runner to avoid mixing GUI and
headless OpenCV packages in the existing environment:

```bash
cd /home/autonomous/bumblebee
python3 -m venv .venv-headless
.venv-headless/bin/python -m pip install -r requirements-headless.txt
ls /dev/serial/by-id/
```

Connect a USB webcam and the ESP32 using a USB **data** cable. If exactly one USB
serial device is connected, it is selected automatically:

```bash
.venv-headless/bin/python main.py
```

If several USB serial devices are connected, the runner lists them and requires
an explicit choice. Use the ESP's actual serial device path below; the camera
index is independent of that port:

```bash
.venv-headless/bin/python main.py \
  --serial-port /dev/serial/by-id/REPLACE_WITH_YOUR_ESP_DEVICE \
  --usb-index 0
```

If the ESP has no by-id entry, use its actual `/dev/ttyUSB0` or `/dev/ttyACM0`
device. Close Arduino Serial Monitor before starting Python. On Raspberry Pi OS,
if opening the serial device gives permission denied, add your user to `dialout`
(`sudo usermod -aG dialout "$USER"`) and log out/in.

The default model is `yolo11n.onnx` beside `main.py`. The runner requires model
metadata containing exactly the five class names below and maps by name, so
reordered IDs are supported. It refuses an unknown model instead of guessing
labels. No retraining, internet connection, Ultralytics, or Pi camera libraries
are required at runtime. Install dependencies before running offline.

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--serial-port` | auto | Select the only USB serial device, or specify the ESP path |
| `--model` | adjacent `yolo11n.onnx` | Road detection ONNX model |
| `--usb-index` | `0` | USB webcam `/dev/video` index |
| `--inference-threads` | `2` | ONNX CPU threads; benchmark 1–4 on your Pi |
| `--conf` | `0.3` | Minimum detection confidence |
| `--iou` | `0.45` | NMS threshold for traditional exports |
| `--confirm-frames` | `2` | Consecutive frames before asserting a class |
| `--ttl-ms` | `1500` | Detection lifetime, including host processing time |
| `--output` | none | Optional annotated MJPEG AVI, recorded at fixed 15 FPS |
| `--show-window` | off | Show annotated camera feed; Q or Esc exits |

To preview detections, run `python main.py --show-window` from a terminal in the
Pi desktop session. This requires GUI-enabled OpenCV. If your environment uses
the headless package, replace it with `python -m pip uninstall opencv-python-headless`
followed by `python -m pip install opencv-python` (do not install multiple OpenCV
variants together). The window updates after each inference. Q or Esc closes the
runner and clears ESP requests; Ctrl+C also works. Omit the flag for headless use.

No class is filtered out. Multiple simultaneous classes are transmitted together.
Use `--confirm-frames 1` to signal on the first detected frame. Absence clears a
class immediately. These signals mean “class present in this image,” not distance,
collision likelihood, or a steering decision. No ROI or spatial control policy
is applied. Optional video uses fixed playback timing and can play faster than
real time when inference runs below 15 FPS.

## ESP32 firmware and partner handoff

1. Install the Espressif ESP32 board package in Arduino IDE and select your exact
   board and USB port.
2. Open `esp/road_actuation/road_actuation.ino` and upload it.
3. Leave `OUTPUTS_ENABLED = false` for communication-only operation. This does
   **not** disable receiving signals, the callback, or acknowledgements.
4. Your partner adds actuator control inside `handleClassSignals(uint8_t mask)`.
   The function is called for every accepted state and when requests are cleared
   at startup, a new connection, or watchdog expiry. Keep it nonblocking.

For boards using a USB-to-UART connector, select that connector and disable
“USB CDC On Boot” if that option is present. For supported boards using native
USB, enable USB CDC so Arduino `Serial` is routed to the connected port. See
[Espressif serial troubleshooting](https://docs.espressif.com/projects/arduino-esp32/en/latest/troubleshooting.html)
and [USB CDC setup](https://docs.espressif.com/projects/arduino-esp32/en/latest/tutorials/cdc_dfu_flash.html).

The protocol uses a five-bit mask with this **fixed order**, independent of model
class IDs:

| Class | Bit | Value | Partner can read |
| --- | --- | --- | --- |
| `cracks` | 0 | 1 | `mask & 0x01` |
| `speed_bumps` | 1 | 2 | `mask & 0x02` |
| `potholes` | 2 | 4 | `mask & 0x04` |
| `left_road_boundaries` | 3 | 8 | `mask & 0x08` |
| `right_road_boundaries` | 4 | 16 | `mask & 0x10` |

For example, mask `5` means cracks and potholes; `31` means all classes; `0`
clears all class requests. Repeated states should not repeatedly start a one-shot
action: compare with the prior mask if the machine needs rising-edge triggers.
Define the machine's safe response to cleared requests in the callback. An ACK
confirms reception and callback return, not physical actuator completion.

The sketch also contains optional maintained digital outputs: configure
`CLASS_PINS`, `ACTIVE_HIGH`, and enable `OUTPUTS_ENABLED` if those outputs fit
your partner's hardware. Pins are unassigned by default. GPIO signals need the
appropriate driver interface for the actual actuator.

## Wire protocol

115200 baud, 8 data bits, no parity, 1 stop bit. Every ASCII line is:

```text
PAYLOAD*CCCC\n
```

`CCCC` is four hexadecimal digits of CRC-16/CCITT-FALSE over the payload bytes
(polynomial `0x1021`, initial value `0xFFFF`, no reflection, no final XOR).
The `\n` above means a literal newline byte. Messages are:

```text
Pi  → ESP: HELLO,<16-lowercase-hex-session>
ESP → Pi:  READY,<session>
Pi  → ESP: STATE,<session>,<sequence>,<mask>,<ttl_ms>
ESP → Pi:  ACK,<session>,<sequence>,<mask>
```

These examples show payloads; the checksum and newline must be appended. Python
`main.packet(payload)` produces complete wire bytes. A HELLO clears all requests
and resets sequencing. Each STATE must have a strictly increasing sequence,
mask 0–31, TTL 100–5000 ms, matching session and valid checksum. Invalid messages
are ignored and do not refresh the watchdog. The CRC detects corruption; it is
not authentication.

Python sends the latest state about every 100 ms plus ACK latency. It waits for
the matching ACK before sending another state and never queues detection history.
A dedicated serial thread keeps camera/inference work independent of ACK waits.
Serial writes and ACK waits each have a one-second timeout. A write timeout
means the host write stalled, not that a class was missing. The runner still
stops on link failure; it does not retry active commands. Before its final
zero-state attempt, it discards queued host output and terminates any partial
line so the ESP can parse the zero-state message.

The headless runner uses two inference threads by default and disables idle
worker spinning to reduce CPU contention. Thread count can affect inference
speed; compare the timing logs on the actual Pi. See
[ONNX Runtime thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
Refreshes contain the **remaining** detection lifetime, so a frozen detector
cannot keep an old class active indefinitely. Transport and ESP scheduling add
small latency to the host age bound. If frames are too slow for the configured
lifetime, the runner logs a warning and sends zero; measure latency before
increasing the TTL.

Timing logs print on the first processed frame and approximately every five
seconds afterward. They separate camera capture, preprocessing, inference and
postprocessing, and show raw detected class names independently of the confirmed
signal mask. `state=expired` means processing exceeded `ttl_ms - 100` (100 ms is
reserved for communication). `mask=0` can therefore mean no detections, pending
confirmation, or expired detections; use these fields to distinguish them.

If the ESP connects but warnings repeat, USB communication is already working.
Compare first-frame timing with later frames to identify warm-up versus sustained
latency. For a communication-only bench run, `--ttl-ms 5000 --confirm-frames 1`
allows slower frames and signals on their first detection. This also permits
older detections and longer held requests, so choose the operating lifetime with
the partner implementing machine control. Frames above about 4.9 seconds still
expire. Changing camera resolution alone does not shrink a fixed-size ONNX model
input; sustained excessive inference time calls for a suitable smaller model
export or faster inference hardware.

For repeated USB write timeouts, run `sudo dmesg --follow` in another terminal
while reproducing the issue and look for disconnects or USB resets. On Raspberry
Pi, `vcgencmd get_throttled` provides additional power/throttling diagnostics.
Compare with the communication-only bench check below: if that also fails,
vision processing is not required to reproduce the fault. Check the USB data
cable, connectors, and power supply. Save the kernel messages alongside the
Python logs rather than assuming the timeout is caused by inference.

The ESP clears requests locally when its TTL expires, including if Python exits
abruptly or USB is unplugged. Python attempts an acknowledged zero on shutdown
or failure. Camera failure and lost ACKs stop the runner; automatic reconnection
is intentionally not attempted. A new run establishes a new session. Ctrl+C and
SIGTERM request clean shutdown.

## Communication-only bench check

After flashing, this sends each class separately, then all classes, without
requiring a camera or loading the model. It sends actual class requests: use
the default communication-only firmware or an appropriate bench setup.

```bash
.venv-headless/bin/python - /dev/serial/by-id/REPLACE_WITH_YOUR_ESP_DEVICE <<'PY'
import sys
import time
from main import CLASSES, ESPLink

link = ESPLink(sys.argv[1])
link.start()
try:
    if not link.ready.wait(8) or link.error:
        raise RuntimeError(f"ESP connection failed: {link.error}")
    for mask, label in [(1 << i, name) for i, name in enumerate(CLASSES)] + [(31, "all"), (0, "clear")]:
        print(f"Sending {label}: mask={mask}", flush=True)
        until = time.monotonic() + 2
        while time.monotonic() < until:
            if link.error:
                raise RuntimeError(f"ESP link failed: {link.error}")
            link.publish(mask, time.monotonic())
            time.sleep(0.1)
finally:
    link.close()
if link.error:
    raise RuntimeError(f"ESP link failed: {link.error}")
print("Completed; ESP acknowledged the state stream.")
PY
```

## Verification

```bash
python3 -B -m unittest discover -s tests -v
```

Tests cover class mapping, simultaneous signals, confirmation/clearing, stale
data, CRC corruption, ACK matching and timeout. A C++ host harness compiles the
actual ESP sketch (requires `g++`) and feeds Python-generated protocol messages
to check acceptance, replay/session rejection, invalid/oversized packets, safe
reconnect, and watchdog expiry. This does not replace compiling for your exact
ESP32 board or checking a physical USB camera, serial connection and actuators.
