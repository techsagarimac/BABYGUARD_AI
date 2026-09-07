# BabyGuard AI

**Camera-Based Intelligent Baby Monitoring and Alert System**

Version 1 — local webcam only.

BabyGuard AI is an educational computer-vision project. It watches a laptop camera, detects a person in the scene, estimates body pose and movement, and shows visual alerts when activity looks unusual or when the subject leaves a user-drawn monitoring area.

It does **not** replace adult supervision. It does **not** diagnose illness, detect crying, or guarantee baby safety.

---

## Problem Statement

A caregiver cannot stare at a crib every second. Cameras already exist in many homes, but a raw video feed still requires a person to watch it. For a college project, the useful question is:

> Can a local computer-vision pipeline turn a webcam into a **visual monitoring aid** that reports presence, movement, and simple activity estimates?

The constraints for Version 1 are deliberate:

- laptop or desktop only
- one webcam
- no Arduino, ESP32, temperature, humidity, or other sensors
- no microphone and no cloud service
- honest labels (generic person detection is not baby identification)

---

## Proposed Solution

BabyGuard AI runs entirely on the student’s computer.

1. OpenCV captures webcam frames.
2. A lightweight YOLO nano model looks for the COCO **person** class.
3. MediaPipe Pose finds body landmarks (nose, shoulders, elbows, wrists, hips, knees, ankles).
4. A rolling-window analyser classifies movement as LOW / NORMAL / HIGH.
5. A heuristic estimates **Possibly Sleeping**, **Possibly Awake**, or **Active**.
6. Monitoring rules watch for persistent high motion, a sudden position jump, leaving the monitoring rectangle, or disappearing after being seen.
7. A Tkinter dashboard shows the live view, status cards, alerts, and an event log.
8. Optional local text-to-speech (pyttsx3) can speak a cautious alert sentence.

If the system notices unusual **visual** activity it says:

**Possible unusual activity**

It never says that a baby is in danger or that a medical emergency was detected.

---

## Features

- Live webcam preview that does not freeze the GUI
- YOLO person / subject detection with a bounding box
- MediaPipe pose landmarks and a simplified skeleton
- Body-position hint: Upright or Horizontal
- Movement classification with a short frame history
- Estimated sleep / awake / active state (visual heuristic only)
- Optional rectangular monitoring area (draw on the camera or set in `config.py`)
- Persistent-condition alerts with cooldown
- Optional beep and optional voice alerts
- Event log with clear-log control
- Friendly error handling when the camera, YOLO, MediaPipe, or TTS is missing
- Unit tests for the analysis rules

---

## Technologies

| Component | Role |
| --- | --- |
| **Python 3.10+** | Application language. 3.10–3.12 is the most reliable range for MediaPipe and Ultralytics. |
| **OpenCV** | Webcam capture, resize, drawing, colour conversion. |
| **YOLO (Ultralytics)** | Real-time object detection. This project uses a nano model (`yolo11n.pt` or `yolov8n.pt`) and the COCO *person* class. |
| **MediaPipe** | Pose landmark detection. Supports the classic `mp.solutions.pose` API and the newer Pose Landmarker task API. |
| **NumPy** | Landmark and bounding-box maths. |
| **Tkinter** | Desktop dashboard. |
| **Pillow** | Convert OpenCV frames to Tkinter images. |
| **pyttsx3** | Optional offline speech. If it is missing, visual alerts still work. |

No cloud API, no database, and no extra deep-learning framework beyond what Ultralytics already needs.

---

## Architecture

```
                          +----------------------+
                          |   Laptop webcam      |
                          +----------+-----------+
                                     |
                                     v
                          +----------------------+
                          |  OpenCV capture      |
                          |  camera_handler.py   |
                          +----------+-----------+
                                     |
                                     v
                          +----------------------+
                          | YOLO person detect   |
                          | object_detector.py   |
                          +----------+-----------+
                                     |
                                     v
                          +----------------------+
                          | MediaPipe Pose       |
                          | pose_detector.py     |
                          +----------+-----------+
                                     |
                                     v
                          +----------------------+
                          | Landmark / box maths |
                          | movement_analyzer.py |
                          +----------+-----------+
                                     |
                                     v
                          +----------------------+
                          | Activity estimate    |
                          | activity_analyzer.py |
                          +----------+-----------+
                                     |
                                     v
                          +----------------------+
                          | Monitoring rules     |
                          | monitoring_logic.py  |
                          +----------+-----------+
                                     |
                                     v
                          +----------------------+
                          | Alert manager        |
                          | alert_manager.py     |
                          +----------+-----------+
                                     |
                                     v
                          +----------------------+
                          | Tkinter dashboard    |
                          | dashboard.py         |
                          +----------------------+
```

`python main.py` starts the dashboard. The other files exist only for organisation and testing.

---

## Project structure

```
BABYGUARD_AI/
├── main.py
├── requirements.txt
├── README.md
├── PROJECT_DOCUMENTATION.md
├── VIVA_NOTES.md
├── config.py
├── camera/
│   ├── __init__.py
│   └── camera_handler.py
├── detection/
│   ├── __init__.py
│   ├── object_detector.py
│   └── pose_detector.py
├── analysis/
│   ├── __init__.py
│   ├── movement_analyzer.py
│   ├── activity_analyzer.py
│   └── monitoring_logic.py
├── alerts/
│   ├── __init__.py
│   └── alert_manager.py
├── ui/
│   ├── __init__.py
│   └── dashboard.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── models/
│   └── README.md
├── data/
│   └── README.md
└── tests/
    ├── __init__.py
    └── test_pipeline.py
```

---

## Installation

Use Python **3.10, 3.11, or 3.12** if you can. Python 3.13 may fail to install MediaPipe or PyTorch.

```bash
cd BABYGUARD_AI
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

The first launch may download `yolo11n.pt` into `models/` (about 5–6 MB). That needs internet once. After that the project runs offline.

If the download is blocked, place `yolo11n.pt` or `yolov8n.pt` in `models/` yourself.

---

## Running

```bash
python main.py
```

1. Wait for the status line to say the models are loaded.
2. Click **START**.
3. Stand (or sit) in the camera view. A green box and pose skeleton should appear.
4. Click **SET AREA** and drag a rectangle around the crib / chair / demo zone.
5. Step outside the rectangle to trigger the outside-area alert.
6. Wave both arms quickly for about one second to trigger **Possible unusual activity**.
7. Stay still for several seconds to see **Possibly Sleeping**.
8. Optional: tick **Voice Alerts**.

### Demo video (no baby required)

For the college presentation you can play a looping clip instead of the webcam.

1. Copy a short `.mp4` into `data/` (for example `data/demo_baby.mp4`).
2. Click **DEMO VIDEO**.
3. If `data/` has exactly one video, it starts immediately. Otherwise pick the file.
4. The overlay shows **DEMO CLIP** so the examiner knows it is not a live infant.
5. Click **LIVE** (or **START**) to return to the webcam.

Use a clip you are allowed to show: royalty-free stock footage of a baby in a crib, a doll, or a student in front of the camera. The pipeline is the same as live monitoring.

You can also set a default path in `config.py`:

```python
DEMO_VIDEO_PATH = "data/demo_baby.mp4"
DEMO_VIDEO_LOOP = True
```

Keyboard: `Space` starts monitoring, `Esc` stops it.

### macOS crash (Abort trap / MediaPipe)

On some Apple Silicon Macs, the old `mp.solutions.pose` graph dies inside TensorFlow Lite:

`Feedback manager requires a model with a single signature`  
`Check failed: service_ Service is unavailable.`

Python cannot catch that C++ `abort()`. BabyGuard now:

1. Skips the crashing `solutions.pose` path on macOS
2. Runs MediaPipe PoseLandmarker in a **separate process** (away from YOLO/PyTorch)
3. If Pose still cannot start, uses an **approximate skeleton from the person box** and keeps the dashboard open

If the window still does not appear, force the box fallback:

```bash
# in config.py
POSE_BACKEND = "box"
```

Then run `python main.py` again.

---

## Configuration

Edit `config.py`. Useful keys:

| Variable | Meaning |
| --- | --- |
| `CAMERA_INDEX` | `0` is usually the built-in webcam. Try `1` for a USB camera. |
| `YOLO_CONFIDENCE` | Minimum person-box score (default `0.40`). |
| `YOLO_IMAGE_SIZE` | `320` is faster; `416` or `640` is slower and a bit sharper. |
| `MOVEMENT_LOW_THRESHOLD` / `MOVEMENT_HIGH_THRESHOLD` | Landmark motion cut-offs. |
| `SLEEP_STILL_SECONDS` | How long LOW motion must last before “Possibly Sleeping”. |
| `HIGH_MOVEMENT_PERSIST_FRAMES` | HIGH frames required before an unusual-activity alert. |
| `ALERT_COOLDOWN_SECONDS` | Minimum time between repeated beep/voice alerts of the same type. |
| `MONITORING_ZONE_ENABLED` | Start with the rectangle defined below. |
| `MONITORING_ZONE` | Normalized `(x1, y1, x2, y2)` in the range 0–1. |
| `VOICE_ALERTS_ENABLED` | Default for the Voice Alerts checkbox. |
| `YOLO_ALLOW_DOWNLOAD` | If `False`, a missing `.pt` file shows an error instead of downloading. |

---

## Monitoring zone

Two ways to set it:

1. **Dashboard (recommended for the demo)**  
   Click **SET AREA**, then click-and-drag on the live camera. A cyan/gold rectangle is drawn. If the subject’s bounding-box centre leaves that rectangle for several frames, the alert becomes **Subject outside monitoring area**.

2. **config.py**  
   Set `MONITORING_ZONE_ENABLED = True` and edit `MONITORING_ZONE = (0.12, 0.08, 0.88, 0.92)`.

**RESET** clears the drawn area and the analysis history. **CLEAR LOG** only clears the event list.

The zone is optional. If it is disabled, outside-area alerts are not raised.

---

## Testing

```bash
python -m unittest discover -s tests -v
```

The unit tests cover camera failure handling, subject/pose result shapes, low and high movement, zone violation, alert cooldown, voice-disabled behaviour, and a missing YOLO weight file. They do not require a webcam.

Manual live checks are listed in `PROJECT_DOCUMENTATION.md` (Testing section).

---

## Limitations

- This is an **educational prototype**, not a baby monitor product.
- It does **not** replace an adult in the room.
- It does **not** provide medical diagnosis or emergency detection.
- Visual analysis produces **false positives and false negatives** (pets, adults, blankets, poor light, motion blur).
- Generic YOLO person detection does **not** prove the subject is a baby.
- Sleep / awake / active is only an **estimate** from movement and pose.
- There is no cry detection, no temperature reading, and no night-vision hardware.
- Performance depends on the laptop. A nano model at 320 px is the intended setting.

---

## Future enhancements

These are **not** implemented in Version 1:

- A dedicated infant-detection model and infant pose dataset
- Cry / audio detection (microphone)
- Multi-camera support
- Mobile push notifications
- Cloud dashboard
- Infrared / night-vision camera
- Stronger activity classification (rolling over, sitting up)
- Optional IoT sensors (temperature, humidity) if a later version needs them

---

## License / academic use

Built as a college major-project demonstration. Keep the disclaimer visible in the report and the viva.
