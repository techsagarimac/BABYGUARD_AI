# BabyGuard AI — Project Documentation

**Title:** BabyGuard AI: Camera-Based Intelligent Baby Monitoring and Alert System  
**Type:** Undergraduate major project (computer science / AI / computer vision)  
**Version:** 1.0 — local webcam only  

This document is written in the style of an engineering college project report. It describes an educational prototype. It is not a clinical evaluation and not a product safety certificate.

---

## 1. Abstract

Continuous visual supervision of an infant is tiring and incomplete when a caregiver must also do other work. Commercial baby monitors often add sensors, cloud accounts, or audio channels. This project studies a simpler question: can a standard laptop webcam, processed entirely offline, provide a useful **visual monitoring aid**?

BabyGuard AI captures frames with OpenCV, detects a person with a lightweight YOLO nano model, estimates body landmarks with MediaPipe Pose, and classifies movement from a short rolling history of landmark or bounding-box displacement. A rule-based layer estimates a cautious activity state (possibly sleeping, possibly awake, or active) and raises dashboard alerts when high motion persists, the subject leaves a user-defined rectangle, or the subject disappears after being visible. Optional local text-to-speech can speak a non-medical sentence such as “Possible unusual activity detected.”

The system is modular, runs on a normal college laptop, and is designed so a student can explain every stage in a viva. Accuracy is limited by lighting, occlusion, and the fact that generic person detection is not infant identification. The software must never be treated as a substitute for adult supervision.

---

## 2. Introduction

Computer vision is the field that enables computers to extract information from images and video. A webcam already produces a dense stream of pixels. Without analysis, those pixels are only a picture. With detection and tracking, they can become structured status: “a person is present”, “the body is mostly still”, “the subject crossed a boundary”.

Baby monitoring is a socially familiar application, which makes it suitable for a major-project demonstration. It is also an ethically sensitive one. Version 1 therefore uses only a camera, keeps all processing on the laptop, and uses conservative language on the dashboard. The engineering goal is a **stable, explainable pipeline**, not a claim of clinical reliability.

The implementation follows a classic student-friendly stack:

- Python for the application
- OpenCV for capture and drawing
- Ultralytics YOLO for person detection
- MediaPipe for pose landmarks
- Tkinter for a desktop dashboard

---

## 3. Problem Statement

Parents and caregivers cannot watch a crib continuously. A plain camera feed still requires a human to interpret it. Hardware-heavy academic projects often attach an ESP32, a DHT11 temperature sensor, and a cloud dashboard. Those extras increase wiring complexity and shift attention away from vision algorithms.

The problem addressed here is:

**Design a real-time, camera-only desktop application that detects a subject, estimates pose and movement, presents an honest activity state, and raises visual (and optional voice) alerts for unusual visual conditions, without external sensors or cloud services.**

Sub-problems include keeping the GUI responsive, avoiding one-frame false alarms, loading a YOLO model from a local folder, and failing gracefully when a webcam or library is missing.

---

## 4. Objectives

1. Capture a live webcam stream with OpenCV and handle camera failure without crashing.
2. Detect a person in the frame using a lightweight YOLO model and draw a bounding box.
3. Estimate body landmarks with MediaPipe Pose and draw a readable skeleton.
4. Compute movement from a rolling window of landmark or box motion and label it LOW, NORMAL, or HIGH.
5. Estimate a visual activity state: possibly sleeping, possibly awake, or active.
6. Allow the user to draw a rectangular monitoring zone and detect when the subject leaves it.
7. Raise “possible unusual activity” and zone/missing-subject alerts only after persistence and cooldown.
8. Display all results on a clean Tkinter dashboard with an event log.
9. Document limitations clearly for academic evaluation.

**Non-objectives (Version 1):** medical diagnosis, cry detection, temperature/humidity sensing, cloud notifications, multi-camera fusion, and dedicated baby-class recognition.

---

## 5. Existing System

Typical existing approaches seen in student literature and consumer products:

| Approach | Typical parts | Limitation for this project |
| --- | --- | --- |
| Hardware baby monitor | Camera + radio/Wi-Fi + parent unit | Closed system; little algorithm work to show in a viva |
| IoT crib project | ESP32, DHT11, sound sensor, MQTT | Shifts effort to wiring and cloud accounts |
| Cloud smart camera | Vendor app, remote servers | Privacy and dependency; not fully explainable |
| Raw OpenCV motion | Frame differencing only | Cannot separate a person from a curtain moving |

What is missing in many academic IoT demos is a **clear vision pipeline** that a student can justify: detection, landmarks, temporal smoothing, and honest alerts.

---

## 6. Proposed System

BabyGuard AI is a single-machine desktop application.

**Inputs:** one webcam.  
**Outputs:** annotated video, status cards, event log, optional beep, optional local speech.  
**User controls:** Start, Stop, Set Area, Reset, Voice Alerts, Clear Log.

The proposed system treats YOLO as a **person/scene detector**, not as a baby classifier. The dashboard therefore says “Baby/Person detected” or “Subject detected”. Pose and movement are computed only from vision. Sleep is labelled **estimated**.

If YOLO cannot run, MediaPipe pose can still imply presence so a demonstration is not completely blocked. The footer shows which backends actually loaded.

---

## 7. System Architecture

The runtime pipeline is:

```
Webcam
  → OpenCV CameraHandler
  → YOLO ObjectDetector (COCO person)
  → MediaPipe PoseDetector
  → MovementAnalyzer (rolling history)
  → ActivityAnalyzer (heuristic state)
  → MonitoringLogic (persistence rules)
  → AlertManager (cooldown, beep, TTS)
  → Tkinter BabyGuardApp
```

A **background thread** performs capture and inference. The Tkinter main thread only copies the latest annotated frame and updates labels. That split is why the window stays usable while YOLO runs.

Configuration is centralized in `config.py` so thresholds can be changed without editing detector code.

---

## 8. Hardware Requirements

Version 1 hardware is intentionally small:

- Laptop or desktop with a webcam (built-in or USB)
- 8 GB RAM recommended (16 GB more comfortable when PyTorch is installed)
- CPU is sufficient; NVIDIA GPU or Apple MPS is used automatically if it works
- Speakers only if voice alerts or beep are demonstrated
- No microcontroller, no extra environmental sensors

---

## 9. Software Requirements

- Python 3.10 or newer (3.10–3.12 recommended)
- Operating system: macOS, Windows, or Linux
- Packages listed in `requirements.txt`:
  - opencv-python
  - mediapipe
  - ultralytics
  - numpy
  - Pillow
  - pyttsx3
- Ultralytics will install a matching PyTorch build as a dependency
- YOLO nano weights (`yolo11n.pt` or `yolov8n.pt`) in `models/`
- Optional: MediaPipe `pose_landmarker_lite.task` if the Tasks API is used

---

## 10. Modules

| Module | File | Responsibility |
| --- | --- | --- |
| Entry | `main.py` | Adds the project root to `sys.path` and launches the dashboard |
| Config | `config.py` | Camera, model, thresholds, zone, cooldown |
| Camera | `camera/camera_handler.py` | Open, read, resize, release, placeholder frame |
| Detection | `detection/object_detector.py` | YOLO person box |
| Pose | `detection/pose_detector.py` | Landmarks, body-position hint, skeleton drawing |
| Movement | `analysis/movement_analyzer.py` | Rolling displacement → LOW/NORMAL/HIGH |
| Activity | `analysis/activity_analyzer.py` | Estimated sleep/awake/active |
| Monitoring | `analysis/monitoring_logic.py` | Unusual motion, zone, disappearance |
| Alerts | `alerts/alert_manager.py` | Visual state, beep, TTS, cooldown |
| UI | `ui/dashboard.py` | Layout, buttons, zone drawing, event log |
| Helpers | `utils/helpers.py` | Geometry and frame-to-PhotoImage |
| Tests | `tests/test_pipeline.py` | Offline unit tests |

---

## 11. Algorithms

### 11.1 Person detection (YOLO)

YOLO (You Only Look Once) is a single-shot detector: the network looks at the whole image and predicts boxes and class scores in one pass. This project uses a **nano** checkpoint so a college laptop can keep roughly 15–30 frames per second at 320-pixel inference size. Only class id 0 (person) is requested.

The largest reasonably confident box is kept so a parent walking past is preferred over a tiny false box.

### 11.2 Pose landmarks (MediaPipe)

MediaPipe Pose returns up to 33 normalized landmarks. The project tracks a subset: nose, shoulders, elbows, wrists, hips, knees, ankles. Connections between those joints are drawn as a stick figure.

Body position is a cheap geometric test: if shoulder width is large compared with shoulder-to-hip height, the torso is labelled **Horizontal**; otherwise **Upright**. This is a demo hint, not a fall detector.

### 11.3 Movement score

For each processed frame the analyser stores key landmark coordinates. The score is the mean Euclidean displacement between the last two sets, then averaged over a history of about ten frames:

```
score = mean_t( mean_i  || p_i(t) − p_i(t−1) || )
```

If landmarks are missing, the bounding-box centre is used instead. Thresholds in `config.py` map the score to LOW, NORMAL, or HIGH.

### 11.4 Estimated state

- HIGH movement → **Active**
- LOW movement for `SLEEP_STILL_SECONDS` (shorter if the torso looks horizontal) → **Possibly Sleeping**
- otherwise → **Possibly Awake**

The word **Possibly** is required. Stillness can mean sleep, a blanket, or a person sitting still for a viva.

### 11.5 Unusual activity and zone logic

A HIGH label must repeat for `HIGH_MOVEMENT_PERSIST_FRAMES` before “possible unusual activity” is raised. A sudden jump of the box centre above `SUDDEN_JUMP_THRESHOLD` can contribute. Leaving the monitoring rectangle must persist for `ZONE_OUTSIDE_PERSIST_FRAMES`. Disappearance is reported only if the subject was visible for at least two seconds and then stays gone for a grace window.

### 11.6 Alert cooldown

Visual banners follow the current decision every frame. Beep and speech fire only when the alert **kind** changes, and only if that kind has not fired inside `ALERT_COOLDOWN_SECONDS`. This prevents a strobe of audio on every frame.

---

## 12. Implementation

Implementation choices that matter for a viva:

1. **No blocking OpenCV window.** `cv2.imshow` is not used. Frames are converted with Pillow (or a PPM fallback) and shown on a Tkinter label.
2. **Worker thread.** Inference never sits in the Tk event callback.
3. **Honest labels.** The detector is COCO person, so the UI says subject / baby-or-person.
4. **Two MediaPipe backends.** Classic `solutions.pose` for tutorial familiarity; Tasks API if solutions are absent.
5. **Model folder.** Weights are resolved from `models/` first; download is allowed only when `YOLO_ALLOW_DOWNLOAD` is true.
6. **Interactive zone.** Mouse drag on the preview is mapped back to normalized frame coordinates, so the rectangle stays correct after letterboxing.
7. **Failure isolation.** Missing OpenCV, YOLO, MediaPipe, camera, or pyttsx3 produces a message instead of a stack-trace exit from the UI path.

---

## 13. Testing

### 13.1 Automated tests

```bash
python -m unittest discover -s tests -v
```

| # | Case | How it is tested |
| --- | --- | --- |
| 1 | Camera available | Manual live test (see 13.2). Automated tests do not require a webcam. |
| 2 | Camera unavailable | `VideoCapture` is mocked to a closed device; `start()` returns False and the message contains “unavailable”. Placeholder frame size is checked. |
| 3 | Subject detected | Synthetic `PersonDetection(detected=True)` status text and centre. |
| 4 | Subject not detected | Synthetic empty detection. |
| 5 | Pose detected | Synthetic 33 landmarks; status “Pose detected”. |
| 6 | Pose unavailable | Empty `PoseResult`. |
| 7 | Low movement | Repeated identical landmarks stay LOW. |
| 8 | High movement | Rapidly shifting landmarks become HIGH. |
| 9 | Monitoring zone violation | Centre inside vs outside a rectangle after persistence frames. |
| 10 | Alert cooldown | Unusual → normal → unusual inside the cooldown does not re-fire audio. |
| 11 | Voice alert disabled | `_speak` is stubbed; it is not called when voice is off. |
| 12 | Missing YOLO model | `find_yolo_weights` on an empty folder returns `None`. |

### 13.2 Manual demonstration tests

1. Start the app without a webcam: dashboard shows **Camera unavailable** after START; the process stays up.
2. Start with a webcam: preview appears, FPS is shown.
3. Enter the frame: subject box and “Detected”.
4. Leave the frame: “Not Detected”; after a short time, missing-subject alert if you were visible long enough.
5. Sit still for `SLEEP_STILL_SECONDS`: estimated state becomes Possibly Sleeping.
6. Wave both arms: movement HIGH, then possible unusual activity.
7. Draw a small zone and step out: outside-area alert.
8. Enable voice: one spoken sentence, not a loop.
9. Disable voice: only visual/beep behaviour.
10. Remove `models/*.pt` and set `YOLO_ALLOW_DOWNLOAD = False`: clear model error in the log.

---

## 14. Results

On a typical college laptop the expected live result is:

- Application window opens from `python main.py`
- After model load, START opens the camera
- A person in view receives a green box and a pose skeleton
- Status cards update without the window locking
- Stillness produces Possibly Awake, then Possibly Sleeping
- Fast motion produces Active and, if it persists, Possible unusual activity
- A drawn zone produces a clear outside-area banner
- Event log records start, detection, activity changes, and alerts

Quantitative “accuracy %” is not claimed. There is no labelled infant dataset in Version 1. Results should be presented as **qualitative demo behaviour** plus unit-test pass/fail.

---

## 15. Limitations

1. Educational prototype only.
2. Does not replace adult supervision.
3. No medical diagnosis, no illness detection, no emergency guarantee.
4. False positives: adults, older children, pets near the crib, camera shake.
5. False negatives: heavy blankets, low light, subject facing away, tiny distant person.
6. YOLO person ≠ baby identity.
7. Sleep state is a stillness heuristic.
8. No audio / cry channel in Version 1.
9. Night performance depends on the webcam; there is no IR illuminator.
10. YOLO + pose together can drop below 15 FPS on older CPUs; `YOLO_IMAGE_SIZE` and `PROCESS_EVERY_N_FRAMES` are the levers.

---

## 16. Future Scope

- Fine-tuned infant detector and age-appropriate pose model
- Microphone cry analysis as a separate optional module
- Multiple cameras and a simple room map
- Local notifications to a phone on the same Wi-Fi (still no medical claim)
- Night-vision USB camera support
- Longer temporal models (HMMs or small classifiers) for “rolling over” vs “arm twitch”
- Optional environmental sensors only if a later hardware phase is approved
- Cloud dashboard only with explicit privacy design

None of these are part of the current delivery.

---

## 17. Conclusion

BabyGuard AI shows that a complete, demonstrable monitoring **aid** can be built from a webcam and a short, explainable vision pipeline. YOLO answers “is there a person?”, MediaPipe answers “where are the joints?”, a rolling average answers “how much did they move?”, and a persistence layer answers “is this worth an alert?”.

The important academic result is not a claim of infant safety. It is a working, modular system that a student can install, run, configure, test, and defend — including its limits.

---

**Disclaimer.** BabyGuard AI is a college demonstration. It must not be used as the only means of watching an infant. Always keep a responsible adult in charge.
