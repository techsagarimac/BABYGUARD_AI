# BabyGuard AI — Viva Notes

Short answers you can say out loud. Do not claim medical accuracy.

---

## Core ideas (student language)

### What is computer vision?

Computer vision is the set of methods that let a program take an image or a video frame and pull out useful information — “there is a person here”, “the wrist moved”, “the box left this rectangle”. The computer does not *understand* the baby the way a parent does. It only measures pixels, shapes, and motion.

### Why OpenCV?

OpenCV is the standard library for talking to a webcam, resizing frames, converting BGR to RGB, and drawing boxes and text. Without OpenCV we would write a lot of camera and drawing code ourselves. In this project OpenCV is the **eyes and the pen**, not the brain.

### What is YOLO?

YOLO means **You Only Look Once**. It is an object detector: one neural network looks at the whole frame and predicts bounding boxes plus class names. We use a **nano** model (`yolo11n` or `yolov8n`) so a laptop can run it live. We only keep class **person**. That is why we say “subject” or “baby/person”, not “I identified an infant”.

### Why is Pose in a separate process on Mac?

On some Apple Silicon Macs, MediaPipe’s older Pose graph calls C++ `abort()` (TensorFlow Lite “Feedback manager” / “Service is unavailable”). Python `try/except` cannot catch that. BabyGuard starts Pose in a child process so a crash there does not kill the dashboard. If Pose cannot start at all, movement still uses the YOLO box.

### Why MediaPipe?

MediaPipe (from Google) gives a ready-made **pose** model. It is lighter than training our own keypoint network. It returns body joints as numbers between 0 and 1. Those numbers are easy to use in a college project and easy to explain in a viva.

### What are pose landmarks?

A landmark is one joint or face point. MediaPipe Pose has 33. We care about:

- 0 nose  
- 11–12 shoulders  
- 13–14 elbows  
- 15–16 wrists  
- 23–24 hips  
- 25–26 knees  
- 27–28 ankles  

If we connect shoulders–elbows–wrists and hips–knees–ankles we get a stick figure on the video.

### How is movement calculated?

We save the last few frames of landmark positions. For each joint we compute the distance it moved, then take the mean. We average that score over about 10 frames. Small mean → LOW, medium → NORMAL, large → HIGH. If pose fails, we use the centre of the YOLO box instead.

### How does the alert system work?

The dashboard **always** shows the current status (green / orange / red). Sound and speech are separate: they fire when the alert *type* changes, and only again after a **cooldown** (default 6 seconds). Messages are cautious: “Possible unusual activity”, never “baby in danger”.

### Why use a rolling frame history?

A single frame is noisy. The detector jitters. A blanket moves. If we alerted on one spike, the demo would beep constantly. A short history is a simple low-pass filter. Examiners like this answer.

### Why use thresholds?

Computers need cut-offs. “High movement” is not a feeling; it is `score >= MOVEMENT_HIGH_THRESHOLD` for several frames. Thresholds live in `config.py` so we can tune the demo without rewriting the algorithm. They are chosen for a live presentation, not from a hospital dataset.

### Why is the sleep state only an estimate?

Sleep is a **brain and body state**. We only see pixels. A still person might be asleep, or reading, or hiding under a sheet. So the label is **Possibly Sleeping**. If the examiner asks “can it detect SIDS or illness?” the answer is **no**.

### What are the limitations?

Educational prototype. No adult replacement. No medical claim. Person ≠ baby. False positives and negatives. No cry sound. Bad in the dark. Speed depends on the laptop.

### What can be improved later?

Infant-specific detector, cry audio, better activity classes, night camera, phone notification, more cameras. Not in Version 1.

---

## 20 likely viva questions

**1. What is the aim of your project?**  
To build a local, camera-only visual monitoring aid that detects a subject, estimates pose and movement, and shows honest alerts. It is a college prototype, not a medical device.

**2. Why did you not use Arduino or temperature sensors?**  
Version 1 is a computer-vision project. Extra hardware would hide the AI pipeline. Those can be future work.

**3. Which model detects the person?**  
A YOLO nano model from Ultralytics, COCO class 0 (person).

**4. How do you know it is a baby?**  
We do not. We detect a person-shaped object. The UI says baby/person or subject.

**5. What does MediaPipe give you that YOLO does not?**  
YOLO gives a box. MediaPipe gives joint locations so we can measure body motion and draw a skeleton.

**6. What if pose is not found?**  
The overlay says “Pose not detected”. Movement can still use the bounding-box centre. Alerts that need motion become less precise.

**7. How do you avoid GUI freeze?**  
A background thread reads the camera and runs YOLO/pose. Tkinter only displays the latest frame with `after()`.

**8. What is frame skipping?**  
`PROCESS_EVERY_N_FRAMES` in config. If the laptop is slow we can run detectors every second frame and still refresh the picture.

**9. Explain LOW / NORMAL / HIGH.**  
They are bins of the smoothed motion score. LOW ≈ still, HIGH ≈ large or rapid movement over several frames.

**10. Why not alert on the first HIGH frame?**  
Persistence. `HIGH_MOVEMENT_PERSIST_FRAMES` must be reached so a twitch or a detection jump is ignored.

**11. How does the monitoring area work?**  
The user drags a rectangle. Coordinates are stored as 0–1 fractions of the frame. If the box centre stays outside for several frames we alert “Subject outside monitoring area”.

**12. What is cooldown?**  
After a beep or voice line, the same alert type will not make sound again until `ALERT_COOLDOWN_SECONDS` has passed. The visual banner can stay.

**13. What happens if pyttsx3 is missing?**  
Voice checkbox cannot stay on. Visual alerts and the event log still work.

**14. What happens if the YOLO file is missing?**  
We look in `models/`. If download is allowed, Ultralytics may fetch `yolo11n.pt`. If not, we show a clear error and do not crash the window.

**15. Can it detect crying?**  
No. There is no microphone in Version 1. Anyone who says it detects crying is wrong.

**16. Can it detect fever or breathing problems?**  
No. Camera pixels cannot measure temperature or diagnose illness in this project.

**17. Which Python version should we use?**  
3.10 to 3.12. 3.13 often breaks MediaPipe or PyTorch wheels.

**18. How did you test without a baby?**  
Unit tests use synthetic landmarks and boxes. The live demo uses a student as the subject. That is acceptable because we claim person detection, not infant ID.

**19. What is the difference between estimated state and alert?**  
State is a continuous label (sleep/awake/active). An alert is a **condition that persisted** and should grab attention (unusual motion, outside zone, disappeared).

**20. If you had three more months, what would you add first?**  
A model trained on infants, then optional cry audio, then better night-time capture. I would still keep the “not a medical device” disclaimer.

---

## Demo script (say this while you click)

1. “This is BabyGuard AI. Only a webcam, all processing on this laptop.”  
2. START — “OpenCV is reading the camera.”  
3. Step in — “YOLO found a person. MediaPipe is drawing pose.”  
   If you have no baby, click **DEMO VIDEO** and say: “This is a looping demo clip. Same pipeline as the live camera.”  
4. Move a little — “Movement is NORMAL, estimated state possibly awake.”  
5. SET AREA — drag a box — “This is the monitoring zone.”  
6. Step out — “Subject outside monitoring area. Not ‘danger’, just outside the zone.”  
7. Step in and wave — “High movement for several frames: possible unusual activity.”  
8. Stand still — “If I stay still long enough it will estimate possibly sleeping.”  
9. “Limitations: not medical, not baby ID, not a replacement for a parent.”

If something fails (no pose in bad light), say so. Examiners prefer an honest recovery to a frozen claim.
