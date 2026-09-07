# BabyGuard AI models

Place a lightweight YOLO nano weight file here:

- `yolo11n.pt` (preferred)
- `yolov8n.pt` (supported fallback)

The first time you run `python main.py` with an internet connection,
Ultralytics can download `yolo11n.pt` into this folder automatically.

If download is blocked, copy the `.pt` file here yourself.

This project uses generic COCO **person** detection.
It does **not** identify a baby as a distinct class.

MediaPipe Pose 1.x may also download:

- `pose_landmarker_lite.task`

Do not commit large model files to git.
