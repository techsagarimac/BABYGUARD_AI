"""
BabyGuard AI — configuration.

All thresholds live here so a student can change behaviour without
editing the detection or UI code. Values are chosen for a live college
demo on a normal laptop, not for clinical use.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Optional demo clip for presentations (no infant required).
# Leave empty to pick a file with the DEMO VIDEO button, or put an .mp4 in data/.
DEMO_VIDEO_PATH = ""
DEMO_VIDEO_LOOP = True
DEMO_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")

# Processing size sent to YOLO. Smaller = faster, slightly less accurate.
YOLO_IMAGE_SIZE = 320

# Target UI refresh. The processing thread may run a little slower.
TARGET_FPS = 18
UI_REFRESH_MS = 40

# Run the heavier detectors every N captured frames (1 = every frame).
PROCESS_EVERY_N_FRAMES = 1

# ---------------------------------------------------------------------------
# YOLO person detection
# ---------------------------------------------------------------------------
# Preferred local filenames. Ultralytics will download a nano model into
# models/ if none of these files exist and the network is available.
YOLO_WEIGHT_CANDIDATES = ("yolo11n.pt", "yolov8n.pt")
YOLO_CONFIDENCE = 0.40
YOLO_IOU = 0.45
# COCO class 0 = person. This is generic person detection, not baby ID.
YOLO_PERSON_CLASS_ID = 0
YOLO_ALLOW_DOWNLOAD = True

# ---------------------------------------------------------------------------
# MediaPipe Pose
# ---------------------------------------------------------------------------
# auto = safest order for this computer (on macOS skip the crashing
# mp.solutions.pose graph and prefer the Tasks API, then a box fallback).
# Other values: "tasks", "solutions", "box", "off"
POSE_BACKEND = "auto"
POSE_SAFE_PROBE = True
POSE_ALLOW_BOX_FALLBACK = True
POSE_DETECTION_CONFIDENCE = 0.50
POSE_TRACKING_CONFIDENCE = 0.50
POSE_MODEL_COMPLEXITY = 0  # 0 = lite (better for laptops)
POSE_TASK_MODEL = MODELS_DIR / "pose_landmarker_lite.task"
POSE_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# Landmarks used for movement (MediaPipe Pose 33-point set).
KEY_LANDMARK_INDICES = (
    0,  # nose
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
)

# ---------------------------------------------------------------------------
# Movement analysis (normalized 0–1 image coordinates)
# ---------------------------------------------------------------------------
MOVEMENT_HISTORY = 10
MOVEMENT_LOW_THRESHOLD = 0.012
MOVEMENT_HIGH_THRESHOLD = 0.055
BBOX_MOVEMENT_LOW = 0.010
BBOX_MOVEMENT_HIGH = 0.045
MIN_LANDMARK_VISIBILITY = 0.40

# ---------------------------------------------------------------------------
# Estimated activity state (visual heuristic only)
# ---------------------------------------------------------------------------
SLEEP_STILL_SECONDS = 7.0
HORIZONTAL_SLEEP_SECONDS = 4.5
HORIZONTAL_TORSO_RATIO = 1.15

# ---------------------------------------------------------------------------
# Unusual activity / monitoring logic
# ---------------------------------------------------------------------------
HIGH_MOVEMENT_PERSIST_FRAMES = 10
SUDDEN_JUMP_THRESHOLD = 0.18
DISAPPEAR_GRACE_FRAMES = 12
MIN_SEEN_SECONDS_BEFORE_MISSING = 2.0

# ---------------------------------------------------------------------------
# Monitoring zone (normalized x1, y1, x2, y2)
# Set MONITORING_ZONE_ENABLED = True to start with a default rectangle.
# The dashboard "SET AREA" button overrides these values at runtime.
# ---------------------------------------------------------------------------
MONITORING_ZONE_ENABLED = False
MONITORING_ZONE = (0.12, 0.08, 0.88, 0.92)
ZONE_OUTSIDE_PERSIST_FRAMES = 8
# A subject is "inside" if the bbox center is inside the rectangle.
# Set True to instead require most of the box to stay inside.
ZONE_REQUIRE_MOSTLY_INSIDE = False
ZONE_INSIDE_FRACTION = 0.55

# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
ALERT_COOLDOWN_SECONDS = 6.0
VOICE_ALERTS_ENABLED = False
TTS_RATE = 170
BEEP_ENABLED = True

ALERT_TTS = {
    "unusual": "Possible unusual activity detected.",
    "outside": "Subject moved outside the monitoring area.",
    "missing": "Subject is no longer visible in the camera view.",
}

# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------
EVENT_LOG_LIMIT = 80
EVENT_LOG_DISPLAY = 16

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
WINDOW_TITLE = "BabyGuard AI — Intelligent Camera Monitoring"
WINDOW_SIZE = "1280x820"
PREVIEW_MAX_WIDTH = 760
PREVIEW_MAX_HEIGHT = 520
