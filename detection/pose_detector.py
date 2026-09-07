"""
MediaPipe Pose wrapper.

Supports:
- MediaPipe Tasks PoseLandmarker (preferred on macOS)
- MediaPipe 0.10.x mp.solutions.pose (probed in a child process)
- Bounding-box approximate pose if MediaPipe cannot start

On some Apple Silicon + macOS 15 setups, mp.solutions.pose hits a
TensorFlow Lite C++ abort ("Feedback manager requires a model with a
single signature" / "Service is unavailable"). That abort cannot be
caught in Python, so we probe Pose in a subprocess first.
"""

from __future__ import annotations

import os
import ssl
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty
from typing import Optional, Protocol
from urllib.request import urlopen, urlretrieve

import multiprocessing

import numpy as np

from utils.runtime import apply_safe_runtime

apply_safe_runtime()

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

import config

# Torso + limbs. Face mesh points are skipped so the overlay stays readable.
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]

KEY_NAMES = {
    0: "nose",
    11: "l_shoulder",
    12: "r_shoulder",
    13: "l_elbow",
    14: "r_elbow",
    15: "l_wrist",
    16: "r_wrist",
    23: "l_hip",
    24: "r_hip",
    25: "l_knee",
    26: "r_knee",
    27: "l_ankle",
    28: "r_ankle",
}


@dataclass
class LandmarkPoint:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class PoseResult:
    detected: bool
    landmarks: list = field(default_factory=list)
    key_points: list = field(default_factory=list)
    body_position: str = "Unknown"
    status_text: str = "Pose not detected"
    error_message: str = ""
    backend: str = ""


class PoseBackend(Protocol):
    name: str

    def process(self, bgr_frame: np.ndarray) -> PoseResult: ...

    def close(self) -> None: ...


def _download_task_model(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(config.POSE_TASK_URL, destination)
        if destination.stat().st_size > 1000:
            return
    except Exception:
        pass

    context = ssl._create_unverified_context()
    with urlopen(config.POSE_TASK_URL, context=context, timeout=90) as source:
        destination.write_bytes(source.read())
    if destination.stat().st_size < 1000:
        raise RuntimeError("Downloaded pose_landmarker_lite.task looks empty")


def ensure_task_model() -> Path:
    path = config.POSE_TASK_MODEL
    if not path.exists() or path.stat().st_size < 1000:
        _download_task_model(path)
    return path


def estimate_body_position(landmarks: list[LandmarkPoint]) -> str:
    """Simple visual pose: horizontal vs upright. Not a medical posture test."""
    if len(landmarks) < 29:
        return "Unknown"
    left_s, right_s = landmarks[11], landmarks[12]
    left_h, right_h = landmarks[23], landmarks[24]
    if min(left_s.visibility, right_s.visibility, left_h.visibility, right_h.visibility) < 0.3:
        return "Unknown"
    torso_w = abs(left_s.x - right_s.x)
    shoulder_y = (left_s.y + right_s.y) / 2.0
    hip_y = (left_h.y + right_h.y) / 2.0
    torso_h = abs(hip_y - shoulder_y)
    if torso_h < 1e-4:
        return "Unknown"
    if torso_w > torso_h * config.HORIZONTAL_TORSO_RATIO:
        return "Horizontal"
    return "Upright"


def extract_key_points(landmarks: list[LandmarkPoint]) -> list[tuple[float, float]]:
    points = []
    for index in config.KEY_LANDMARK_INDICES:
        if index >= len(landmarks):
            continue
        lm = landmarks[index]
        if lm.visibility >= config.MIN_LANDMARK_VISIBILITY:
            points.append((lm.x, lm.y))
    return points


def result_from_landmarks(landmarks: list[LandmarkPoint], backend: str, status: str) -> PoseResult:
    return PoseResult(
        detected=True,
        landmarks=landmarks,
        key_points=extract_key_points(landmarks),
        body_position=estimate_body_position(landmarks),
        status_text=status,
        backend=backend,
    )


def landmarks_from_bbox(bbox_norm: tuple[float, float, float, float]) -> list[LandmarkPoint]:
    """
    Approximate 33 pose points from a person box.

    Used only when MediaPipe cannot start. Honest label: approximate.
    """
    x1, y1, x2, y2 = bbox_norm
    cx = (x1 + x2) / 2.0
    width = max(x2 - x1, 1e-4)
    height = max(y2 - y1, 1e-4)
    points = [LandmarkPoint(cx, y1 + 0.5 * height, 0.0, 0.15) for _ in range(33)]

    def put(index: int, x: float, y: float, vis: float = 0.85) -> None:
        points[index] = LandmarkPoint(x, y, 0.0, vis)

    put(0, cx, y1 + 0.12 * height)
    put(11, cx - 0.22 * width, y1 + 0.28 * height)
    put(12, cx + 0.22 * width, y1 + 0.28 * height)
    put(13, cx - 0.30 * width, y1 + 0.48 * height)
    put(14, cx + 0.30 * width, y1 + 0.48 * height)
    put(15, cx - 0.26 * width, y1 + 0.66 * height)
    put(16, cx + 0.26 * width, y1 + 0.66 * height)
    put(23, cx - 0.16 * width, y1 + 0.58 * height)
    put(24, cx + 0.16 * width, y1 + 0.58 * height)
    put(25, cx - 0.16 * width, y1 + 0.78 * height)
    put(26, cx + 0.16 * width, y1 + 0.78 * height)
    put(27, cx - 0.14 * width, y1 + 0.95 * height)
    put(28, cx + 0.14 * width, y1 + 0.95 * height)
    return points


class SolutionsPoseBackend:
    name = "solutions"

    def __init__(self):
        if mp is None or not hasattr(mp, "solutions") or not hasattr(mp.solutions, "pose"):
            raise RuntimeError("MediaPipe solutions.pose is not available")
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=config.POSE_MODEL_COMPLEXITY,
            smooth_landmarks=False,
            enable_segmentation=False,
            min_detection_confidence=config.POSE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_TRACKING_CONFIDENCE,
        )

    def process(self, bgr_frame: np.ndarray) -> PoseResult:
        if cv2 is None:
            return PoseResult(detected=False, status_text="Pose not detected", backend=self.name)
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True
        if not results.pose_landmarks:
            return PoseResult(detected=False, status_text="Pose not detected", backend=self.name)
        landmarks = [
            LandmarkPoint(lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0))
            for lm in results.pose_landmarks.landmark
        ]
        return result_from_landmarks(landmarks, self.name, "Pose detected")

    def close(self) -> None:
        try:
            self._pose.close()
        except Exception:
            pass


class TasksPoseBackend:
    name = "tasks"

    def __init__(self):
        if mp is None or not hasattr(mp, "tasks"):
            raise RuntimeError("MediaPipe tasks API is not available")
        path = ensure_task_model()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=config.POSE_TRACKING_CONFIDENCE,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def process(self, bgr_frame: np.ndarray) -> PoseResult:
        if cv2 is None:
            return PoseResult(detected=False, status_text="Pose not detected", backend=self.name)
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(image)
        if not results.pose_landmarks:
            return PoseResult(detected=False, status_text="Pose not detected", backend=self.name)
        raw = results.pose_landmarks[0]
        landmarks = [
            LandmarkPoint(lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0))
            for lm in raw
        ]
        return result_from_landmarks(landmarks, self.name, "Pose detected")

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


class BoxPoseBackend:
    """Last-resort stick figure from the YOLO box. Not MediaPipe."""

    name = "box"

    def process(self, bgr_frame: np.ndarray) -> PoseResult:
        return PoseResult(
            detected=False,
            status_text="Pose not detected",
            backend=self.name,
        )

    def close(self) -> None:
        return None


def create_pose_backend(name: str) -> PoseBackend:
    name = (name or "").strip().lower()
    if name == "solutions":
        return SolutionsPoseBackend()
    if name == "tasks":
        return TasksPoseBackend()
    if name == "box":
        return BoxPoseBackend()
    raise ValueError(f"Unknown pose backend: {name}")


def _in_probe_process() -> bool:
    return os.environ.get("BABYGUARD_POSE_PROBE") == "1"


def _should_probe() -> bool:
    if not getattr(config, "POSE_SAFE_PROBE", True):
        return False
    if os.environ.get("BABYGUARD_SKIP_POSE_PROBE") == "1":
        return False
    if _in_probe_process():
        return False
    return True


def probe_pose_backend(backend: str, timeout: float = 40.0) -> bool:
    """Return True if a child process can start this MediaPipe backend."""
    if not _should_probe():
        return True
    env = os.environ.copy()
    env["BABYGUARD_POSE_PROBE"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["GLOG_minloglevel"] = "2"
    env["MEDIAPIPE_DISABLE_GPU"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "detection.pose_probe", backend],
            cwd=str(config.PROJECT_ROOT),
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return completed.returncode == 0 and "POSE_PROBE_OK" in (completed.stdout or "")


def _use_isolated_process() -> bool:
    """Keep MediaPipe out of the YOLO/Tk process on macOS."""
    if os.environ.get("BABYGUARD_FORCE_MEDIAPIPE") == "1":
        return False
    return sys.platform == "darwin"


class IsolatedPoseClient:
    """Talk to detection.pose_worker over spawn queues."""

    def __init__(self, backend: str):
        self.name = backend
        ctx = multiprocessing.get_context("spawn")
        self._in = ctx.Queue(maxsize=2)
        self._out = ctx.Queue(maxsize=2)
        self._proc = ctx.Process(
            target=_isolated_entry,
            args=(backend, self._in, self._out),
            name="babyguard-pose",
            daemon=True,
        )
        self._proc.start()
        try:
            kind, payload = self._out.get(timeout=45)
        except Empty as exc:
            self.close()
            raise RuntimeError("Pose worker did not start in time") from exc
        if kind != "ready":
            self.close()
            raise RuntimeError(payload or "Pose worker failed to start")

    def process(self, bgr_frame: np.ndarray) -> PoseResult:
        if self._proc is None or not self._proc.is_alive():
            raise RuntimeError("Pose worker stopped")
        try:
            self._in.put_nowait(bgr_frame)
        except Exception:
            try:
                self._in.get_nowait()
            except Exception:
                pass
            self._in.put_nowait(bgr_frame)
        try:
            kind, payload = self._out.get(timeout=2.5)
        except Empty as exc:
            raise RuntimeError("Pose worker timed out") from exc
        if kind == "ok":
            return payload
        raise RuntimeError(str(payload))

    def close(self) -> None:
        try:
            self._in.put_nowait(None)
        except Exception:
            pass
        if self._proc is not None:
            self._proc.join(timeout=1.5)
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc = None


def _isolated_entry(backend: str, in_queue, out_queue) -> None:
    from detection.pose_worker import run_worker

    run_worker(backend, in_queue, out_queue)


def choose_pose_backends() -> list[str]:
    requested = str(getattr(config, "POSE_BACKEND", "auto")).strip().lower()
    if requested in {"tasks", "solutions", "box"}:
        return [requested]
    if requested in {"off", "none", "disabled"}:
        return ["box"] if config.POSE_ALLOW_BOX_FALLBACK else []

    # mp.solutions.pose is the graph that SIGABRTs on macOS 15 + Apple Silicon.
    if sys.platform == "darwin":
        order = ["tasks"]
    else:
        order = ["tasks", "solutions"]
    if config.POSE_ALLOW_BOX_FALLBACK:
        order.append("box")
    return order


class PoseDetector:
    """Detect one body pose and draw a simplified skeleton."""

    def __init__(self):
        self.available = False
        self.error_message = ""
        self.backend = ""
        self._handle: Optional[PoseBackend] = None

        if mp is None and "box" not in choose_pose_backends():
            self.error_message = "MediaPipe is not installed. Run: pip install -r requirements.txt"
            return

        last_error = ""
        for name in choose_pose_backends():
            if name != "box" and mp is None:
                last_error = "MediaPipe is not installed"
                continue
            if name == "tasks":
                try:
                    ensure_task_model()
                except Exception as exc:
                    last_error = f"Could not download pose_landmarker_lite.task: {exc}"
                    continue
            # Isolated workers already protect the GUI, so skip the extra probe.
            if name in {"tasks", "solutions"} and _should_probe() and not _use_isolated_process():
                if not probe_pose_backend(name):
                    last_error = (
                        f"MediaPipe {name} aborted in a safety probe "
                        "(known TensorFlow Lite crash on some Macs)."
                    )
                    continue
            handle = None
            try:
                if name != "box" and _use_isolated_process():
                    handle = IsolatedPoseClient(name)
                else:
                    handle = create_pose_backend(name)
                    if name != "box":
                        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
                        handle.process(dummy)
                self._handle = handle
                self.backend = name
                self.available = True
                self.error_message = (
                    "Using approximate pose from the person box because MediaPipe Pose "
                    "is unstable on this computer."
                    if name == "box"
                    else ""
                )
                return
            except Exception as exc:
                last_error = str(exc)
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass

        self.error_message = last_error or "Pose detector is not ready"

    def detect(
        self,
        frame: np.ndarray,
        bbox_norm: Optional[tuple[float, float, float, float]] = None,
    ) -> PoseResult:
        if self.backend == "box":
            if bbox_norm is None:
                return PoseResult(
                    detected=False,
                    status_text="Pose not detected",
                    error_message=self.error_message,
                    backend="box",
                )
            landmarks = landmarks_from_bbox(bbox_norm)
            return result_from_landmarks(
                landmarks,
                "box",
                "Approximate pose (MediaPipe unavailable)",
            )

        if not self.available or self._handle is None or cv2 is None:
            return PoseResult(
                detected=False,
                status_text="Pose not detected",
                error_message=self.error_message or "Pose detector is not ready",
                backend=self.backend,
            )
        if frame is None or getattr(frame, "size", 0) == 0:
            return PoseResult(
                detected=False,
                status_text="Pose not detected",
                error_message="Invalid camera frame",
                backend=self.backend,
            )

        try:
            return self._handle.process(frame)
        except Exception as exc:
            if bbox_norm is not None and config.POSE_ALLOW_BOX_FALLBACK:
                landmarks = landmarks_from_bbox(bbox_norm)
                return result_from_landmarks(
                    landmarks,
                    "box",
                    "Approximate pose (MediaPipe worker unavailable)",
                )
            return PoseResult(
                detected=False,
                status_text="Pose not detected",
                error_message=str(exc),
                backend=self.backend,
            )

    def draw(self, frame: np.ndarray, result: PoseResult) -> np.ndarray:
        if cv2 is None or frame is None or not result.detected or not result.landmarks:
            return frame
        height, width = frame.shape[:2]
        points = result.landmarks

        def xy(index: int) -> Optional[tuple[int, int]]:
            if index >= len(points):
                return None
            lm = points[index]
            if lm.visibility < 0.25:
                return None
            return int(lm.x * width), int(lm.y * height)

        colour = (180, 180, 180) if result.backend == "box" else (255, 180, 80)
        for start, end in POSE_CONNECTIONS:
            a, b = xy(start), xy(end)
            if a and b:
                cv2.line(frame, a, b, colour, 2, cv2.LINE_AA)

        for index in config.KEY_LANDMARK_INDICES:
            point = xy(index)
            if point:
                dot = (200, 200, 200) if result.backend == "box" else (
                    (80, 220, 255) if index == 0 else (80, 255, 190)
                )
                cv2.circle(frame, point, 5, dot, -1, cv2.LINE_AA)
        return frame

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None
