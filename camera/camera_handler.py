"""
Webcam and demo-video capture for BabyGuard AI.

The rest of the app never talks to OpenCV's VideoCapture directly.
Camera errors (missing webcam, busy device, empty frames) stay here.
A video file can be looped for college demonstrations when no infant
is available.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import config


def find_demo_videos(folder: Optional[Path] = None) -> list[Path]:
    """Return video files in data/ that can be used for a presentation demo."""
    root = Path(folder or config.DATA_DIR)
    if not root.exists():
        return []
    clips = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in config.DEMO_VIDEO_EXTENSIONS:
            clips.append(path)
    preferred = str(getattr(config, "DEMO_VIDEO_PATH", "") or "").strip()
    if preferred:
        named = Path(preferred)
        if not named.is_absolute():
            named = config.PROJECT_ROOT / named
        if named.exists():
            return [named]
    return clips


class CameraHandler:
    """Open, read, resize, and close a webcam or a looping demo video."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture = None
        self.error_message = ""
        self.source = "live"
        self.video_path: Optional[Path] = None
        self.loop_video = True
        self.video_fps = float(config.TARGET_FPS)
        self._fail_count = 0

    @property
    def is_open(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    @property
    def is_video(self) -> bool:
        return self.source == "video"

    @property
    def source_label(self) -> str:
        if self.source == "video" and self.video_path is not None:
            return f"Demo video · {self.video_path.name}"
        return "Live camera"

    def start(self, camera_index: Optional[int] = None) -> bool:
        """Try to open the webcam. Returns False instead of raising."""
        if cv2 is None:
            self.error_message = "OpenCV is not installed. Run: pip install -r requirements.txt"
            return False

        self.stop()
        self.source = "live"
        self.video_path = None
        if camera_index is not None:
            self.camera_index = camera_index

        backends = [None]
        if sys.platform == "darwin":
            backends.append(getattr(cv2, "CAP_AVFOUNDATION", None))
        elif sys.platform.startswith("win"):
            backends.append(getattr(cv2, "CAP_DSHOW", None))

        indices = [self.camera_index]
        for extra in (0, 1):
            if extra not in indices:
                indices.append(extra)

        for index in indices:
            for backend in backends:
                try:
                    if backend is None:
                        cap = cv2.VideoCapture(index)
                    else:
                        cap = cv2.VideoCapture(index, backend)
                except Exception:
                    continue

                if not cap or not cap.isOpened():
                    if cap:
                        cap.release()
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ok, frame = cap.read()
                if ok and frame is not None:
                    self.capture = cap
                    self.camera_index = index
                    self.error_message = ""
                    self._fail_count = 0
                    self.video_fps = float(config.TARGET_FPS)
                    return True
                cap.release()

        self.capture = None
        self.error_message = "Camera unavailable"
        return False

    def start_video(self, video_path, loop: bool = True) -> bool:
        """Open a local video file and play it as the monitoring source."""
        if cv2 is None:
            self.error_message = "OpenCV is not installed. Run: pip install -r requirements.txt"
            return False

        path = Path(video_path).expanduser()
        if not path.is_absolute():
            path = (config.PROJECT_ROOT / path).resolve()
        if not path.exists() or not path.is_file():
            self.error_message = f"Demo video not found: {path}"
            return False

        self.stop()
        try:
            cap = cv2.VideoCapture(str(path))
        except Exception as exc:
            self.error_message = f"Could not open demo video: {exc}"
            return False

        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            self.error_message = f"Could not open demo video: {path.name}"
            return False

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            self.error_message = f"Demo video has no readable frames: {path.name}"
            return False

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.capture = cap
        self.source = "video"
        self.video_path = path
        self.loop_video = bool(loop)
        self.video_fps = fps if fps > 1.0 else 24.0
        self.error_message = ""
        self._fail_count = 0
        return True

    def read(self) -> Optional[np.ndarray]:
        """Return one BGR frame, or None if the source failed."""
        if not self.is_open:
            return None

        try:
            ok, frame = self.capture.read()
        except Exception:
            ok, frame = False, None

        if (not ok or frame is None or getattr(frame, "size", 0) == 0) and self.source == "video" and self.loop_video:
            try:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.capture.read()
            except Exception:
                ok, frame = False, None

        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            self._fail_count += 1
            if self._fail_count >= 15:
                self.error_message = (
                    "Demo video ended" if self.source == "video" else "Camera unavailable"
                )
                self.stop()
            return None

        self._fail_count = 0
        return self._fit(frame)

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        """Letterbox the frame into the configured preview size."""
        if cv2 is None or frame is None:
            return frame
        height, width = frame.shape[:2]
        if width == self.width and height == self.height:
            return frame
        scale = min(self.width / max(width, 1), self.height / max(height, 1))
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (18, 14, 12)
        x = (self.width - new_w) // 2
        y = (self.height - new_h) // 2
        canvas[y : y + new_h, x : x + new_w] = resized
        return canvas

    def resize(self, frame: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        if cv2 is None or frame is None:
            return frame
        target_w = width or self.width
        target_h = height or self.height
        h, w = frame.shape[:2]
        if w == target_w and h == target_h:
            return frame
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def stop(self) -> None:
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None

    def placeholder_frame(self, message: str = "Camera unavailable") -> np.ndarray:
        """Dark frame used when the webcam is off or missing."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (28, 22, 18)
        if cv2 is None:
            return frame
        lines = message.split("\n")
        y = self.height // 2 - 12 * (len(lines) - 1)
        for line in lines:
            size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
            x = max(16, (self.width - size[0]) // 2)
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (210, 200, 190), 2, cv2.LINE_AA)
            y += 36
        return frame
