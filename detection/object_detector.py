"""
YOLO person detector.

Uses a lightweight Ultralytics nano model (YOLO11n or YOLOv8n) to find
COCO class 0 ("person"). That is generic person detection — it does not
prove the subject is a baby.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import config
from utils.helpers import bbox_center, draw_label


@dataclass
class PersonDetection:
    detected: bool
    bbox_px: Optional[tuple[int, int, int, int]]
    bbox_norm: Optional[tuple[float, float, float, float]]
    center_norm: Optional[tuple[float, float]]
    confidence: float
    status_text: str
    error_message: str = ""
    model_name: str = ""


def find_yolo_weights(models_dir: Path) -> Optional[Path]:
    """Return a local nano-weight file if one is already downloaded."""
    for name in config.YOLO_WEIGHT_CANDIDATES:
        path = Path(models_dir) / name
        try:
            if path.exists() and path.stat().st_size > 1000:
                return path
        except OSError:
            continue
    return None


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class ObjectDetector:
    """Detect the most confident person in a BGR frame."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir or config.MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.available = False
        self.ready = False
        self.error_message = ""
        self.model_name = ""
        self.device = "cpu"
        self._model = None
        self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.error_message = (
                "YOLO is unavailable. Install dependencies with: "
                f"pip install -r requirements.txt  ({exc})"
            )
            return

        weights = find_yolo_weights(self.models_dir)
        self.device = _pick_device()

        if weights is None and not config.YOLO_ALLOW_DOWNLOAD:
            self.error_message = (
                f"No YOLO model found in {self.models_dir}. "
                "Place yolo11n.pt or yolov8n.pt in the models folder, "
                "or set YOLO_ALLOW_DOWNLOAD = True in config.py."
            )
            return

        candidates = []
        if weights is not None:
            candidates.append(str(weights))
        else:
            candidates.extend(config.YOLO_WEIGHT_CANDIDATES)

        last_error = ""
        original_cwd = os.getcwd()
        try:
            os.chdir(self.models_dir)
            for candidate in candidates:
                try:
                    model = YOLO(candidate)
                    # One tiny warmup so the first live frame is not extra-slow.
                    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
                    model.predict(
                        dummy,
                        imgsz=160,
                        verbose=False,
                        device=self.device,
                        classes=[config.YOLO_PERSON_CLASS_ID],
                    )
                    self._model = model
                    self.model_name = Path(candidate).name
                    self.available = True
                    self.ready = True
                    self.error_message = ""
                    return
                except Exception as exc:
                    last_error = str(exc)
                    if self.device != "cpu":
                        try:
                            model = YOLO(candidate)
                            dummy = np.zeros((160, 160, 3), dtype=np.uint8)
                            model.predict(
                                dummy,
                                imgsz=160,
                                verbose=False,
                                device="cpu",
                                classes=[config.YOLO_PERSON_CLASS_ID],
                            )
                            self._model = model
                            self.device = "cpu"
                            self.model_name = Path(candidate).name
                            self.available = True
                            self.ready = True
                            self.error_message = ""
                            return
                        except Exception as cpu_exc:
                            last_error = str(cpu_exc)
        finally:
            os.chdir(original_cwd)

        self.error_message = (
            "Could not load a YOLO nano model. "
            f"Last error: {last_error or 'unknown'}. "
            f"Download yolo11n.pt into {self.models_dir} and try again."
        )

    def detect(self, frame: np.ndarray) -> PersonDetection:
        if not self.ready or self._model is None:
            return PersonDetection(
                detected=False,
                bbox_px=None,
                bbox_norm=None,
                center_norm=None,
                confidence=0.0,
                status_text="Detector unavailable",
                error_message=self.error_message or "YOLO model is not ready",
                model_name=self.model_name,
            )

        if frame is None or getattr(frame, "size", 0) == 0:
            return PersonDetection(
                detected=False,
                bbox_px=None,
                bbox_norm=None,
                center_norm=None,
                confidence=0.0,
                status_text="No person detected",
                error_message="Invalid camera frame",
                model_name=self.model_name,
            )

        try:
            results = self._model.predict(
                frame,
                imgsz=config.YOLO_IMAGE_SIZE,
                conf=config.YOLO_CONFIDENCE,
                iou=config.YOLO_IOU,
                verbose=False,
                device=self.device,
                classes=[config.YOLO_PERSON_CLASS_ID],
            )
        except Exception as exc:
            return PersonDetection(
                detected=False,
                bbox_px=None,
                bbox_norm=None,
                center_norm=None,
                confidence=0.0,
                status_text="Detector error",
                error_message=str(exc),
                model_name=self.model_name,
            )

        height, width = frame.shape[:2]
        best = None
        best_score = -1.0

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = (int(v) for v in xyxy)
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    score = conf * (1.0 + area / float(max(width * height, 1)))
                    if score > best_score:
                        best_score = score
                        best = (x1, y1, x2, y2, conf)
                except Exception:
                    continue

        if best is None:
            return PersonDetection(
                detected=False,
                bbox_px=None,
                bbox_norm=None,
                center_norm=None,
                confidence=0.0,
                status_text="No person detected",
                model_name=self.model_name,
            )

        x1, y1, x2, y2, conf = best
        x1 = int(max(0, min(x1, width - 1)))
        y1 = int(max(0, min(y1, height - 1)))
        x2 = int(max(0, min(x2, width)))
        y2 = int(max(0, min(y2, height)))
        bbox_norm = (x1 / width, y1 / height, x2 / width, y2 / height)
        return PersonDetection(
            detected=True,
            bbox_px=(x1, y1, x2, y2),
            bbox_norm=bbox_norm,
            center_norm=bbox_center(bbox_norm),
            confidence=conf,
            status_text="Baby/Person detected",
            model_name=self.model_name,
        )

    def draw(self, frame: np.ndarray, detection: PersonDetection) -> np.ndarray:
        if cv2 is None or frame is None or not detection.detected or detection.bbox_px is None:
            return frame
        x1, y1, x2, y2 = detection.bbox_px
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 140), 2)
        label = f"Subject {detection.confidence:.2f}"
        draw_label(frame, label, (x1, max(24, y1 - 8)), (80, 220, 140))
        return frame
