"""
Movement analysis from pose landmarks or bounding-box centres.

A single noisy frame is not enough to change the activity label.
A short rolling history is averaged first.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import config
from utils.helpers import landmark_displacement


@dataclass
class MovementResult:
    level: str  # LOW / NORMAL / HIGH
    score: float
    source: str  # pose / bbox / none
    history_len: int


class MovementAnalyzer:
    """Track subject motion across a small window of frames."""

    def __init__(self, history: int = config.MOVEMENT_HISTORY):
        self.history = max(3, history)
        self._pose_points: deque[list[tuple[float, float]]] = deque(maxlen=self.history)
        self._centers: deque[tuple[float, float]] = deque(maxlen=self.history)
        self._scores: deque[float] = deque(maxlen=self.history)

    def reset(self) -> None:
        self._pose_points.clear()
        self._centers.clear()
        self._scores.clear()

    def update(
        self,
        key_points: Optional[list[tuple[float, float]]] = None,
        center_norm: Optional[tuple[float, float]] = None,
        detected: bool = False,
    ) -> MovementResult:
        if not detected:
            # Do not keep stale motion if the subject left the frame.
            if len(self._scores) > 0:
                self._scores.append(0.0)
            return MovementResult(
                level="LOW" if len(self._scores) else "LOW",
                score=self._mean_score(),
                source="none",
                history_len=len(self._scores),
            )

        score = 0.0
        source = "none"

        if key_points and len(key_points) >= 4:
            if self._pose_points:
                prev = self._pose_points[-1]
                if len(prev) == len(key_points):
                    score = landmark_displacement(prev, key_points)
                    source = "pose"
            self._pose_points.append(list(key_points))
        elif center_norm is not None:
            self._pose_points.clear()

        if center_norm is not None:
            if self._centers:
                px, py = self._centers[-1]
                cx, cy = center_norm
                bbox_score = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if source != "pose":
                    score = bbox_score
                    source = "bbox"
                else:
                    # Blend a little box motion so a sliding subject is noticed
                    # even if pose landmarks jitter in place.
                    score = max(score, bbox_score * 0.85)
            self._centers.append(center_norm)

        self._scores.append(score)
        mean_score = self._mean_score()
        return MovementResult(
            level=self._classify(mean_score, source),
            score=mean_score,
            source=source if source != "none" else ("bbox" if center_norm else "none"),
            history_len=len(self._scores),
        )

    def last_center(self) -> Optional[tuple[float, float]]:
        if not self._centers:
            return None
        return self._centers[-1]

    def _mean_score(self) -> float:
        if not self._scores:
            return 0.0
        return sum(self._scores) / len(self._scores)

    @staticmethod
    def _classify(score: float, source: str) -> str:
        low = config.MOVEMENT_LOW_THRESHOLD if source == "pose" else config.BBOX_MOVEMENT_LOW
        high = config.MOVEMENT_HIGH_THRESHOLD if source == "pose" else config.BBOX_MOVEMENT_HIGH
        if source == "none":
            return "LOW"
        if score <= low:
            return "LOW"
        if score >= high:
            return "HIGH"
        return "NORMAL"
