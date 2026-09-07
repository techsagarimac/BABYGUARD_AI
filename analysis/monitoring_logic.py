"""
Monitoring rules: unusual motion, zone exit, and disappearance.

Alerts are not fired from a single noisy frame. Each condition must
persist for a configurable number of frames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import config
from analysis.activity_analyzer import ActivityState
from analysis.movement_analyzer import MovementResult
from detection.object_detector import PersonDetection
from detection.pose_detector import PoseResult
from utils.helpers import bbox_center, bbox_intersection_fraction, point_in_rect


@dataclass
class MonitorDecision:
    subject_present: bool
    pose_detected: bool
    movement_level: str
    movement_score: float
    estimated_state: str
    estimated_state_display: str
    body_position: str
    zone_enabled: bool
    zone_status: str  # inside / outside / disabled / unknown
    unusual: bool
    disappeared: bool
    alert_kind: str  # normal / unusual / outside / missing
    alert_message: str
    events: list[str] = field(default_factory=list)


class MonitoringLogic:
    """Combine detector outputs into a single monitoring decision."""

    def __init__(self):
        self._high_streak = 0
        self._outside_streak = 0
        self._missing_streak = 0
        self._seen_since: Optional[float] = None
        self._was_present = False
        self._last_center: Optional[tuple[float, float]] = None
        self._sudden = False

    def reset(self) -> None:
        self._high_streak = 0
        self._outside_streak = 0
        self._missing_streak = 0
        self._seen_since = None
        self._was_present = False
        self._last_center = None
        self._sudden = False

    def evaluate(
        self,
        person: PersonDetection,
        pose: PoseResult,
        movement: MovementResult,
        activity: ActivityState,
        zone: Optional[tuple[float, float, float, float]],
        zone_enabled: bool,
    ) -> MonitorDecision:
        events: list[str] = []
        present = bool(person.detected)
        now = time.time()

        if present:
            if not self._was_present:
                events.append("Subject detected")
            self._was_present = True
            if self._seen_since is None:
                self._seen_since = now
            self._missing_streak = 0
        else:
            if self._was_present:
                self._missing_streak += 1
            self._high_streak = 0
            self._outside_streak = 0
            self._sudden = False

        # Sudden large change in position (bbox centre jump).
        center = person.center_norm
        if present and center is not None and self._last_center is not None:
            dx = center[0] - self._last_center[0]
            dy = center[1] - self._last_center[1]
            jump = (dx * dx + dy * dy) ** 0.5
            self._sudden = jump >= config.SUDDEN_JUMP_THRESHOLD
        else:
            self._sudden = False
        if present and center is not None:
            self._last_center = center

        if movement.level == "HIGH":
            self._high_streak += 1
        else:
            self._high_streak = 0

        unusual = False
        if present and self._high_streak >= config.HIGH_MOVEMENT_PERSIST_FRAMES:
            unusual = True
        if present and self._sudden and self._high_streak >= 2:
            unusual = True

        zone_status = "disabled"
        if zone_enabled and zone is not None:
            zone_status = "unknown"
            if present and person.bbox_norm is not None:
                inside = self._is_inside(person, zone)
                if inside:
                    self._outside_streak = 0
                    zone_status = "inside"
                else:
                    self._outside_streak += 1
                    zone_status = "outside" if self._outside_streak >= config.ZONE_OUTSIDE_PERSIST_FRAMES else "inside"
            elif not present:
                zone_status = "unknown"

        disappeared = False
        seen_long_enough = (
            self._seen_since is not None
            and (now - self._seen_since) >= config.MIN_SEEN_SECONDS_BEFORE_MISSING
        )
        if (
            not present
            and self._was_present
            and seen_long_enough
            and self._missing_streak >= config.DISAPPEAR_GRACE_FRAMES
        ):
            disappeared = True

        alert_kind = "normal"
        alert_message = "Monitoring normally"
        if zone_status == "outside":
            alert_kind = "outside"
            alert_message = "Subject outside monitoring area"
        elif disappeared:
            alert_kind = "missing"
            alert_message = "Subject no longer visible"
        elif unusual:
            alert_kind = "unusual"
            alert_message = "Possible unusual activity"

        return MonitorDecision(
            subject_present=present,
            pose_detected=bool(pose.detected),
            movement_level=movement.level,
            movement_score=movement.score,
            estimated_state=activity.label,
            estimated_state_display=activity.display,
            body_position=activity.body_position,
            zone_enabled=bool(zone_enabled and zone is not None),
            zone_status=zone_status,
            unusual=unusual,
            disappeared=disappeared,
            alert_kind=alert_kind,
            alert_message=alert_message,
            events=events,
        )

    @staticmethod
    def _is_inside(person: PersonDetection, zone: tuple[float, float, float, float]) -> bool:
        if config.ZONE_REQUIRE_MOSTLY_INSIDE and person.bbox_norm is not None:
            return bbox_intersection_fraction(person.bbox_norm, zone) >= config.ZONE_INSIDE_FRACTION
        if person.center_norm is not None:
            return point_in_rect(person.center_norm[0], person.center_norm[1], zone)
        if person.bbox_norm is not None:
            cx, cy = bbox_center(person.bbox_norm)
            return point_in_rect(cx, cy, zone)
        return True
