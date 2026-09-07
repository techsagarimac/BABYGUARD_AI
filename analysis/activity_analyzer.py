"""
Estimated activity state from visual movement and pose.

This is a heuristic for a college demonstration. It is not a sleep
study, not a medical detector, and not a guarantee of wakefulness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import config
from analysis.movement_analyzer import MovementResult


@dataclass
class ActivityState:
    label: str
    display: str
    still_seconds: float
    body_position: str


DISPLAY = {
    "POSSIBLY SLEEPING": "😴 Possibly Sleeping",
    "POSSIBLY AWAKE": "👀 Possibly Awake",
    "ACTIVE": "🤸 Active",
}


class ActivityAnalyzer:
    """Map LOW / NORMAL / HIGH movement onto an estimated state."""

    def __init__(self):
        self._low_since: float | None = None
        self._last_label = "POSSIBLY AWAKE"

    def reset(self) -> None:
        self._low_since = None
        self._last_label = "POSSIBLY AWAKE"

    def update(
        self,
        movement: MovementResult,
        body_position: str = "Unknown",
        present: bool = True,
    ) -> ActivityState:
        now = time.time()
        if not present:
            self._low_since = None
            self._last_label = "POSSIBLY AWAKE"
            return ActivityState("POSSIBLY AWAKE", DISPLAY["POSSIBLY AWAKE"], 0.0, "Unknown")
        level = movement.level

        if level == "HIGH":
            self._low_since = None
            self._last_label = "ACTIVE"
            return ActivityState("ACTIVE", DISPLAY["ACTIVE"], 0.0, body_position)

        if level == "LOW":
            if self._low_since is None:
                self._low_since = now
            still = now - self._low_since
            needed = config.SLEEP_STILL_SECONDS
            if body_position == "Horizontal":
                needed = config.HORIZONTAL_SLEEP_SECONDS
            if still >= needed:
                self._last_label = "POSSIBLY SLEEPING"
                return ActivityState(
                    "POSSIBLY SLEEPING",
                    DISPLAY["POSSIBLY SLEEPING"],
                    still,
                    body_position,
                )
            self._last_label = "POSSIBLY AWAKE"
            return ActivityState("POSSIBLY AWAKE", DISPLAY["POSSIBLY AWAKE"], still, body_position)

        self._low_since = None
        self._last_label = "POSSIBLY AWAKE"
        return ActivityState("POSSIBLY AWAKE", DISPLAY["POSSIBLY AWAKE"], 0.0, body_position)
