"""
BabyGuard AI unit tests.

These tests do not require a webcam or downloaded YOLO weights.
They check the analysis rules that a viva examiner will ask about.

Run from the project folder:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from alerts.alert_manager import AlertManager
from analysis.activity_analyzer import ActivityAnalyzer
from analysis.monitoring_logic import MonitoringLogic
from analysis.movement_analyzer import MovementAnalyzer
from camera.camera_handler import CameraHandler, find_demo_videos
from detection.object_detector import PersonDetection, find_yolo_weights
from detection.pose_detector import (
    LandmarkPoint,
    PoseResult,
    estimate_body_position,
    landmarks_from_bbox,
)
from utils.helpers import (
    bbox_center,
    bbox_intersection_fraction,
    landmark_displacement,
    point_in_rect,
    rect_from_points,
)


def _person(detected: bool, cx: float = 0.5, cy: float = 0.5, size: float = 0.2) -> PersonDetection:
    if not detected:
        return PersonDetection(
            detected=False,
            bbox_px=None,
            bbox_norm=None,
            center_norm=None,
            confidence=0.0,
            status_text="No person detected",
        )
    half = size / 2.0
    bbox = (cx - half, cy - half, cx + half, cy + half)
    return PersonDetection(
        detected=True,
        bbox_px=(0, 0, 10, 10),
        bbox_norm=bbox,
        center_norm=(cx, cy),
        confidence=0.9,
        status_text="Baby/Person detected",
    )


def _pose(detected: bool, horizontal: bool = False) -> PoseResult:
    if not detected:
        return PoseResult(detected=False, status_text="Pose not detected")
    landmarks = [LandmarkPoint(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
    if horizontal:
        landmarks[11] = LandmarkPoint(0.30, 0.45, 0.0, 1.0)
        landmarks[12] = LandmarkPoint(0.70, 0.45, 0.0, 1.0)
        landmarks[23] = LandmarkPoint(0.32, 0.52, 0.0, 1.0)
        landmarks[24] = LandmarkPoint(0.68, 0.52, 0.0, 1.0)
    else:
        landmarks[11] = LandmarkPoint(0.40, 0.30, 0.0, 1.0)
        landmarks[12] = LandmarkPoint(0.60, 0.30, 0.0, 1.0)
        landmarks[23] = LandmarkPoint(0.42, 0.70, 0.0, 1.0)
        landmarks[24] = LandmarkPoint(0.58, 0.70, 0.0, 1.0)
    return PoseResult(
        detected=True,
        landmarks=landmarks,
        key_points=[(lm.x, lm.y) for lm in landmarks[:8]],
        body_position=estimate_body_position(landmarks),
        status_text="Pose detected",
    )


class CameraTests(unittest.TestCase):
    def test_placeholder_camera_unavailable(self):
        cam = CameraHandler(width=320, height=240)
        frame = cam.placeholder_frame("Camera unavailable")
        self.assertEqual(frame.shape, (240, 320, 3))

    def test_camera_unavailable_does_not_raise(self):
        class FakeCap:
            def isOpened(self):
                return False

            def release(self):
                return None

        cam = CameraHandler(camera_index=99)
        if cam.__class__.__module__:
            with patch("camera.camera_handler.cv2") as mock_cv:
                mock_cv.VideoCapture.return_value = FakeCap()
                mock_cv.CAP_AVFOUNDATION = 1200
                mock_cv.CAP_DSHOW = 700
                ok = cam.start(99)
                self.assertFalse(ok)
                self.assertIn("unavailable", cam.error_message.lower())

    def test_demo_video_missing_file(self):
        cam = CameraHandler()
        ok = cam.start_video("/tmp/does-not-exist-babyguard.mp4")
        self.assertFalse(ok)
        self.assertIn("not found", cam.error_message.lower())

    def test_find_demo_videos(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "baby.mp4").write_bytes(b"placeholder")
            found = find_demo_videos(folder)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].name, "baby.mp4")


class DetectionHelperTests(unittest.TestCase):
    def test_subject_detected_structure(self):
        person = _person(True)
        self.assertTrue(person.detected)
        self.assertEqual(person.status_text, "Baby/Person detected")
        self.assertIsNotNone(person.center_norm)

    def test_subject_not_detected(self):
        person = _person(False)
        self.assertFalse(person.detected)
        self.assertEqual(person.status_text, "No person detected")

    def test_pose_detected(self):
        pose = _pose(True)
        self.assertTrue(pose.detected)
        self.assertEqual(pose.status_text, "Pose detected")
        self.assertIn(pose.body_position, ("Upright", "Horizontal", "Unknown"))

    def test_pose_unavailable(self):
        pose = _pose(False)
        self.assertFalse(pose.detected)
        self.assertEqual(pose.status_text, "Pose not detected")

    def test_horizontal_body_position(self):
        pose = _pose(True, horizontal=True)
        self.assertEqual(pose.body_position, "Horizontal")

    def test_bbox_pose_fallback(self):
        points = landmarks_from_bbox((0.2, 0.1, 0.8, 0.9))
        self.assertEqual(len(points), 33)
        self.assertGreater(points[0].visibility, 0.5)
        self.assertEqual(estimate_body_position(points), "Upright")

    def test_missing_yolo_model(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_yolo_weights(Path(tmp)))


class MovementTests(unittest.TestCase):
    def test_low_movement(self):
        analyzer = MovementAnalyzer(history=6)
        point = [(0.50, 0.40), (0.45, 0.42), (0.55, 0.42), (0.50, 0.60)]
        result = None
        for _ in range(8):
            result = analyzer.update(key_points=point, center_norm=(0.50, 0.45), detected=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.level, "LOW")

    def test_high_movement(self):
        analyzer = MovementAnalyzer(history=6)
        result = None
        for i in range(10):
            shift = 0.12 * i
            points = [
                (0.20 + shift, 0.30),
                (0.25 + shift, 0.32),
                (0.30 + shift, 0.32),
                (0.22 + shift, 0.55),
            ]
            result = analyzer.update(
                key_points=points,
                center_norm=(0.20 + shift, 0.40),
                detected=True,
            )
        self.assertIsNotNone(result)
        self.assertEqual(result.level, "HIGH")


class ActivityTests(unittest.TestCase):
    def test_active_when_high(self):
        analyzer = ActivityAnalyzer()
        movement = MovementAnalyzer().update(detected=False)
        movement.level = "HIGH"
        state = analyzer.update(movement, present=True)
        self.assertEqual(state.label, "ACTIVE")

    def test_sleep_estimate_after_stillness(self):
        analyzer = ActivityAnalyzer()
        analyzer._low_since = time.time() - (config.SLEEP_STILL_SECONDS + 1)
        movement = MovementAnalyzer().update(detected=False)
        movement.level = "LOW"
        state = analyzer.update(movement, body_position="Upright", present=True)
        self.assertEqual(state.label, "POSSIBLY SLEEPING")

    def test_absent_subject_is_not_sleeping(self):
        analyzer = ActivityAnalyzer()
        movement = MovementAnalyzer().update(detected=False)
        movement.level = "LOW"
        state = analyzer.update(movement, present=False)
        self.assertEqual(state.label, "POSSIBLY AWAKE")


class ZoneAndAlertTests(unittest.TestCase):
    def test_monitoring_zone_violation(self):
        logic = MonitoringLogic()
        zone = (0.2, 0.2, 0.8, 0.8)
        pose = _pose(True)
        movement = MovementAnalyzer().update(detected=True, center_norm=(0.5, 0.5), key_points=[(0.5, 0.5)] * 4)
        from analysis.activity_analyzer import ActivityState

        activity = ActivityState("POSSIBLY AWAKE", "Possibly Awake", 0.0, "Upright")
        inside = logic.evaluate(_person(True, 0.5, 0.5), pose, movement, activity, zone, True)
        self.assertEqual(inside.zone_status, "inside")

        decision = None
        for _ in range(config.ZONE_OUTSIDE_PERSIST_FRAMES + 1):
            decision = logic.evaluate(_person(True, 0.05, 0.05, 0.08), pose, movement, activity, zone, True)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.zone_status, "outside")
        self.assertEqual(decision.alert_kind, "outside")

    def test_alert_cooldown(self):
        manager = AlertManager()
        manager.voice_enabled = False
        manager.beep_enabled = False
        manager.cooldown = 8.0
        first = manager.handle("unusual")
        self.assertTrue(first.is_alert)
        self.assertTrue(first.fired_audio)
        manager.handle("normal")
        second = manager.handle("unusual")
        self.assertTrue(second.is_alert)
        self.assertFalse(second.fired_audio)

    def test_voice_alert_disabled(self):
        manager = AlertManager()
        manager.voice_enabled = False
        manager.beep_enabled = False
        spoken = {"called": False}

        def fake_speak(_text):
            spoken["called"] = True

        manager._speak = fake_speak  # type: ignore[method-assign]
        manager.handle("outside")
        self.assertFalse(spoken["called"])


class HelperTests(unittest.TestCase):
    def test_geometry(self):
        rect = rect_from_points(0.8, 0.7, 0.2, 0.1)
        self.assertEqual(rect, (0.2, 0.1, 0.8, 0.7))
        self.assertTrue(point_in_rect(0.5, 0.4, rect))
        self.assertFalse(point_in_rect(0.05, 0.05, rect))
        self.assertEqual(bbox_center((0.0, 0.0, 1.0, 0.5)), (0.5, 0.25))
        self.assertGreater(bbox_intersection_fraction((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.5, 1.0)), 0.4)
        self.assertAlmostEqual(landmark_displacement([(0.0, 0.0)], [(0.3, 0.4)]), 0.5)


if __name__ == "__main__":
    unittest.main()
