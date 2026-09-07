"""
BabyGuard AI dashboard.

Tkinter owns the window. A background thread reads the camera and runs
YOLO + MediaPipe so the interface stays responsive.
"""

from __future__ import annotations

import sys
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import config
from alerts.alert_manager import AlertEvent, AlertManager
from analysis.activity_analyzer import ActivityAnalyzer, ActivityState
from analysis.monitoring_logic import MonitorDecision, MonitoringLogic
from analysis.movement_analyzer import MovementAnalyzer, MovementResult
from camera.camera_handler import CameraHandler, find_demo_videos
from detection.object_detector import ObjectDetector, PersonDetection
from detection.pose_detector import PoseDetector, PoseResult
from utils.helpers import draw_label, format_hms, frame_to_photo, widget_to_norm

if sys.platform == "darwin":
    UI_FONT = "Helvetica Neue"
elif sys.platform.startswith("win"):
    UI_FONT = "Segoe UI"
else:
    UI_FONT = "DejaVu Sans"

BG = "#07111F"
PANEL = "#122038"
PANEL_ALT = "#182844"
ACCENT = "#5EEAD4"
ACCENT_DIM = "#134E4A"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
SUCCESS = "#34D399"
WARNING = "#FBBF24"
DANGER = "#FB7185"
LINE = "#243656"
VIDEO_BG = "#070B14"


@dataclass
class EngineState:
    running: bool = False
    voice_enabled: bool = False
    zone_enabled: bool = False
    zone: Optional[tuple[float, float, float, float]] = None
    draft_zone: Optional[tuple[float, float, float, float]] = None
    reset_requested: bool = False


@dataclass
class FramePacket:
    frame: np.ndarray
    person: PersonDetection
    pose: PoseResult
    movement: MovementResult
    activity: ActivityState
    decision: MonitorDecision
    alert: AlertEvent
    fps: float
    camera_ok: bool
    message: str
    events: list[str] = field(default_factory=list)


class BabyGuardApp:
    """Main application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(config.WINDOW_TITLE)
        self.root.configure(bg=BG)
        self.root.minsize(1120, 740)
        self.root.geometry(config.WINDOW_SIZE)

        self.camera = CameraHandler(
            camera_index=config.CAMERA_INDEX,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
        )
        self.detector: Optional[ObjectDetector] = None
        self.pose_detector: Optional[PoseDetector] = None
        self.movement = MovementAnalyzer()
        self.activity = ActivityAnalyzer()
        self.monitoring = MonitoringLogic()
        self.alerts = AlertManager()
        self.alerts.voice_enabled = config.VOICE_ALERTS_ENABLED
        self.alerts.on_beep = self._beep

        self._state = EngineState(
            voice_enabled=config.VOICE_ALERTS_ENABLED,
            zone_enabled=config.MONITORING_ZONE_ENABLED,
            zone=config.MONITORING_ZONE if config.MONITORING_ZONE_ENABLED else None,
        )
        self._state_lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._worker: Optional[threading.Thread] = None
        self._stop_worker = threading.Event()
        self._closing = False
        self._booted = False
        self._photo = None
        self._disp = {"scale": 1.0, "ox": 0, "oy": 0, "fw": 640, "fh": 480}
        self._selecting = False
        self._drag_start = None
        self._log_lines: deque[str] = deque(maxlen=config.EVENT_LOG_LIMIT)
        self._prev_move = None
        self._prev_state = None
        self._prev_alert = None
        self._prev_subject = None
        self._prev_zone_status = None

        self.subject_var = tk.StringVar(value="Not Detected")
        self.pose_var = tk.StringVar(value="Not Detected")
        self.move_var = tk.StringVar(value="—")
        self.state_var = tk.StringVar(value="—")
        self.position_var = tk.StringVar(value="Unknown")
        self.zone_var = tk.StringVar(value="Disabled")
        self.monitor_var = tk.StringVar(value="Inactive")
        self.alert_var = tk.StringVar(value="🟢 Monitoring normally")
        self.fps_var = tk.StringVar(value="FPS: —")
        self.model_var = tk.StringVar(value="Models: loading…")
        self.status_var = tk.StringVar(value="Starting BabyGuard AI…")
        self.voice_var = tk.BooleanVar(value=config.VOICE_ALERTS_ENABLED)
        self.source_title_var = tk.StringVar(value="LIVE CAMERA")
        self._pending_video: Optional[str] = None

        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._boot)
        self.root.after(config.UI_REFRESH_MS, self._refresh_ui)

    def run(self) -> None:
        self.root.mainloop()

    # ------------------------------------------------------------------ UI
    def _card(self, parent, title: str, title_var: Optional[tk.StringVar] = None) -> tuple[tk.Frame, tk.Frame]:
        wrap = tk.Frame(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        header_kwargs = {"textvariable": title_var} if title_var is not None else {"text": title}
        header = tk.Label(
            wrap,
            bg=PANEL,
            fg=MUTED,
            font=(UI_FONT, 10, "bold"),
            anchor="w",
            **header_kwargs,
        )
        header.pack(fill="x", padx=14, pady=(10, 2))
        body = tk.Frame(wrap, bg=PANEL)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        return wrap, body

    def _button(self, parent, text, command, bg, fg="#06201C", width=12) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=10,
            pady=8,
            font=(UI_FONT, 11, "bold"),
            cursor="hand2",
            width=width,
        )

    def _stat(self, parent, caption: str, variable: tk.StringVar, color: str = TEXT) -> tk.Label:
        box = tk.Frame(parent, bg=PANEL_ALT)
        box.pack(side="left", fill="both", expand=True, padx=4)
        tk.Label(box, text=caption, bg=PANEL_ALT, fg=MUTED, font=(UI_FONT, 9, "bold")).pack(
            anchor="w", padx=10, pady=(8, 0)
        )
        label = tk.Label(
            box,
            textvariable=variable,
            bg=PANEL_ALT,
            fg=color,
            font=(UI_FONT, 14, "bold"),
            wraplength=180,
            justify="left",
            anchor="w",
        )
        label.pack(anchor="w", padx=10, pady=(2, 10))
        return label

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=20, pady=(14, 4))
        tk.Label(
            header,
            text="👶 BABYGUARD AI",
            bg=BG,
            fg=ACCENT,
            font=(UI_FONT, 26, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Intelligent Camera Monitoring System   ·   Educational prototype — not a substitute for adult supervision",
            bg=BG,
            fg=MUTED,
            font=(UI_FONT, 13),
        ).pack(anchor="w")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=8)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left, left_body = self._card(body, "LIVE CAMERA", self.source_title_var)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.video_wrap = tk.Frame(
            left_body,
            bg=VIDEO_BG,
            width=config.PREVIEW_MAX_WIDTH,
            height=config.PREVIEW_MAX_HEIGHT,
        )
        self.video_wrap.pack(fill="both", expand=True)
        self.video_wrap.pack_propagate(False)
        self.video_label = tk.Label(self.video_wrap, bg=VIDEO_BG, cursor="crosshair")
        self.video_label.pack(fill="both", expand=True)
        self.video_label.bind("<ButtonPress-1>", self._on_drag_start)
        self.video_label.bind("<B1-Motion>", self._on_drag_move)
        self.video_label.bind("<ButtonRelease-1>", self._on_drag_end)

        cam_btns = tk.Frame(left_body, bg=PANEL)
        cam_btns.pack(fill="x", pady=(10, 0))
        self._button(cam_btns, "START", self.start_monitoring, ACCENT, width=10).pack(
            side="left", padx=(0, 6)
        )
        self._button(cam_btns, "STOP", self.stop_monitoring, "#334155", TEXT, width=10).pack(
            side="left", padx=(0, 6)
        )
        self._button(cam_btns, "SET AREA", self._begin_set_area, "#38BDF8", "#082F49", width=11).pack(
            side="left", padx=(0, 6)
        )
        self._button(cam_btns, "RESET", self.reset_session, WARNING, "#3F2E00", width=10).pack(
            side="left", padx=(0, 6)
        )
        self._button(cam_btns, "DEMO VIDEO", self.start_demo_video, "#A78BFA", "#2E1064", width=12).pack(
            side="left", padx=(0, 6)
        )
        self._button(cam_btns, "LIVE", self.start_live_camera, "#334155", TEXT, width=8).pack(
            side="left"
        )

        opt_row = tk.Frame(left_body, bg=PANEL)
        opt_row.pack(fill="x", pady=(10, 0))
        self.voice_check = tk.Checkbutton(
            opt_row,
            text="Voice Alerts",
            variable=self.voice_var,
            command=self._on_voice_toggle,
            bg=PANEL,
            fg=TEXT,
            selectcolor=PANEL_ALT,
            activebackground=PANEL,
            activeforeground=TEXT,
            font=(UI_FONT, 11),
            highlightthickness=0,
        )
        self.voice_check.pack(side="left")
        tk.Label(
            opt_row,
            text="START = webcam.  DEMO VIDEO = looping baby clip for the viva.",
            bg=PANEL,
            fg=MUTED,
            font=(UI_FONT, 10),
        ).pack(side="left", padx=12)

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(2, weight=1)

        stats, stats_body = self._card(right, "STATUS PANEL")
        stats.pack(fill="x", pady=(0, 8))
        row1 = tk.Frame(stats_body, bg=PANEL)
        row1.pack(fill="x")
        self.subject_label = self._stat(row1, "SUBJECT", self.subject_var, SUCCESS)
        self.pose_label = self._stat(row1, "POSE", self.pose_var)
        self.move_label = self._stat(row1, "MOVEMENT", self.move_var, ACCENT)
        row2 = tk.Frame(stats_body, bg=PANEL)
        row2.pack(fill="x", pady=(8, 0))
        self.state_label = self._stat(row2, "ESTIMATED STATE", self.state_var, WARNING)
        self.position_label = self._stat(row2, "BODY POSITION", self.position_var)
        self.zone_label = self._stat(row2, "MONITORING ZONE", self.zone_var)

        alert_card, alert_body = self._card(right, "ALERT STATUS")
        alert_card.pack(fill="x", pady=(0, 8))
        self.alert_badge = tk.Label(
            alert_body,
            textvariable=self.alert_var,
            bg=PANEL_ALT,
            fg=SUCCESS,
            font=(UI_FONT, 16, "bold"),
            anchor="w",
            padx=12,
            pady=14,
            wraplength=420,
            justify="left",
        )
        self.alert_badge.pack(fill="x")
        tk.Label(
            alert_body,
            text="Wording is visual only. This is not a medical or safety guarantee.",
            bg=PANEL,
            fg=MUTED,
            font=(UI_FONT, 9),
            wraplength=420,
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(8, 0))

        log_card, log_body = self._card(right, "EVENT LOG")
        log_card.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            log_body,
            height=10,
            bg=PANEL_ALT,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=(UI_FONT, 11),
            wrap="word",
            state="disabled",
        )
        self.log_text.pack(fill="both", expand=True)
        self._button(log_body, "CLEAR LOG", self.clear_log, "#334155", TEXT, width=12).pack(
            anchor="e", pady=(8, 0)
        )

        footer = tk.Frame(self.root, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        footer.pack(fill="x", padx=20, pady=(4, 14))
        tk.Label(
            footer,
            textvariable=self.status_var,
            bg=PANEL,
            fg=TEXT,
            font=(UI_FONT, 11),
            anchor="w",
        ).pack(side="left", padx=12, pady=8)
        tk.Label(footer, textvariable=self.model_var, bg=PANEL, fg=MUTED, font=(UI_FONT, 10)).pack(
            side="right", padx=12
        )
        tk.Label(footer, textvariable=self.fps_var, bg=PANEL, fg=MUTED, font=(UI_FONT, 10)).pack(
            side="right", padx=8
        )
        tk.Label(
            footer,
            textvariable=self.monitor_var,
            bg=PANEL,
            fg=ACCENT,
            font=(UI_FONT, 10, "bold"),
        ).pack(side="right", padx=8)

        self._log("BabyGuard AI ready. Click START to begin monitoring.")

    def _bind_keys(self) -> None:
        self.root.bind("<space>", lambda _e: self.start_monitoring())
        self.root.bind("<Escape>", lambda _e: self.stop_monitoring())

    # ------------------------------------------------------------------ boot
    def _boot(self) -> None:
        if self._closing:
            return
        self.status_var.set("Checking MediaPipe Pose (isolated safety probe)…")
        self.root.update_idletasks()

        # Pose before YOLO: on some Macs MediaPipe aborts if PyTorch loaded first.
        try:
            self.pose_detector = PoseDetector()
        except Exception as exc:
            self.pose_detector = None
            self._log(f"MediaPipe Pose failed to initialize: {exc}")

        self.status_var.set("Loading YOLO person model…")
        self.root.update_idletasks()

        try:
            self.detector = ObjectDetector()
        except Exception as exc:
            self.detector = None
            self._log(f"YOLO failed to initialize: {exc}")

        model_bits = []
        if self.detector and self.detector.ready:
            model_bits.append(f"YOLO {self.detector.model_name} ({self.detector.device})")
        else:
            err = getattr(self.detector, "error_message", "YOLO unavailable") if self.detector else "YOLO unavailable"
            model_bits.append("YOLO off")
            self._log(err)
        if self.pose_detector and self.pose_detector.available:
            model_bits.append(f"Pose/{self.pose_detector.backend}")
            if self.pose_detector.backend == "box":
                self._log(
                    "MediaPipe Pose is unstable on this Mac. "
                    "Using approximate pose from the person box."
                )
            elif self.pose_detector.error_message:
                self._log(self.pose_detector.error_message)
        else:
            err = (
                getattr(self.pose_detector, "error_message", "Pose unavailable")
                if self.pose_detector
                else "Pose unavailable"
            )
            model_bits.append("Pose off")
            self._log(err)

        self.model_var.set("  ·  ".join(model_bits))
        self.status_var.set("Models loaded. START = live camera. DEMO VIDEO = presentation clip.")
        self._booted = True
        self._show_placeholder("Click START to begin")

        if self.alerts.tts_error and not self.alerts.tts_available:
            self._log(self.alerts.tts_error)

    # ------------------------------------------------------------------ controls
    def start_live_camera(self) -> None:
        self._pending_video = None
        self.source_title_var.set("LIVE CAMERA")
        self.start_monitoring(video_path=None)

    def start_demo_video(self) -> None:
        if not self._booted:
            self.status_var.set("Please wait — models are still loading.")
            return
        path = self._choose_demo_video()
        if not path:
            return
        self._pending_video = path
        self.source_title_var.set(f"DEMO VIDEO  ·  {Path(path).name}")
        self.start_monitoring(video_path=path)

    def _choose_demo_video(self) -> Optional[str]:
        clips = find_demo_videos()
        if len(clips) == 1:
            return str(clips[0])
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        chosen = filedialog.askopenfilename(
            parent=self.root,
            title="Choose a demo video clip (baby / crib footage for the viva)",
            initialdir=str(config.DATA_DIR if clips else Path.home()),
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.webm"),
                ("All files", "*.*"),
            ],
        )
        return chosen or None

    def start_monitoring(self, video_path: Optional[str] = None) -> None:
        if not self._booted:
            self.status_var.set("Please wait — models are still loading.")
            return
        if self._state.running:
            self.stop_monitoring()

        if video_path is None and self._pending_video and self.source_title_var.get().startswith("DEMO"):
            video_path = self._pending_video

        if video_path:
            ok = self.camera.start_video(video_path, loop=config.DEMO_VIDEO_LOOP)
            self.source_title_var.set(f"DEMO VIDEO  ·  {Path(video_path).name}")
        else:
            ok = self.camera.start()
            self.source_title_var.set("LIVE CAMERA")

        if not ok:
            message = self.camera.error_message or "Camera unavailable"
            self.status_var.set(message)
            self._show_placeholder(message)
            self._log(message)
            if video_path:
                messagebox.showerror("Demo video", message, parent=self.root)
            return

        self.movement.reset()
        self.activity.reset()
        self.monitoring.reset()
        self.alerts.reset()
        self._prev_move = None
        self._prev_state = None
        self._prev_alert = None
        self._prev_subject = None
        self._prev_zone_status = None

        with self._state_lock:
            self._state.running = True
            self._state.reset_requested = False

        self._stop_worker.clear()
        self._worker = threading.Thread(target=self._process_loop, name="babyguard-engine", daemon=True)
        self._worker.start()
        self.monitor_var.set("Active")
        label = self.camera.source_label
        self.status_var.set(f"Monitoring started — {label}")
        self._log(f"Monitoring started ({label})")

    def stop_monitoring(self) -> None:
        with self._state_lock:
            was_running = self._state.running
            self._state.running = False
        self._stop_worker.set()
        worker = self._worker
        if worker is not None and worker.is_alive() and threading.current_thread() is not worker:
            worker.join(timeout=1.2)
        self.camera.stop()
        self.monitor_var.set("Inactive")
        if was_running:
            self.status_var.set("Monitoring stopped")
            self._log("Monitoring stopped")
            self.fps_var.set("FPS: —")

    def reset_session(self) -> None:
        with self._state_lock:
            self._state.zone = None
            self._state.zone_enabled = False
            self._state.draft_zone = None
            self._state.reset_requested = True
        self._selecting = False
        self.alerts.reset()
        self.zone_var.set("Disabled")
        self.alert_var.set("🟢 Monitoring normally")
        self._color_alert("normal")
        self.status_var.set("Session reset. Monitoring area cleared.")
        self._log("Session reset")

    def clear_log(self) -> None:
        self._log_lines.clear()
        self._render_log()

    def _on_voice_toggle(self) -> None:
        enabled = bool(self.voice_var.get())
        with self._state_lock:
            self._state.voice_enabled = enabled
        self.alerts.voice_enabled = enabled
        if enabled and not self.alerts.tts_available:
            self.voice_var.set(False)
            self.alerts.voice_enabled = False
            with self._state_lock:
                self._state.voice_enabled = False
            self.status_var.set(self.alerts.tts_error or "Voice alerts unavailable")
            self._log("Voice alerts unavailable — visual alerts still work")
            return
        self._log("Voice alerts enabled" if enabled else "Voice alerts disabled")

    def _begin_set_area(self) -> None:
        self._selecting = True
        self._drag_start = None
        self.status_var.set("Draw a rectangle on the live camera to set the monitoring area.")
        self._log("Set monitoring area: drag on the camera preview")

    def _on_drag_start(self, event) -> None:
        if not self._selecting:
            return
        self._drag_start = (event.x, event.y)

    def _on_drag_move(self, event) -> None:
        if not self._selecting or self._drag_start is None:
            return
        zone = self._points_to_zone(self._drag_start, (event.x, event.y))
        if zone:
            with self._state_lock:
                self._state.draft_zone = zone

    def _on_drag_end(self, event) -> None:
        if not self._selecting or self._drag_start is None:
            return
        zone = self._points_to_zone(self._drag_start, (event.x, event.y))
        self._selecting = False
        self._drag_start = None
        if not zone:
            self.status_var.set("Monitoring area was too small. Try again.")
            with self._state_lock:
                self._state.draft_zone = None
            return
        with self._state_lock:
            self._state.zone = zone
            self._state.zone_enabled = True
            self._state.draft_zone = None
        self.zone_var.set("Inside")
        self.status_var.set("Monitoring area set")
        self._log("Monitoring area set")

    def _points_to_zone(self, start, end) -> Optional[tuple[float, float, float, float]]:
        a = widget_to_norm(
            start[0], start[1], self._disp["scale"], self._disp["ox"], self._disp["oy"],
            self._disp["fw"], self._disp["fh"],
        )
        b = widget_to_norm(
            end[0], end[1], self._disp["scale"], self._disp["ox"], self._disp["oy"],
            self._disp["fw"], self._disp["fh"],
        )
        if not a or not b:
            return None
        from utils.helpers import rect_from_points

        x1, y1, x2, y2 = rect_from_points(a[0], a[1], b[0], b[1])
        if (x2 - x1) < 0.08 or (y2 - y1) < 0.08:
            return None
        return x1, y1, x2, y2

    # ------------------------------------------------------------------ engine
    def _snapshot_state(self) -> EngineState:
        with self._state_lock:
            reset = self._state.reset_requested
            if reset:
                self._state.reset_requested = False
            return EngineState(
                running=self._state.running,
                voice_enabled=self._state.voice_enabled,
                zone_enabled=self._state.zone_enabled,
                zone=self._state.zone,
                draft_zone=self._state.draft_zone,
                reset_requested=reset,
            )

    def _process_loop(self) -> None:
        fps_window: deque[float] = deque(maxlen=16)
        frame_index = 0
        last_person = _empty_person()
        last_pose = PoseResult(detected=False, status_text="Pose not detected")
        last_movement = MovementResult("LOW", 0.0, "none", 0)
        last_activity = ActivityState("POSSIBLY AWAKE", "Possibly Awake", 0.0, "Unknown")

        while not self._stop_worker.is_set() and not self._closing:
            t0 = time.time()
            state = self._snapshot_state()
            if state.reset_requested:
                self.movement.reset()
                self.activity.reset()
                self.monitoring.reset()

            if not state.running:
                break

            frame = self.camera.read()
            camera_ok = frame is not None
            message = ""
            events: list[str] = []

            if frame is None:
                frame = self.camera.placeholder_frame(self.camera.error_message or "Camera unavailable")
                message = self.camera.error_message or "Camera unavailable"
                person = _empty_person()
                pose = PoseResult(detected=False, status_text="Pose not detected")
                movement = MovementResult("LOW", 0.0, "none", 0)
                activity = self.activity.update(movement, present=False)
                decision = self.monitoring.evaluate(
                    person, pose, movement, activity, state.zone, state.zone_enabled
                )
            else:
                frame_index += 1
                should_process = frame_index % max(1, config.PROCESS_EVERY_N_FRAMES) == 0
                if should_process:
                    person = self._detect_person(frame)
                    pose = self._detect_pose(frame, person)
                    if not person.detected and pose.detected:
                        person = _person_from_pose(pose, frame.shape)
                    last_person, last_pose = person, pose
                else:
                    person, pose = last_person, last_pose

                movement = self.movement.update(
                    key_points=pose.key_points if pose.detected else None,
                    center_norm=person.center_norm,
                    detected=person.detected,
                )
                activity = self.activity.update(
                    movement,
                    body_position=pose.body_position if pose.detected else "Unknown",
                    present=person.detected,
                )
                decision = self.monitoring.evaluate(
                    person, pose, movement, activity, state.zone, state.zone_enabled
                )
                last_movement, last_activity = movement, activity
                events.extend(decision.events)

            self.alerts.voice_enabled = state.voice_enabled
            alert = self.alerts.handle(decision.alert_kind)
            annotated = self._annotate(frame, person, pose, decision, state)

            fps_window.append(time.time() - t0)
            fps = 1.0 / (sum(fps_window) / len(fps_window)) if fps_window else 0.0

            packet = FramePacket(
                frame=annotated,
                person=person,
                pose=pose,
                movement=last_movement if camera_ok else movement,
                activity=last_activity if camera_ok else activity,
                decision=decision,
                alert=alert,
                fps=fps,
                camera_ok=camera_ok,
                message=message,
                events=events,
            )
            with self._result_lock:
                self._latest = packet

            # Keep the laptop cool and the UI snappy.
            elapsed = time.time() - t0
            target_fps = self.camera.video_fps if self.camera.is_video else float(config.TARGET_FPS)
            delay = max(0.0, (1.0 / max(target_fps, 1)) - elapsed)
            time.sleep(delay)

    def _detect_person(self, frame) -> PersonDetection:
        if self.detector and self.detector.ready:
            return self.detector.detect(frame)
        return _empty_person(self.detector.error_message if self.detector else "YOLO unavailable")

    def _detect_pose(self, frame, person: Optional[PersonDetection] = None) -> PoseResult:
        bbox = person.bbox_norm if person is not None and person.detected else None
        if self.pose_detector and self.pose_detector.available:
            return self.pose_detector.detect(frame, bbox_norm=bbox)
        return PoseResult(
            detected=False,
            status_text="Pose not detected",
            error_message=self.pose_detector.error_message if self.pose_detector else "Pose unavailable",
        )

    def _annotate(
        self,
        frame: np.ndarray,
        person: PersonDetection,
        pose: PoseResult,
        decision: MonitorDecision,
        state: EngineState,
    ) -> np.ndarray:
        out = frame.copy()
        if cv2 is None:
            return out
        h, w = out.shape[:2]

        zone = state.draft_zone or (state.zone if state.zone_enabled else None)
        if zone:
            x1, y1, x2, y2 = zone
            color = (80, 220, 255) if state.draft_zone else (255, 200, 80)
            cv2.rectangle(
                out,
                (int(x1 * w), int(y1 * h)),
                (int(x2 * w), int(y2 * h)),
                color,
                2,
            )
            draw_label(out, "MONITORING AREA", (int(x1 * w) + 8, int(y1 * h) + 22), color)

        if self.detector:
            self.detector.draw(out, person)
        if self.pose_detector:
            self.pose_detector.draw(out, pose)

        lines = [
            ("SUBJECT: DETECTED" if person.detected else "SUBJECT: NOT DETECTED",
             (80, 220, 140) if person.detected else (180, 180, 180)),
            (f"POSE: {'DETECTED' if pose.detected else 'NOT DETECTED'}", (80, 200, 255)),
            (f"MOVE: {decision.movement_level}", (80, 220, 255)),
            (f"STATE: {decision.estimated_state_display.upper()}", (180, 200, 255)),
        ]
        if self.camera.is_video:
            lines.insert(0, ("DEMO CLIP  (looping presentation video)", (180, 160, 255)))
        y = 28
        for text, color in lines:
            draw_label(out, text, (16, y), color)
            y += 26

        if decision.alert_kind != "normal":
            banner = decision.alert_message.upper()
            cv2.rectangle(out, (0, h - 46), (w, h), (20, 40, 180), -1)
            draw_label(out, banner, (16, h - 16), (220, 230, 255))
        return out

    # ------------------------------------------------------------------ UI refresh
    def _refresh_ui(self) -> None:
        if self._closing:
            return
        packet = None
        with self._result_lock:
            packet = self._latest

        if packet is None:
            if not self._state.running:
                pass
        else:
            self._apply_packet(packet)

        self.root.after(config.UI_REFRESH_MS, self._refresh_ui)

    def _apply_packet(self, packet: FramePacket) -> None:
        wrap_w = max(self.video_wrap.winfo_width(), 320)
        wrap_h = max(self.video_wrap.winfo_height(), 240)
        photo, scale, ox, oy, dw, dh = frame_to_photo(packet.frame, wrap_w, wrap_h)
        self._photo = photo
        self.video_label.configure(image=photo)
        fh, fw = packet.frame.shape[:2]
        self._disp = {"scale": scale, "ox": ox, "oy": oy, "fw": fw, "fh": fh}

        person_txt = "Detected" if packet.person.detected else "Not Detected"
        if packet.pose.backend == "box" and packet.pose.detected:
            pose_txt = "Approximate"
        else:
            pose_txt = "Detected" if packet.pose.detected else "Not Detected"
        self.subject_var.set(person_txt)
        self.pose_var.set(pose_txt)
        self.move_var.set(packet.decision.movement_level.title())
        self.state_var.set(packet.decision.estimated_state_display)
        self.position_var.set(packet.decision.body_position)
        if not packet.decision.zone_enabled:
            self.zone_var.set("Disabled")
        else:
            self.zone_var.set(packet.decision.zone_status.title())

        prefix = {
            "normal": "🟢 Monitoring normally",
            "unusual": "🟠 Possible unusual activity",
            "outside": "🔴 Subject outside monitoring area",
            "missing": "🔴 Subject no longer visible",
        }
        self.alert_var.set(prefix.get(packet.decision.alert_kind, packet.decision.alert_message))
        self._color_alert(packet.decision.alert_kind)
        self._tint_status_labels(packet)

        if packet.camera_ok:
            self.fps_var.set(f"FPS: {packet.fps:.0f}")
            if packet.message:
                self.status_var.set(packet.message)
        else:
            self.fps_var.set("FPS: —")
            self.status_var.set(packet.message or "Camera unavailable")

        self._emit_transitions(packet)
        for event in packet.events:
            self._log(event)

    def _tint_status_labels(self, packet: FramePacket) -> None:
        self.subject_label.configure(fg=SUCCESS if packet.person.detected else MUTED)
        self.pose_label.configure(fg=ACCENT if packet.pose.detected else MUTED)
        move_color = {"LOW": MUTED, "NORMAL": SUCCESS, "HIGH": DANGER}.get(
            packet.decision.movement_level, TEXT
        )
        self.move_label.configure(fg=move_color)
        state_color = {
            "POSSIBLY SLEEPING": ACCENT,
            "POSSIBLY AWAKE": SUCCESS,
            "ACTIVE": WARNING,
        }.get(packet.decision.estimated_state, TEXT)
        self.state_label.configure(fg=state_color)

    def _color_alert(self, kind: str) -> None:
        styles = {
            "normal": (PANEL_ALT, SUCCESS),
            "unusual": ("#3F2E00", WARNING),
            "outside": ("#3F0D16", DANGER),
            "missing": ("#3F0D16", DANGER),
        }
        bg, fg = styles.get(kind, (PANEL_ALT, TEXT))
        self.alert_badge.configure(bg=bg, fg=fg)

    def _emit_transitions(self, packet: FramePacket) -> None:
        move = packet.decision.movement_level
        if self._prev_move and move != self._prev_move:
            self._log(f"Activity changed to {move}")
        self._prev_move = move

        state = packet.decision.estimated_state
        if self._prev_state and state != self._prev_state:
            self._log(f"Estimated state: {packet.decision.estimated_state_display}")
        self._prev_state = state

        if packet.person.detected != self._prev_subject and self._prev_subject is not None:
            if not packet.person.detected:
                self._log("Subject lost")
        self._prev_subject = packet.person.detected

        if packet.decision.zone_status == "outside" and self._prev_zone_status != "outside":
            self._log("Subject outside monitoring area")
        self._prev_zone_status = packet.decision.zone_status

        kind = packet.decision.alert_kind
        if kind != self._prev_alert and kind != "normal":
            self._log(packet.decision.alert_message)
        if kind == "normal" and self._prev_alert not in (None, "normal"):
            self._log("Activity returned to normal")
        self._prev_alert = kind

    def _show_placeholder(self, message: str) -> None:
        frame = self.camera.placeholder_frame(message)
        wrap_w = max(self.video_wrap.winfo_width(), config.PREVIEW_MAX_WIDTH)
        wrap_h = max(self.video_wrap.winfo_height(), 320)
        photo, scale, ox, oy, dw, dh = frame_to_photo(frame, wrap_w, wrap_h)
        self._photo = photo
        self.video_label.configure(image=photo)
        self._disp = {"scale": scale, "ox": ox, "oy": oy, "fw": frame.shape[1], "fh": frame.shape[0]}

    def _log(self, message: str) -> None:
        line = f"{format_hms()} - {message}"
        if self._log_lines and self._log_lines[-1].endswith(message):
            return
        self._log_lines.append(line)
        self._render_log()

    def _render_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        shown = list(self._log_lines)[-config.EVENT_LOG_DISPLAY :]
        self.log_text.insert("end", "\n".join(shown))
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _beep(self, _kind: str) -> None:
        try:
            self.root.bell()
        except Exception:
            AlertManager.system_beep()

    def _on_close(self) -> None:
        self._closing = True
        self.stop_monitoring()
        if self.pose_detector:
            self.pose_detector.close()
        self.root.destroy()


def _empty_person(error: str = "") -> PersonDetection:
    return PersonDetection(
        detected=False,
        bbox_px=None,
        bbox_norm=None,
        center_norm=None,
        confidence=0.0,
        status_text="No person detected",
        error_message=error,
    )


def _person_from_pose(pose: PoseResult, shape) -> PersonDetection:
    """When YOLO misses, pose landmarks can still mark a subject as present."""
    visible = [lm for lm in pose.landmarks if lm.visibility >= 0.3]
    if len(visible) < 4:
        return _empty_person()
    xs = [lm.x for lm in visible]
    ys = [lm.y for lm in visible]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    h, w = shape[:2]
    px = (int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h))
    return PersonDetection(
        detected=True,
        bbox_px=px,
        bbox_norm=bbox,
        center_norm=(cx, cy),
        confidence=0.0,
        status_text="Baby/Person detected",
        error_message="Presence inferred from pose (YOLO did not return a box)",
    )
