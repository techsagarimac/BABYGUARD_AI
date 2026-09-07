"""
Alert manager: dashboard message, optional beep, optional voice.

Visual status updates every frame. Beep and speech only fire on a new
alert, and only after a cooldown so the app does not shout every frame.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import config

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


AlertCallback = Optional[Callable[[str], None]]


@dataclass
class AlertEvent:
    kind: str
    title: str
    message: str
    voice_text: str
    is_alert: bool
    fired_audio: bool


class AlertManager:
    """Convert a monitoring decision into user-facing alerts."""

    TITLES = {
        "normal": "Monitoring normally",
        "unusual": "Possible unusual activity",
        "outside": "Subject outside monitoring area",
        "missing": "Subject no longer visible",
    }

    def __init__(self):
        self.voice_enabled = bool(config.VOICE_ALERTS_ENABLED)
        self.beep_enabled = bool(config.BEEP_ENABLED)
        self.cooldown = float(config.ALERT_COOLDOWN_SECONDS)
        self.tts_available = pyttsx3 is not None
        self.tts_error = "" if self.tts_available else "pyttsx3 is not installed"
        self._last_fire: dict[str, float] = {}
        self._last_kind = "normal"
        self._lock = threading.Lock()
        self._speaking = False
        self.on_beep: AlertCallback = None
        self._probe_tts()

    def _probe_tts(self) -> None:
        if pyttsx3 is None:
            self.tts_available = False
            return
        try:
            engine = pyttsx3.init()
            engine.stop()
            del engine
            self.tts_available = True
            self.tts_error = ""
        except Exception as exc:
            self.tts_available = False
            self.tts_error = f"Text-to-speech is unavailable: {exc}"

    def reset(self) -> None:
        self._last_fire.clear()
        self._last_kind = "normal"

    def handle(self, kind: str) -> AlertEvent:
        kind = kind if kind in self.TITLES else "normal"
        is_alert = kind != "normal"
        title = self.TITLES[kind]
        voice_text = config.ALERT_TTS.get(kind, "")
        fired = False

        if is_alert and kind != self._last_kind:
            fired = self._maybe_fire(kind, voice_text)

        self._last_kind = kind
        return AlertEvent(
            kind=kind,
            title=title,
            message=title,
            voice_text=voice_text,
            is_alert=is_alert,
            fired_audio=fired,
        )

    def _maybe_fire(self, kind: str, voice_text: str) -> bool:
        now = time.time()
        last = self._last_fire.get(kind, 0.0)
        if now - last < self.cooldown:
            return False
        self._last_fire[kind] = now

        if self.beep_enabled and self.on_beep:
            try:
                self.on_beep(kind)
            except Exception:
                pass

        if self.voice_enabled and voice_text:
            self._speak(voice_text)
        return True

    def _speak(self, text: str) -> None:
        if not self.tts_available or pyttsx3 is None:
            return
        if self._speaking:
            return

        def worker() -> None:
            with self._lock:
                self._speaking = True
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", config.TTS_RATE)
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                self.tts_error = f"Text-to-speech error: {exc}"
            finally:
                self._speaking = False

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def system_beep() -> None:
        """Best-effort local beep. Tkinter's widget.bell is preferred."""
        try:
            if sys.platform == "darwin":
                import os

                os.system("afplay /System/Library/Sounds/Ping.aiff >/dev/null 2>&1 &")
                return
            if sys.platform.startswith("win"):
                import winsound

                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                return
        except Exception:
            pass
        print("\a", end="", flush=True)
