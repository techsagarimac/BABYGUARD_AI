#!/usr/bin/env python3
"""
BabyGuard AI
Camera-Based Intelligent Baby Monitoring and Alert System

Run:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime import apply_safe_runtime

apply_safe_runtime()


def main() -> None:
    if sys.version_info < (3, 10):
        print("BabyGuard AI needs Python 3.10 or newer.")
        raise SystemExit(1)

    if sys.version_info >= (3, 13):
        print("Note: MediaPipe and YOLO are most reliable on Python 3.10–3.12.")
        print("If installation fails, create the venv with python3.11 or python3.12.")
        print()

    try:
        from ui.dashboard import BabyGuardApp
    except Exception as exc:
        print("BabyGuard AI could not start.")
        print(f"Reason: {exc}")
        print("Install dependencies with:  pip install -r requirements.txt")
        raise SystemExit(1) from exc

    app = BabyGuardApp()
    app.run()


if __name__ == "__main__":
    main()
