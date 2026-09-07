"""
Isolated MediaPipe Pose process.

The GUI process loads YOLO/PyTorch. On some Macs, constructing MediaPipe
in that same process calls abort(). This worker is a fresh Python
interpreter that never imports Ultralytics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["BABYGUARD_POSE_PROBE"] = "1"

from utils.runtime import apply_safe_runtime

apply_safe_runtime()


def run_worker(backend: str, in_queue, out_queue) -> None:
    try:
        from detection.pose_detector import create_pose_backend

        handle = create_pose_backend(backend)
        out_queue.put(("ready", backend))
    except Exception as exc:
        out_queue.put(("fail", str(exc)))
        return

    try:
        while True:
            item = in_queue.get()
            if item is None:
                break
            try:
                result = handle.process(item)
                out_queue.put(("ok", result))
            except Exception as exc:
                out_queue.put(("err", str(exc)))
    finally:
        try:
            handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit("Start this module through PoseDetector, not directly.")
