"""
Child-process probe for MediaPipe Pose.

Run:
    python -m detection.pose_probe tasks
    python -m detection.pose_probe solutions

Prints POSE_PROBE_OK and exits 0 if that backend can construct and
run one dummy frame. A C++ abort here kills only this child process.
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


def main() -> int:
    backend = (sys.argv[1] if len(sys.argv) > 1 else "tasks").strip().lower()
    try:
        import numpy as np

        from detection.pose_detector import create_pose_backend

        handle = create_pose_backend(backend)
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        handle.process(dummy)
        handle.close()
    except Exception as exc:
        print(f"POSE_PROBE_FAIL {backend}: {exc}", file=sys.stderr)
        return 1
    print("POSE_PROBE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
