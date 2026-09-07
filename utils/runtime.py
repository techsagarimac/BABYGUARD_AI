"""
Process-level environment tweaks.

MediaPipe's C++ graph can abort the whole Python process on some
macOS / Apple Silicon setups. These variables must be set before
MediaPipe or TensorFlow Lite is imported.
"""

from __future__ import annotations

import os


def apply_safe_runtime() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("GLOG_logtostderr", "1")
    os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # Stops some macOS + native-library abort paths from taking down
    # a parent process when we probe Pose in a child.
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
