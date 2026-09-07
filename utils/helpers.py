"""
Shared geometry, drawing, and image-conversion helpers.

Keeping these functions here stops the detectors and the dashboard from
duplicating the same maths.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


def format_hms(ts: Optional[float] = None) -> str:
    if ts is None:
        return datetime.now().strftime("%H:%M:%S")
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_rect(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float]:
    """Order a rectangle and clamp it to the 0–1 image range."""
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    left = clamp(left, 0.0, 1.0)
    right = clamp(right, 0.0, 1.0)
    top = clamp(top, 0.0, 1.0)
    bottom = clamp(bottom, 0.0, 1.0)
    if right - left < 0.04:
        right = clamp(left + 0.04, 0.0, 1.0)
    if bottom - top < 0.04:
        bottom = clamp(top + 0.04, 0.0, 1.0)
    return left, top, right, bottom


def rect_from_points(
    ax: float, ay: float, bx: float, by: float
) -> tuple[float, float, float, float]:
    return normalize_rect(ax, ay, bx, by)


def point_in_rect(px: float, py: float, rect: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_intersection_fraction(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    """How much of `inner` overlaps `outer` (0–1)."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    area = iw * ih
    if area <= 1e-9:
        return 0.0
    cx1 = max(ix1, ox1)
    cy1 = max(iy1, oy1)
    cx2 = min(ix2, ox2)
    cy2 = min(iy2, oy2)
    overlap = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    return overlap / area


def landmark_displacement(
    previous: list[tuple[float, float]],
    current: list[tuple[float, float]],
) -> float:
    """Mean Euclidean distance between matching landmark pairs."""
    if not previous or not current or len(previous) != len(current):
        return 0.0
    total = 0.0
    count = 0
    for (x0, y0), (x1, y1) in zip(previous, current):
        total += math.hypot(x1 - x0, y1 - y0)
        count += 1
    return total / count if count else 0.0


def draw_label(frame, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    """Readable OpenCV text with a dark outline (ASCII only)."""
    if cv2 is None:
        return
    x, y = origin
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def frame_to_photo(frame, max_w: int, max_h: int):
    """
    Convert a BGR OpenCV frame to a Tkinter PhotoImage.

    Returns (photo, scale, offset_x, offset_y, disp_w, disp_h).
    The offset/scale values map widget clicks back onto the original frame.
    """
    import tkinter as tk

    if frame is None:
        photo = tk.PhotoImage(width=max_w, height=max_h)
        return photo, 1.0, 0, 0, max_w, max_h

    height, width = frame.shape[:2]
    scale = min(max_w / max(width, 1), max_h / max(height, 1))
    disp_w = max(1, int(width * scale))
    disp_h = max(1, int(height * scale))
    ox = max(0, (max_w - disp_w) // 2)
    oy = max(0, (max_h - disp_h) // 2)

    if cv2 is not None:
        resized = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        rgb = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)

    if Image is not None and ImageTk is not None:
        image = Image.fromarray(rgb)
        if ox or oy:
            canvas = Image.new("RGB", (max_w, max_h), (7, 11, 20))
            canvas.paste(image, (ox, oy))
            photo = ImageTk.PhotoImage(canvas)
        else:
            photo = ImageTk.PhotoImage(image)
        return photo, scale, ox, oy, disp_w, disp_h

    # Pillow-free fallback: PPM wrapped in a PhotoImage.
    header = f"P6 {disp_w} {disp_h} 255\n".encode()
    photo = tk.PhotoImage(data=header + rgb.tobytes())
    return photo, scale, 0, 0, disp_w, disp_h


def widget_to_norm(
    widget_x: int,
    widget_y: int,
    scale: float,
    offset_x: int,
    offset_y: int,
    frame_w: int,
    frame_h: int,
) -> Optional[tuple[float, float]]:
    """Map a click on the preview widget to normalized frame coordinates."""
    if scale <= 0 or frame_w <= 0 or frame_h <= 0:
        return None
    fx = (widget_x - offset_x) / scale
    fy = (widget_y - offset_y) / scale
    if fx < 0 or fy < 0 or fx > frame_w or fy > frame_h:
        return None
    return clamp(fx / frame_w, 0.0, 1.0), clamp(fy / frame_h, 0.0, 1.0)
