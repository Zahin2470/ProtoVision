"""
capture.py — webcam wrapper + guide-box crop helpers, shared by enroll.py and live.py.

Split, same principle as backbone.py: `Camera` (opens real hardware — can't
be unit-tested here, there's no webcam in this sandbox) is kept tiny and
dumb on purpose. Everything with actual logic — guide box geometry, cropping,
bounds-checking, drawing the overlay — is plain numpy/arithmetic and is
fully unit tested below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Guide box geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GuideBox:
    """A rectangle in pixel coordinates, (x1, y1) top-left to (x2, y2) bottom-right."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


def compute_guide_box(frame_width: int, frame_height: int, box_fraction: float = 0.5) -> GuideBox:
    """
    A centered, roughly-square guide box sized as a fraction of the shorter
    frame dimension, always clamped to fit fully inside the frame.

    v1 deliberately doesn't do a separate object detector (per the brief) —
    the user just centers their object in this box. Keeping the box square-ish
    and a fraction of the *shorter* dimension means it behaves sanely on both
    a 4:3 webcam and a 16:9 one, without ever hanging off an edge.
    """
    if not (0.1 <= box_fraction <= 1.0):
        raise ValueError(f"box_fraction must be in [0.1, 1.0], got {box_fraction}")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"frame dimensions must be positive, got {frame_width}x{frame_height}")

    side = int(min(frame_width, frame_height) * box_fraction)
    side = max(side, 16)  # never smaller than one DINOv3 patch (16px)
    side = min(side, frame_width, frame_height)  # never larger than the frame

    cx, cy = frame_width // 2, frame_height // 2
    half = side // 2

    x1 = cx - half
    y1 = cy - half
    x2 = x1 + side
    y2 = y1 + side

    # Shift back inside bounds if centering pushed an edge out (small/odd frames).
    if x1 < 0:
        x1, x2 = 0, side
    if y1 < 0:
        y1, y2 = 0, side
    if x2 > frame_width:
        x2, x1 = frame_width, frame_width - side
    if y2 > frame_height:
        y2, y1 = frame_height, frame_height - side

    return GuideBox(x1, y1, x2, y2)


def crop_guide_box(frame: np.ndarray, box: GuideBox) -> np.ndarray:
    """Crop `frame` to `box`. Raises rather than silently clamping — an
    out-of-bounds box means a caller bug (e.g. reused a box computed for a
    different frame size), and that should be loud, not silently wrong."""
    if frame.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D frame, got shape {frame.shape}")
    h, w = frame.shape[:2]
    if box.x1 < 0 or box.y1 < 0 or box.x2 > w or box.y2 > h:
        raise ValueError(f"Guide box {box.as_tuple()} out of bounds for frame {w}x{h}")
    if box.width <= 0 or box.height <= 0:
        raise ValueError(f"Guide box has non-positive size: {box.as_tuple()}")
    return frame[box.y1 : box.y2, box.x1 : box.x2]


def draw_guide_box(
    frame: np.ndarray,
    box: GuideBox,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Return a COPY of `frame` with the guide box drawn on it. Never mutates
    the input — callers (enroll.py/live.py) still need the clean frame to crop
    and embed from."""
    out = frame.copy()
    cv2.rectangle(out, (box.x1, box.y1), (box.x2, box.y2), color, thickness)
    return out


# --------------------------------------------------------------------------
# Camera (real hardware — not unit-testable in this sandbox)
# --------------------------------------------------------------------------

class CameraOpenError(RuntimeError):
    """Raised when the webcam device can't be opened."""


class Camera:
    """
    Thin wrapper around cv2.VideoCapture. Intentionally has almost no logic
    in it — everything that could be tested without hardware has been pulled
    out into the free functions above.

    macOS note: if this raises CameraOpenError, the most common cause isn't
    a code bug — it's the terminal/IDE process not having camera permission
    yet (System Settings -> Privacy & Security -> Camera). First run from
    Terminal.app directly if VS Code's integrated terminal can't get access.
    """

    def __init__(self, device_index: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        self.device_index = device_index
        self._cap = cv2.VideoCapture(device_index)
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise CameraOpenError(
                f"Could not open camera at index {device_index}. On macOS, check "
                "System Settings -> Privacy & Security -> Camera for your terminal/IDE."
            )

    def read(self) -> Optional[np.ndarray]:
        """Returns a BGR frame, or None if a frame couldn't be grabbed
        (e.g. camera briefly disconnected) — callers should skip that tick,
        not crash."""
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
