"""
detect.py — classical, training-free multi-object region proposals.

ProtoVision's core pipeline (a frozen DINOv3 embedding compared against
stored prototypes) never needed a trained object detector — v1
deliberately used a single, user-positioned guide box instead ("no
separate object detector needed for v1", per the original brief).
Recognizing SEVERAL objects in one frame needs some way to find candidate
regions first, but pulling in a full trained detector (YOLO, etc.) would
be a real architecture change and a new heavy dependency, working against
this project's whole "frozen backbone, no training" identity — and there'd
be gated/downloadable weights involved again, the exact thing DINOv3
itself already required extra setup for.

This module proposes regions with plain classical computer vision instead
— Canny edge detection + contours, no learned weights, nothing to
download, nothing to train. Each proposed region is then embedded and
matched exactly like the original single guide-box crop always was;
nothing about backbone.py or prototypes.py had to change.

Honest scope: this works well for this project's actual use case — a
handful of objects on an uncluttered desk or table, roughly the setting
`enroll.py`'s single guide box already assumes. It is not a general
busy-scene detector, has no notion of object CATEGORY (it finds "there's
something here", not "there's a mug here" — that's what DINOv3 is for
afterward), and does no tracking across frames (see live.py's multi-object
mode docstring for what that means for continuity between frames).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .capture import GuideBox

# Plain module constants, not CLI flags — same reasoning as enroll.py's
# quality-warning thresholds: these are classical-CV tuning knobs easiest
# to retune by editing here once tried against a real camera, not
# something that needs its own flag on every command that touches
# detection.
DEFAULT_MIN_AREA_FRACTION = 0.01   # ignore anything smaller than 1% of the frame (likely noise)
DEFAULT_MAX_AREA_FRACTION = 0.5    # ignore anything larger than half the frame (likely background)
DEFAULT_MAX_REGIONS = 6            # cap how many objects get embedded per inference (cost control)
DEFAULT_BLUR_KERNEL = 5
DEFAULT_CANNY_LOW = 50
DEFAULT_CANNY_HIGH = 150
DEFAULT_DILATE_ITERATIONS = 2
DEFAULT_MERGE_OVERLAP_THRESHOLD = 0.5


def _boxes_overlap_fraction(a: GuideBox, b: GuideBox) -> float:
    """Intersection area divided by the SMALLER box's area — used to judge
    whether two candidate boxes are probably fragments of the same
    underlying object (common with edge detection: an object's outline can
    produce more than one contour) rather than two genuinely different
    objects that simply happen to be close together."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    smaller_area = min(a.width * a.height, b.width * b.height)
    return intersection / smaller_area if smaller_area > 0 else 0.0


def _merge_overlapping_boxes(
    boxes: List[GuideBox], overlap_threshold: float = DEFAULT_MERGE_OVERLAP_THRESHOLD
) -> List[GuideBox]:
    """
    Greedy merge: repeatedly keep the largest remaining box and drop any
    other box that overlaps it above `overlap_threshold` (treated as the
    same object), then repeat with what's left. Simple on purpose — this
    is deciding "is this the same blob", not doing real multi-object
    tracking.
    """
    remaining = sorted(boxes, key=lambda b: b.width * b.height, reverse=True)
    kept: List[GuideBox] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [b for b in remaining if _boxes_overlap_fraction(current, b) < overlap_threshold]
    return kept


def propose_regions(
    frame: np.ndarray,
    min_area_fraction: float = DEFAULT_MIN_AREA_FRACTION,
    max_area_fraction: float = DEFAULT_MAX_AREA_FRACTION,
    max_regions: int = DEFAULT_MAX_REGIONS,
    blur_kernel: int = DEFAULT_BLUR_KERNEL,
    canny_low: int = DEFAULT_CANNY_LOW,
    canny_high: int = DEFAULT_CANNY_HIGH,
    dilate_iterations: int = DEFAULT_DILATE_ITERATIONS,
) -> List[GuideBox]:
    """
    Find candidate object regions in `frame` (BGR) via classical edge
    detection. Returns the largest `max_regions` non-overlapping regions,
    sorted largest-first, as GuideBox instances ready for
    `crop_guide_box()` — the exact same type and cropping path
    enroll.py/live.py's single guide box has always used.

    Pipeline: grayscale -> Gaussian blur (suppresses fine texture/noise
    that would otherwise fragment into spurious edges) -> Canny edges ->
    dilate (closes small gaps in an object's outline so it forms one
    closed contour instead of several broken pieces) -> external contours
    -> bounding rectangles -> filtered by area fraction of the frame ->
    overlap-merged -> capped at `max_regions`.

    Returns an empty list for a frame with nothing detected — that's a
    normal, unremarkable result (an empty desk), not an error.
    """
    if frame.ndim != 3:
        raise ValueError(f"Expected an HxWxC frame, got shape {frame.shape}")
    if not (0.0 <= min_area_fraction < max_area_fraction <= 1.0):
        raise ValueError(
            f"Require 0 <= min_area_fraction < max_area_fraction <= 1, "
            f"got {min_area_fraction}, {max_area_fraction}"
        )
    if max_regions < 1:
        raise ValueError(f"max_regions must be >= 1, got {max_regions}")

    h, w = frame.shape[:2]
    frame_area = h * w
    min_area = frame_area * min_area_fraction
    max_area = frame_area * max_area_fraction

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ksize = max(1, blur_kernel | 1)  # Gaussian blur needs an odd kernel size
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    if dilate_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[GuideBox] = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if min_area <= area <= max_area:
            candidates.append(GuideBox(x, y, x + cw, y + ch))

    merged = _merge_overlapping_boxes(candidates)
    merged.sort(key=lambda b: b.width * b.height, reverse=True)
    return merged[:max_regions]
