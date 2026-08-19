"""
live.py — camera app: live recognition against stored prototypes.

Testing approach: same as enroll.py — `__init__` opens a real camera and is
not testable here; `process_frame()` (the actual decision logic, including
the frame-skip strategy) is pure and is fully unit tested by constructing
the instance via `LiveApp.__new__(LiveApp)`.

Frame-skip rationale (from the brief's CPU judgment call): the guide-box
workflow means the user holds an object steadily in the box — unlike
SignSense's hand tracking, it doesn't need a fresh inference every single
frame. Running DINOv3 every `frame_skip`-th frame and holding the last
result in between keeps the UI responsive even if raw inference is slow,
at the cost of the prediction lagging by up to `frame_skip` frames when the
object actually changes. `frame_skip=1` disables skipping entirely (infer
every frame) — useful once real latency is measured and turns out fast
enough not to need this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .backbone import DinoV3Backbone
from .capture import Camera, compute_guide_box, crop_guide_box, draw_guide_box
from .prototypes import MatchResult, PrototypeStore

KEY_QUIT_CODES = (27, ord("q"))  # Esc or 'q'


class LiveApp:
    def __init__(
        self,
        backbone: DinoV3Backbone,
        store: PrototypeStore,
        threshold: float = 0.5,
        match_mode: str = "mean",
        frame_skip: int = 5,
        box_fraction: float = 0.5,
        camera: Optional[Camera] = None,
    ):
        if match_mode not in ("mean", "max"):
            raise ValueError(f"match_mode must be 'mean' or 'max', got {match_mode!r}")
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")

        self.backbone = backbone
        self.store = store
        self.threshold = threshold
        self.match_mode = match_mode
        self.frame_skip = frame_skip
        self.box_fraction = box_fraction

        self._frame_counter = 0
        self._last_result: Optional[MatchResult] = None

        # The only line in this whole class that touches real hardware.
        self.camera = camera or Camera()

    # -- pure logic (unit tested without a camera) -----------------------

    @property
    def last_result(self) -> Optional[MatchResult]:
        return self._last_result

    def _should_run_inference(self) -> bool:
        # Always infer on the very first frame (no prior result to hold),
        # then every `frame_skip`-th frame after that.
        return self._last_result is None or self._frame_counter % self.frame_skip == 0

    def process_frame(self, frame: np.ndarray) -> MatchResult:
        """
        Advance the frame counter and return the current best-match result —
        either freshly computed this call, or the held-over result from a
        previous frame, per the frame-skip strategy.
        """
        run_inference = self._should_run_inference()

        if run_inference:
            box = compute_guide_box(frame.shape[1], frame.shape[0], self.box_fraction)
            crop = crop_guide_box(frame, box)
            embedding = self.backbone.embed(crop)
            self._last_result = self.store.best_match(
                embedding, threshold=self.threshold, mode=self.match_mode
            )

        self._frame_counter += 1
        return self._last_result

    def render_preview(self, frame: np.ndarray) -> np.ndarray:
        """Frame with the guide box overlay drawn (label/similarity text is
        added by ui.py in Phase 2 — this just gives the box for now)."""
        box = compute_guide_box(frame.shape[1], frame.shape[0], self.box_fraction)
        return draw_guide_box(frame, box)

    @staticmethod
    def is_quit_key(key: int) -> bool:
        return key in KEY_QUIT_CODES

    # -- real camera loop (NOT unit tested — needs actual hardware/display) --

    def run(self) -> None:  # pragma: no cover
        """Blocking loop: show the webcam feed with live predictions overlaid,
        'q'/Esc to quit. Not covered by tests — see module docstring."""
        import cv2  # local import: only needed for the real, non-testable loop

        window_name = "ProtoVision — Live"
        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    continue
                result = self.process_frame(frame)
                preview = self.render_preview(frame)
                label_text = result.label if result.is_known else "unknown"
                cv2.putText(
                    preview,
                    f"{label_text} ({result.similarity:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if result.is_known else (0, 0, 255),
                    2,
                )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if self.is_quit_key(key):
                    break
        finally:
            self.camera.release()
            cv2.destroyWindow(window_name)
