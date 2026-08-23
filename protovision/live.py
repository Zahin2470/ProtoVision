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
from typing import Dict, Optional

import numpy as np

from .backbone import DinoV3Backbone
from .capture import Camera, compute_guide_box, crop_guide_box, draw_guide_box
from .prototypes import MatchResult, PrototypeStore
from .ui import (
    GlyphCache,
    ThemeManager,
    apply_theme_vignette,
    draw_glass_panel,
    draw_similarity_meter,
    draw_text,
    similarity_meter_height,
)

KEY_QUIT_CODES = (27, ord("q"))  # Esc or 'q'

# HUD layout constants (see render_preview).
_PANEL_X, _PANEL_Y = 20, 20
_PANEL_WIDTH = 240
_PANEL_PAD = 16
_TITLE_SIZE = 20
_STATUS_SIZE = 16
_METER_FONT_SIZE = 13
_METER_BAR_HEIGHT = 14
_METER_ROW_GAP = 10
_ROW_GAP = 10


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
        theme_manager: Optional[ThemeManager] = None,
        glyph_cache: Optional[GlyphCache] = None,
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
        self._last_similarities: Dict[str, float] = {}

        # Visual design system (Phase 2) — defaulted rather than required,
        # so anything constructing a LiveApp without caring about the HUD
        # (most tests) doesn't need to know these exist.
        self.theme_manager = theme_manager or ThemeManager()
        self.glyph_cache = glyph_cache or GlyphCache()

        # The only line in this whole class that touches real hardware.
        self.camera = camera or Camera()

    # -- pure logic (unit tested without a camera) -----------------------

    @property
    def last_result(self) -> Optional[MatchResult]:
        return self._last_result

    @property
    def last_similarities(self) -> Dict[str, float]:
        """Every known class's similarity to the most recent inference —
        what feeds the similarity-meter HUD. Empty until the first frame
        with at least one enrolled class has been processed."""
        return self._last_similarities

    def _should_run_inference(self) -> bool:
        # Always infer on the very first frame (no prior result to hold),
        # then every `frame_skip`-th frame after that.
        return self._last_result is None or self._frame_counter % self.frame_skip == 0

    def process_frame(self, frame: np.ndarray) -> MatchResult:
        """
        Advance the frame counter and return the current best-match result —
        either freshly computed this call, or the held-over result from a
        previous frame, per the frame-skip strategy. Also refreshes
        `last_similarities` (the full per-class picture, not just the
        winner) on the same schedule, so the HUD's similarity meter and the
        headline prediction never disagree about which frame they're from.
        """
        run_inference = self._should_run_inference()

        if run_inference:
            box = compute_guide_box(frame.shape[1], frame.shape[0], self.box_fraction)
            crop = crop_guide_box(frame, box)
            embedding = self.backbone.embed(crop)
            self._last_result = self.store.best_match(
                embedding, threshold=self.threshold, mode=self.match_mode
            )
            self._last_similarities = self.store.all_similarities(embedding, mode=self.match_mode)

        self._frame_counter += 1
        return self._last_result

    def render_preview(self, frame: np.ndarray) -> np.ndarray:
        """
        Full on-screen HUD: guide box, a themed glass panel with the
        current prediction and the similarity-meter signature visual (one
        bar per known class), then a cinematic vignette. Falls back to a
        small status-only panel if nothing's enrolled yet or no inference
        has run yet, rather than drawing an empty meter.
        """
        box = compute_guide_box(frame.shape[1], frame.shape[0], self.box_fraction)
        out = draw_guide_box(frame, box)

        theme = self.theme_manager.theme
        cache = self.glyph_cache

        if self.store.is_empty():
            status = "No classes enrolled yet"
        elif not self._last_similarities:
            status = "Waiting for first frame..."
        else:
            status = None

        if status is not None:
            title_h = cache.line_height("medium", _TITLE_SIZE - 4)
            panel_height = _PANEL_PAD * 2 + title_h
            out = draw_glass_panel(out, _PANEL_X, _PANEL_Y, _PANEL_WIDTH, panel_height, theme, radius=16)
            out = draw_text(
                out, cache, status, _PANEL_X + _PANEL_PAD, _PANEL_Y + _PANEL_PAD,
                weight="medium", size=_TITLE_SIZE - 4, color=theme.text_secondary,
            )
            return apply_theme_vignette(out, theme)

        result = self._last_result
        is_known = bool(result and result.is_known)
        label_text = result.label if is_known else "unknown"
        label_color = theme.accent_known if is_known else theme.accent_unknown

        # Stable alphabetical order rather than sorted-by-score: a ranked
        # order looks nice but causes rows to visibly swap places whenever
        # two close scores cross each other frame to frame (see ui.py's
        # module docstring on draw_similarity_meter's row-order contract).
        meter_entries = sorted(self._last_similarities.items())
        meter_h = similarity_meter_height(
            len(meter_entries), bar_height=_METER_BAR_HEIGHT, row_gap=_METER_ROW_GAP
        )
        title_h = cache.line_height("medium", _TITLE_SIZE)
        panel_height = _PANEL_PAD * 2 + title_h + _ROW_GAP + meter_h

        out = draw_glass_panel(out, _PANEL_X, _PANEL_Y, _PANEL_WIDTH, panel_height, theme, radius=16)

        text_x = _PANEL_X + _PANEL_PAD
        text_y = _PANEL_Y + _PANEL_PAD
        out = draw_text(
            out, cache, label_text, text_x, text_y,
            weight="medium", size=_TITLE_SIZE, color=label_color,
        )

        text_y += title_h + _ROW_GAP
        out = draw_similarity_meter(
            out, text_x, text_y, meter_entries, theme, cache,
            threshold=self.threshold,
            width=_PANEL_WIDTH - _PANEL_PAD * 2,
            bar_height=_METER_BAR_HEIGHT,
            row_gap=_METER_ROW_GAP,
            font_size=_METER_FONT_SIZE,
        )

        return apply_theme_vignette(out, theme)

    def handle_key(self, key: int) -> bool:
        """Theme-toggle handling — factored out so run()'s loop can check
        it before falling through to the quit check. Returns True if the
        key was consumed."""
        return self.theme_manager.handle_key(key)

    @staticmethod
    def is_quit_key(key: int) -> bool:
        return key in KEY_QUIT_CODES

    # -- real camera loop (NOT unit tested — needs actual hardware/display) --

    def run(self) -> None:  # pragma: no cover
        """Blocking loop: show the webcam feed with live predictions overlaid,
        'q'/Esc to quit, 'T' to cycle themes. Not covered by tests — see
        module docstring."""
        import cv2  # local import: only needed for the real, non-testable loop

        window_name = "ProtoVision — Live"
        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    continue
                self.process_frame(frame)
                preview = self.render_preview(frame)
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if self.handle_key(key):
                    continue
                if self.is_quit_key(key):
                    break
        finally:
            self.camera.release()
            cv2.destroyWindow(window_name)
