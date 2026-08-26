"""
enroll.py — camera app: capture N example images of one object, embed each
with the frozen DINOv3 backbone, and add them to the class's prototype.

Testing approach
-----------------
`EnrollApp.__init__` opens a real camera (`Camera()`), which doesn't exist in
this sandbox. Every other method is pure state-machine logic — key handling,
capture bookkeeping, finish/cancel/undo — and is unit tested by constructing
the instance via `EnrollApp.__new__(EnrollApp)` (bypassing `__init__`) and
setting the handful of attributes those methods actually need, the same
pattern used for SignSense's PracticeApp/LiveApp.

macOS note: label input is a CLI argument (see main.py, step 4), not a
tkinter/native dialog popup — sidesteps the Tcl-Tk crash entirely rather than
working around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .backbone import DinoV3Backbone
from .capture import Camera, GuideBox, compute_guide_box, crop_guide_box, draw_guide_box
from .prototypes import PrototypeStore, cosine_similarity
from .ui import GlyphCache, ThemeManager, apply_theme_vignette, draw_glass_panel, draw_text, truncate_to_width
from .audio import AudioManager

# Key codes — plain ASCII / cv2.waitKey(1) & 0xFF values, portable across platforms.
KEY_CAPTURE = ord(" ")
KEY_UNDO = ord("u")
KEY_FINISH = 13  # Enter/Return
KEY_CANCEL = 27  # Esc

# Phase 3 prototype-quality warnings (advisory only — capture still
# succeeds either way). Plain module constants, deliberately, since these
# are exactly the kind of thing that needs recalibrating once real DINOv3
# similarity distributions can actually be observed on real photos rather
# than guessed at — see docs/DINOV3_SETUP.md and the design-decision note
# in README.md.
QUALITY_DUPLICATE_THRESHOLD = 0.97  # vs. an earlier capture THIS session — "not enough variety"
QUALITY_CONFUSION_THRESHOLD = 0.75  # vs. a DIFFERENT existing class's prototype — "may get confused"

# HUD layout constants (see render_preview) — plain module-level numbers
# rather than magic literals scattered through the method.
_PANEL_X, _PANEL_Y = 20, 20
_PANEL_WIDTH = 280
_PANEL_PAD = 16
_TITLE_SIZE = 20
_PROGRESS_SIZE = 16
_HINT_SIZE = 13
_WARNING_SIZE = 12
_ROW_GAP = 6
_MAX_WARNING_LINES = 2  # bounds panel height even if multiple things look off at once


class EnrollState(Enum):
    CAPTURING = auto()
    DONE = auto()
    CANCELLED = auto()


class NotEnoughExamplesError(ValueError):
    """Raised if `finish()` is called before `min_examples` have been captured."""


class EnrollApp:
    """
    State for one enrollment session: capturing `target_examples` crops for
    a single class label, embedding each, then committing them to the
    PrototypeStore and saving to disk.
    """

    def __init__(
        self,
        label: str,
        backbone: DinoV3Backbone,
        store: PrototypeStore,
        store_path: "str | Path",
        target_examples: int = 8,
        min_examples: int = 5,
        box_fraction: float = 0.5,
        camera: Optional[Camera] = None,
        theme_manager: Optional[ThemeManager] = None,
        glyph_cache: Optional[GlyphCache] = None,
        audio: Optional[AudioManager] = None,
    ):
        if not label or not label.strip():
            raise ValueError("label must be a non-empty string")
        if min_examples < 1:
            raise ValueError("min_examples must be >= 1")
        if target_examples < min_examples:
            raise ValueError("target_examples must be >= min_examples")

        self.label = label.strip()
        self.backbone = backbone
        self.store = store
        self.store_path = Path(store_path)
        self.target_examples = target_examples
        self.min_examples = min_examples
        self.box_fraction = box_fraction

        self._captured_embeddings: List[np.ndarray] = []
        self._last_capture_warnings: List[str] = []
        self.state = EnrollState.CAPTURING

        # Visual design system (Phase 2) — defaulted rather than required,
        # so anything constructing an EnrollApp without caring about the
        # HUD (most tests) doesn't need to know these exist.
        self.theme_manager = theme_manager or ThemeManager()
        self.glyph_cache = glyph_cache or GlyphCache()

        # Audio is fail-soft by construction (see audio.py) — always safe
        # to default-construct even with no working audio device.
        self.audio = audio or AudioManager()

        # The only line in this whole class that touches real hardware.
        self.camera = camera or Camera()

    # -- pure logic (unit tested without a camera) -----------------------

    @property
    def progress(self) -> Tuple[int, int]:
        """(captured_so_far, target) — e.g. for an on-screen '3 / 8' counter."""
        return (len(self._captured_embeddings), self.target_examples)

    @property
    def has_min_examples(self) -> bool:
        return len(self._captured_embeddings) >= self.min_examples

    @property
    def last_capture_warnings(self) -> List[str]:
        """Human-readable prototype-quality warnings for the MOST RECENT
        capture — empty if nothing looked off. Purely advisory (see
        _check_capture_quality); capture always succeeds regardless.
        Cleared by undo_last(), since the capture the warnings were about
        no longer exists once undone."""
        return self._last_capture_warnings

    def _check_capture_quality(self, embedding: np.ndarray) -> List[str]:
        """
        Advisory checks on a just-captured embedding, run BEFORE it's added
        to `_captured_embeddings` (so "previous captures" below genuinely
        excludes it): Phase 3's prototype-quality enrichment.

          1. Too similar to an EARLIER capture in this same session — a
             sign the user didn't actually change the angle/distance/
             lighting between captures, so the class ends up with less
             real variety than `target_examples` suggests.
          2. Too similar to a DIFFERENT class's existing prototype — a
             sign this object might get confused with something already
             enrolled, before that confusable prototype ever gets saved.

        Returns a list of short warning strings (possibly empty). Never
        raises and never blocks the capture — these are warnings, not
        validation errors; the brief asks to "warn", not "reject".
        """
        warnings: List[str] = []

        for i, previous in enumerate(self._captured_embeddings):
            if cosine_similarity(embedding, previous) >= QUALITY_DUPLICATE_THRESHOLD:
                warnings.append(f"Like capture #{i + 1} — try a new angle")
                break  # one duplicate warning is enough; don't spam per near-duplicate

        other_label, other_sim = self.store.closest_other_class(self.label, embedding)
        if other_label is not None and other_sim >= QUALITY_CONFUSION_THRESHOLD:
            warnings.append(f"May be confused with '{other_label}'")

        return warnings

    def current_guide_box(self, frame: np.ndarray) -> GuideBox:
        return compute_guide_box(frame.shape[1], frame.shape[0], self.box_fraction)

    def render_preview(self, frame: np.ndarray) -> np.ndarray:
        """
        Full on-screen HUD: guide box, a themed glass panel with the label
        being enrolled, capture progress, key hints, and (Phase 3) up to
        two prototype-quality warnings for the most recent capture, then a
        cinematic vignette on top of everything.
        """
        box = self.current_guide_box(frame)
        out = draw_guide_box(frame, box)

        theme = self.theme_manager.theme
        cache = self.glyph_cache

        warning_lines = self._last_capture_warnings[:_MAX_WARNING_LINES]
        warning_h = cache.line_height("regular", _WARNING_SIZE) if warning_lines else 0

        title_h = cache.line_height("medium", _TITLE_SIZE)
        progress_h = cache.line_height("regular", _PROGRESS_SIZE)
        hint_h = cache.line_height("regular", _HINT_SIZE)
        panel_height = (
            _PANEL_PAD * 2 + title_h + _ROW_GAP + progress_h + _ROW_GAP + hint_h + _ROW_GAP + hint_h
            + len(warning_lines) * (_ROW_GAP + warning_h)
        )

        out = draw_glass_panel(out, _PANEL_X, _PANEL_Y, _PANEL_WIDTH, panel_height, theme, radius=16)

        text_x = _PANEL_X + _PANEL_PAD
        text_y = _PANEL_Y + _PANEL_PAD
        out = draw_text(
            out, cache, f"Enroll: {self.label}", text_x, text_y,
            weight="medium", size=_TITLE_SIZE, color=theme.text_primary,
        )

        text_y += title_h + _ROW_GAP
        captured, target = self.progress
        progress_color = theme.accent_known if self.has_min_examples else theme.text_secondary
        out = draw_text(
            out, cache, f"{captured} / {target} captured", text_x, text_y,
            weight="regular", size=_PROGRESS_SIZE, color=progress_color,
        )

        # Two lines, not one — the full hint string doesn't fit a
        # reasonably-sized panel on one line (measured ~376px at this font
        # size vs. this panel's ~248px of usable width).
        text_y += progress_h + _ROW_GAP
        out = draw_text(
            out, cache, "SPACE capture   U undo   ENTER finish", text_x, text_y,
            weight="regular", size=_HINT_SIZE, color=theme.text_secondary,
        )
        text_y += hint_h + _ROW_GAP
        out = draw_text(
            out, cache, "ESC cancel   T theme", text_x, text_y,
            weight="regular", size=_HINT_SIZE, color=theme.text_secondary,
        )

        # Advisory quality warnings for the most recent capture — shown in
        # the "unknown"/caution accent so they read as a heads-up, not an
        # error, and capped at _MAX_WARNING_LINES so a pathological case
        # (matches several other classes AND duplicates an earlier capture)
        # can't blow up the panel to an unbounded height.
        prev_row_h = hint_h  # the last-drawn row was the second hint line
        for warning in warning_lines:
            text_y += prev_row_h + _ROW_GAP
            display_warning = truncate_to_width(cache, warning, "regular", _WARNING_SIZE, _PANEL_WIDTH - _PANEL_PAD * 2)
            out = draw_text(
                out, cache, display_warning, text_x, text_y,
                weight="regular", size=_WARNING_SIZE, color=theme.accent_unknown,
            )
            prev_row_h = warning_h

        return apply_theme_vignette(out, theme)

    def capture_example(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop the current guide box out of `frame`, embed it, check it for
        quality warnings (too similar to a previous capture, or to a
        different existing class), then store it. Returns the embedding
        (mainly so callers/tests can inspect it). Auto-finishes once
        `target_examples` is reached.
        """
        if self.state != EnrollState.CAPTURING:
            raise RuntimeError(f"Cannot capture in state {self.state}")

        box = self.current_guide_box(frame)
        crop = crop_guide_box(frame, box)
        embedding = self.backbone.embed(crop)

        # Checked BEFORE appending, so "previous captures" genuinely means
        # captures before this one, not including it.
        self._last_capture_warnings = self._check_capture_quality(embedding)
        self._captured_embeddings.append(embedding)

        if len(self._captured_embeddings) >= self.target_examples:
            self.finish()

        return embedding

    def undo_last(self) -> None:
        """Drop the most recently captured example, and the quality
        warnings that were about it (they no longer describe anything that
        exists). No-op if nothing captured yet, or if the session already
        finished/cancelled — undo only makes sense mid-capture."""
        if self.state != EnrollState.CAPTURING:
            return
        if self._captured_embeddings:
            self._captured_embeddings.pop()
            self._last_capture_warnings = []

    def finish(self) -> None:
        """Commit captured examples to the store and save to disk, then play
        the enroll_success chime. Requires at least `min_examples` — raises
        rather than silently saving a prototype built from too few/noisy
        examples (and rather than playing a "success" sound for a failure)."""
        if self.state != EnrollState.CAPTURING:
            raise RuntimeError(f"Cannot finish in state {self.state}")
        if not self.has_min_examples:
            raise NotEnoughExamplesError(
                f"Only {len(self._captured_embeddings)} example(s) captured, "
                f"need at least {self.min_examples}."
            )
        self.store.add_examples(self.label, self._captured_embeddings)
        self.store.save(self.store_path)
        self.state = EnrollState.DONE
        self.audio.play_enroll_success()

    def cancel(self) -> None:
        """Abort without saving anything to the store."""
        self.state = EnrollState.CANCELLED

    def handle_key(self, key: int, frame: np.ndarray) -> None:
        """
        Dispatch a raw key code (as returned by `cv2.waitKey(1) & 0xFF`) to
        the right action. Kept as a single small dispatcher so the real
        camera loop in `run()` stays a thin, untestable shell around this
        testable method.

        Theme cycling (`T`) is checked first and works regardless of
        capture state — there's no reason switching themes should be
        blocked just because the session already finished or was cancelled.
        """
        if self.theme_manager.handle_key(key):
            return
        if self.state != EnrollState.CAPTURING:
            return
        if key == KEY_CAPTURE:
            self.capture_example(frame)
        elif key == KEY_UNDO:
            self.undo_last()
        elif key == KEY_FINISH:
            if self.has_min_examples:
                self.finish()
            # else: silently ignored — UI should be showing "need N more"
        elif key == KEY_CANCEL:
            self.cancel()

    # -- real camera loop (NOT unit tested — needs actual hardware/display) --

    def run(self) -> EnrollState:  # pragma: no cover
        """
        Blocking loop: show the webcam feed with the guide box overlay,
        SPACE to capture, 'u' to undo, Enter to finish early (if enough
        examples), Esc to cancel, 'T' to cycle themes. Not covered by
        tests — see module docstring.
        """
        import cv2  # local import: only needed for the real, non-testable loop

        window_name = f"ProtoVision — Enroll '{self.label}'"
        try:
            while self.state == EnrollState.CAPTURING:
                frame = self.camera.read()
                if frame is None:
                    continue
                preview = self.render_preview(frame)
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                self.handle_key(key, frame)
        finally:
            self.camera.release()
            cv2.destroyWindow(window_name)
        return self.state
