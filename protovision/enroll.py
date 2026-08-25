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
from .prototypes import PrototypeStore
from .ui import GlyphCache, ThemeManager, apply_theme_vignette, draw_glass_panel, draw_text
from .audio import AudioManager

# Key codes — plain ASCII / cv2.waitKey(1) & 0xFF values, portable across platforms.
KEY_CAPTURE = ord(" ")
KEY_UNDO = ord("u")
KEY_FINISH = 13  # Enter/Return
KEY_CANCEL = 27  # Esc

# HUD layout constants (see render_preview) — plain module-level numbers
# rather than magic literals scattered through the method.
_PANEL_X, _PANEL_Y = 20, 20
_PANEL_WIDTH = 280
_PANEL_PAD = 16
_TITLE_SIZE = 20
_PROGRESS_SIZE = 16
_HINT_SIZE = 13
_ROW_GAP = 6


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

    def current_guide_box(self, frame: np.ndarray) -> GuideBox:
        return compute_guide_box(frame.shape[1], frame.shape[0], self.box_fraction)

    def render_preview(self, frame: np.ndarray) -> np.ndarray:
        """
        Full on-screen HUD: guide box, a themed glass panel with the label
        being enrolled, capture progress, and key hints, then a cinematic
        vignette on top of everything.
        """
        box = self.current_guide_box(frame)
        out = draw_guide_box(frame, box)

        theme = self.theme_manager.theme
        cache = self.glyph_cache

        title_h = cache.line_height("medium", _TITLE_SIZE)
        progress_h = cache.line_height("regular", _PROGRESS_SIZE)
        hint_h = cache.line_height("regular", _HINT_SIZE)
        panel_height = (
            _PANEL_PAD * 2 + title_h + _ROW_GAP + progress_h + _ROW_GAP + hint_h + _ROW_GAP + hint_h
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

        return apply_theme_vignette(out, theme)

    def capture_example(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop the current guide box out of `frame`, embed it, and store it.
        Returns the embedding (mainly so callers/tests can inspect it).
        Auto-finishes once `target_examples` is reached.
        """
        if self.state != EnrollState.CAPTURING:
            raise RuntimeError(f"Cannot capture in state {self.state}")

        box = self.current_guide_box(frame)
        crop = crop_guide_box(frame, box)
        embedding = self.backbone.embed(crop)
        self._captured_embeddings.append(embedding)

        if len(self._captured_embeddings) >= self.target_examples:
            self.finish()

        return embedding

    def undo_last(self) -> None:
        """Drop the most recently captured example. No-op if nothing captured
        yet, or if the session already finished/cancelled — undo only makes
        sense mid-capture."""
        if self.state != EnrollState.CAPTURING:
            return
        if self._captured_embeddings:
            self._captured_embeddings.pop()

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
