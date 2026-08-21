"""
ui.py — ProtoVision's visual design system.

Reuses the proven design language from VisionPuzzle Studio and SignSenseLive
(Poppins typography via PIL with glyph caching, glass-panel HUD, theme
switching, cinematic vignette, ambient audio) but with its own signature look
— this project is a research/analysis tool, not a game or a sign-language
aid, so the visual identity leans into that (see the similarity-meter HUD,
still to come).

Built incrementally, same as the rest of this project. This slice covers:
  1. Typography — PIL-rendered Poppins glyphs, cached per (weight, size,
     color, character) so a video loop never re-rasterizes text it's
     already drawn once.
  2. Theme palettes — dark/light/neon/mono, with a `T`-key-driven cycle.

Glass-panel HUD, vignette, the similarity-meter signature visual, and audio
are separate, later slices — this file will grow to hold them too, but
they're not here yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==========================================================================
# Typography — Poppins glyph cache
# ==========================================================================

DEFAULT_FONT_DIR = "assets/fonts"

# Logical weight name -> filename. Matches the static Poppins weights
# actually bundled in assets/fonts/ (Google Fonts' standard release).
FONT_FILES: Dict[str, str] = {
    "light": "Poppins-Light.ttf",
    "regular": "Poppins-Regular.ttf",
    "medium": "Poppins-Medium.ttf",
    "bold": "Poppins-Bold.ttf",
}

BGRColor = Tuple[int, int, int]


class FontNotFoundError(RuntimeError):
    """Raised when a requested font weight's .ttf file isn't on disk."""


@dataclass(frozen=True)
class Glyph:
    """
    One pre-rendered, ready-to-blit character.

    `bgra` is sized (ascent + descent + 2*pad) tall and covers every
    character at a given (weight, size) with a SHARED vertical reference
    (the font's ascender line, at row `pad`) — that's what lets glyphs
    rendered independently of each other still line up correctly on a
    shared baseline when composited side by side in `draw_text`.
    """

    bgra: np.ndarray  # (H, W, 4) uint8, BGRA (matches this project's OpenCV/BGR convention)
    advance: int       # rounded pixel advance — convenience for callers inspecting a single
                        # glyph; draw_text() itself accumulates unrounded advances internally
                        # (see _get_raw_advance) to avoid rounding drift across a long string
    pad: int            # margin baked into bgra's edges; subtract when positioning


class GlyphCache:
    """
    Renders and caches individual character glyphs with PIL. Drawing text
    onto a video frame every tick doesn't re-run PIL's text rasterizer each
    time — only the first time a given (weight, size, color, character)
    combination is needed. Advance widths (needed for layout/measurement
    even before a color is chosen) are cached separately and even more
    cheaply, keyed only by (weight, size, character).

    Known limitation: Poppins is a Latin-script Google Font and doesn't
    include Bengali glyphs. Class labels typed via the CLI are expected to
    be English identifiers (per the project's convention), but if a Bangla
    label is ever entered, PIL will render Poppins' fallback/notdef glyph
    for those characters rather than crashing — worth knowing, not
    silently "wrong."
    """

    def __init__(self, font_dir: "str | Path" = DEFAULT_FONT_DIR, font_files: Optional[Dict[str, str]] = None):
        self.font_dir = Path(font_dir)
        self.font_files = font_files or FONT_FILES
        self._font_objects: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}
        self._metrics: Dict[Tuple[str, int], Tuple[int, int]] = {}
        self._advances: Dict[Tuple[str, int, str], int] = {}
        self._raw_advances: Dict[Tuple[str, int, str], float] = {}
        self._glyphs: Dict[Tuple[str, int, BGRColor, str], Glyph] = {}

    # -- font/metric plumbing -----------------------------------------------

    def _font_path(self, weight: str) -> Path:
        if weight not in self.font_files:
            raise ValueError(f"Unknown font weight {weight!r}, available: {list(self.font_files)}")
        return self.font_dir / self.font_files[weight]

    def _get_font(self, weight: str, size: int) -> ImageFont.FreeTypeFont:
        key = (weight, size)
        if key not in self._font_objects:
            path = self._font_path(weight)
            if not path.exists():
                raise FontNotFoundError(
                    f"Font file not found: '{path}'. Expected '{self.font_files.get(weight, '?')}' "
                    f"in '{self.font_dir}'. See assets/fonts/NOTICE.md."
                )
            self._font_objects[key] = ImageFont.truetype(str(path), size)
        return self._font_objects[key]

    def _get_metrics(self, weight: str, size: int) -> Tuple[int, int]:
        """(ascent, descent) for this (weight, size) — the shared vertical
        reference every glyph at this size is rendered against."""
        key = (weight, size)
        if key not in self._metrics:
            font = self._get_font(weight, size)
            self._metrics[key] = font.getmetrics()
        return self._metrics[key]

    def get_advance(self, char: str, weight: str = "regular", size: int = 24) -> int:
        """Horizontal advance for one character, ROUNDED to a whole pixel —
        cheap, color-independent, and cached separately from full glyph
        rendering so layout/centering code (e.g. the similarity meter,
        later) can measure text without forcing a render.

        For actually positioning a run of characters next to each other,
        `draw_text` uses `_get_raw_advance` (unrounded) instead and rounds
        only once per glyph at blit time — summing pre-rounded integers
        here would let rounding error accumulate across a long string and
        visibly drift the cursor by the end of it.
        """
        key = (weight, size, char)
        if key not in self._advances:
            self._advances[key] = max(1, int(round(self._get_raw_advance(char, weight, size))))
        return self._advances[key]

    def _get_raw_advance(self, char: str, weight: str, size: int) -> float:
        key = (weight, size, char)
        if key not in self._raw_advances:
            font = self._get_font(weight, size)
            self._raw_advances[key] = font.getlength(char)
        return self._raw_advances[key]

    def measure_text(self, text: str, weight: str = "regular", size: int = 24) -> Tuple[int, int]:
        """Total (width, height) `text` would occupy at this weight/size,
        without rendering anything."""
        width = sum(self.get_advance(ch, weight, size) for ch in text)
        ascent, descent = self._get_metrics(weight, size)
        return width, ascent + descent

    # -- glyph rendering -----------------------------------------------

    def get_glyph(self, char: str, weight: str = "regular", size: int = 24, color: BGRColor = (255, 255, 255)) -> Glyph:
        key = (weight, size, color, char)
        if key not in self._glyphs:
            self._glyphs[key] = self._render_glyph(char, weight, size, color)
        return self._glyphs[key]

    def _render_glyph(self, char: str, weight: str, size: int, color: BGRColor) -> Glyph:
        font = self._get_font(weight, size)
        ascent, descent = self._get_metrics(weight, size)
        advance = self.get_advance(char, weight, size)
        bbox = font.getbbox(char, anchor="la")  # (left, top, right, bottom); left is 0 for Poppins in practice

        pad = max(2, size // 10)
        canvas_w = max(advance, bbox[2]) + pad * 2
        canvas_h = ascent + descent + pad * 2

        img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        has_ink = bbox[3] > bbox[1]  # whitespace (e.g. ' ') has zero-height bbox — nothing to draw
        if has_ink:
            draw = ImageDraw.Draw(img)
            b, g, r = color
            draw.text((pad, pad), char, font=font, fill=(r, g, b, 255), anchor="la")

        rgba = np.array(img, dtype=np.uint8)
        bgra = np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])  # RGBA -> BGRA, once, at cache time
        return Glyph(bgra=bgra, advance=advance, pad=pad)


# -- compositing -----------------------------------------------------

def draw_text(
    frame: np.ndarray,
    cache: GlyphCache,
    text: str,
    x: int,
    y: int,
    weight: str = "regular",
    size: int = 24,
    color: BGRColor = (255, 255, 255),
) -> np.ndarray:
    """
    Draw `text` onto a COPY of `frame` (BGR) using cached Poppins glyphs.
    (x, y) is the top-left of the text's ascender line — same convention
    `draw_guide_box` uses for its origin, and like that function, the input
    frame is never mutated.
    """
    out = frame.copy()
    cursor_x = float(x)
    for ch in text:
        glyph = cache.get_glyph(ch, weight, size, color)
        blit_x = int(round(cursor_x)) - glyph.pad
        _blit_bgra(out, glyph.bgra, blit_x, y - glyph.pad)
        cursor_x += cache._get_raw_advance(ch, weight, size)
    return out


def _blit_bgra(frame: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Alpha-composite a BGRA patch onto a BGR frame IN PLACE at (x, y),
    clipped to the frame's bounds (silently drops any part that falls
    outside — text partially off-screen shouldn't crash a live HUD)."""
    fh, fw = frame.shape[:2]
    ph, pw = patch.shape[:2]

    src_x0, src_y0 = 0, 0
    dst_x0, dst_y0 = x, y
    if dst_x0 < 0:
        src_x0 = -dst_x0
        dst_x0 = 0
    if dst_y0 < 0:
        src_y0 = -dst_y0
        dst_y0 = 0
    dst_x1 = min(fw, x + pw)
    dst_y1 = min(fh, y + ph)
    src_x1 = src_x0 + max(0, dst_x1 - dst_x0)
    src_y1 = src_y0 + max(0, dst_y1 - dst_y0)

    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return  # entirely off-frame

    region = frame[dst_y0:dst_y1, dst_x0:dst_x1]
    patch_region = patch[src_y0:src_y1, src_x0:src_x1]

    bgr = patch_region[:, :, :3]
    alpha = patch_region[:, :, 3:4].astype(np.float32) / 255.0

    blended = region.astype(np.float32) * (1.0 - alpha) + bgr.astype(np.float32) * alpha
    region[:] = blended.astype(np.uint8)


# ==========================================================================
# Theme palettes
# ==========================================================================

@dataclass(frozen=True)
class Theme:
    """
    One color palette. All colors are BGR (OpenCV convention). Fields
    anticipate what the glass-panel HUD, vignette, and similarity meter
    will need next — not all of them are consumed by anything yet.
    """

    name: str
    panel_fill: BGRColor          # glass-panel background wash
    panel_fill_alpha: float       # 0..1, panel translucency
    panel_border: BGRColor
    text_primary: BGRColor
    text_secondary: BGRColor
    accent_known: BGRColor        # similarity-meter bar / label color for a confident match
    accent_unknown: BGRColor      # ... for a below-threshold / "unknown" state
    shadow: BGRColor
    vignette_strength: float      # 0..1, how strong the cinematic edge-darkening is


# This project's own identity: cooler, more clinical/"research tool" tones
# than VisionPuzzle's playful rainbow game skeleton or SignSense's
# violet->cyan neural-scan mesh — think lab-instrument HUD, not a game HUD.
THEMES: Dict[str, Theme] = {
    "dark": Theme(
        name="dark",
        panel_fill=(38, 30, 22),         # deep slate-blue-black
        panel_fill_alpha=0.55,
        panel_border=(120, 90, 60),       # muted steel-blue border
        text_primary=(245, 240, 235),
        text_secondary=(190, 180, 170),
        accent_known=(140, 210, 80),      # cool green — "confident match"
        accent_unknown=(90, 90, 220),     # warm red — "unknown / below threshold"
        shadow=(0, 0, 0),
        vignette_strength=0.45,
    ),
    "light": Theme(
        name="light",
        panel_fill=(235, 232, 228),
        panel_fill_alpha=0.65,
        panel_border=(180, 170, 160),
        text_primary=(30, 25, 20),
        text_secondary=(90, 85, 80),
        accent_known=(90, 160, 40),
        accent_unknown=(60, 60, 200),
        shadow=(60, 55, 50),
        vignette_strength=0.15,
    ),
    "neon": Theme(
        name="neon",
        panel_fill=(40, 10, 5),           # near-black with a warm undertone
        panel_fill_alpha=0.5,
        panel_border=(255, 60, 230),       # magenta
        text_primary=(255, 255, 255),
        text_secondary=(255, 210, 130),    # cyan-ish accent text (BGR: high B/G)
        accent_known=(210, 255, 40),        # electric cyan-green
        accent_unknown=(120, 50, 255),      # hot pink-red
        shadow=(0, 0, 0),
        vignette_strength=0.6,
    ),
    "mono": Theme(
        name="mono",
        panel_fill=(35, 35, 35),
        panel_fill_alpha=0.5,
        panel_border=(150, 150, 150),
        text_primary=(235, 235, 235),
        text_secondary=(170, 170, 170),
        accent_known=(220, 220, 220),      # grayscale identity — "known" is bright, not colored
        accent_unknown=(90, 90, 90),        # "unknown" is dim, not colored
        shadow=(0, 0, 0),
        vignette_strength=0.35,
    ),
}

DEFAULT_THEME = "dark"
KEY_THEME_TOGGLE = ord("t")


class ThemeManager:
    """Tracks the active theme and cycles through THEMES on request (bound
    to the `T` key in enroll.py/live.py's key handling, same as the other
    two projects)."""

    def __init__(self, initial: str = DEFAULT_THEME):
        if initial not in THEMES:
            raise ValueError(f"Unknown theme {initial!r}, available: {list(THEMES)}")
        self._name = initial

    @property
    def name(self) -> str:
        return self._name

    @property
    def theme(self) -> Theme:
        return THEMES[self._name]

    def set(self, name: str) -> None:
        if name not in THEMES:
            raise ValueError(f"Unknown theme {name!r}, available: {list(THEMES)}")
        self._name = name

    def cycle_next(self) -> str:
        names = list(THEMES.keys())
        idx = names.index(self._name)
        self._name = names[(idx + 1) % len(names)]
        return self._name

    def handle_key(self, key: int) -> bool:
        """Returns True if the key was consumed (i.e. it was the theme-toggle
        key) — lets callers do `if not theme_mgr.handle_key(key): ...other handling...`."""
        if key == KEY_THEME_TOGGLE:
            self.cycle_next()
            return True
        return False
