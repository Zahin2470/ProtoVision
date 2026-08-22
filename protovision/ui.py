"""
ui.py — ProtoVision's visual design system.

Reuses the proven design language from VisionPuzzle Studio and SignSenseLive
(Poppins typography via PIL with glyph caching, glass-panel HUD, theme
switching, cinematic vignette, ambient audio) but with its own signature look
— this project is a research/analysis tool, not a game or a sign-language
aid, so the visual identity leans into that.

Built incrementally, same as the rest of this project. This slice covers:
  1. Typography — PIL-rendered Poppins glyphs, cached per (weight, size,
     color, character) so a video loop never re-rasterizes text it's
     already drawn once.
  2. Theme palettes — dark/light/neon/mono, with a `T`-key-driven cycle.
  3. Glass-panel HUD — rounded rect, vertical gradient fill, translucent
     alpha, thin border, soft blurred drop shadow — and a cinematic
     radial vignette, both themed off the same Theme dataclass.
  4. Similarity meter — the signature visual: a horizontal bar per known
     class showing its live cosine similarity, so the actual ML decision
     is visible rather than just the winning label. Distinct from both
     prior projects' hand-visualization approaches, and fits this
     project's identity as a data/research tool rather than a game or a
     sign-language aid.

Ambient audio/SFX is a separate, later slice. Panel/vignette/meter
rendering isn't wired into enroll.py/live.py's render_preview() yet — that
assembly happens as its own step once every HUD piece exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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


# ==========================================================================
# Glass-panel HUD
# ==========================================================================
#
# Built the same way text is: render a self-contained BGRA patch with PIL/
# numpy, then composite it with the exact same `_blit_bgra` used for
# glyphs — one tested compositing primitive for the whole HUD, rather than
# a second bespoke blending path just for panels.
#
# A panel is two layers, rendered separately and blitted in this order:
#   1. a blurred, tinted drop shadow (rendered oversized so the blur has
#      room to fall off past the panel's own edges)
#   2. the panel itself: rounded corners, a subtle vertical gradient across
#      the theme's single panel_fill color (lighter top -> darker bottom,
#      for a "glass" sheen without introducing an unrelated color), a
#      translucent alpha from theme.panel_fill_alpha, and a thin border.

def _clamp_channel(value: float) -> int:
    return int(max(0, min(255, round(value))))


def _lighten(color: BGRColor, amount: float) -> BGRColor:
    """Blend `color` toward white by `amount` (0..1)."""
    return tuple(_clamp_channel(c + (255 - c) * amount) for c in color)  # type: ignore[return-value]


def _darken(color: BGRColor, amount: float) -> BGRColor:
    """Blend `color` toward black by `amount` (0..1)."""
    return tuple(_clamp_channel(c * (1 - amount)) for c in color)  # type: ignore[return-value]


def _vertical_gradient(width: int, height: int, top_color: BGRColor, bottom_color: BGRColor) -> np.ndarray:
    """(height, width, 3) uint8 BGR array, linearly interpolated top to bottom."""
    t = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(height, 1, 1)
    top = np.array(top_color, dtype=np.float32).reshape(1, 1, 3)
    bottom = np.array(bottom_color, dtype=np.float32).reshape(1, 1, 3)
    grad = top * (1.0 - t) + bottom * t
    grad = np.broadcast_to(grad, (height, width, 3))
    return grad.astype(np.uint8)


def _rounded_rect_mask(width: int, height: int, radius: int) -> Image.Image:
    """Anti-aliased rounded-rectangle alpha mask (PIL 'L' mode, white=inside).
    Rendered at 4x scale and downsampled — PIL's own rounded_rectangle draw
    is aliased at native resolution, and jagged panel corners would be an
    obvious tell in a "glass" HUD meant to look soft."""
    width, height = max(1, width), max(1, height)
    scale = 4
    big = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(big)
    draw.rounded_rectangle(
        [0, 0, width * scale - 1, height * scale - 1],
        radius=max(0, radius) * scale,
        fill=255,
    )
    return big.resize((width, height), Image.LANCZOS)


def _rounded_rect_border_mask(width: int, height: int, radius: int, border_width: int) -> Image.Image:
    """Alpha mask of just the border ring: outer rounded-rect minus an
    inner one inset by `border_width` on every side."""
    outer = _rounded_rect_mask(width, height, radius)
    if border_width <= 0:
        return Image.new("L", (width, height), 0)

    inner_w = max(1, width - border_width * 2)
    inner_h = max(1, height - border_width * 2)
    inner_radius = max(0, radius - border_width)
    inner_small = _rounded_rect_mask(inner_w, inner_h, inner_radius)

    inner = Image.new("L", (width, height), 0)
    inner.paste(inner_small, (border_width, border_width))

    outer_arr = np.asarray(outer, dtype=np.int16)
    inner_arr = np.asarray(inner, dtype=np.int16)
    ring = np.clip(outer_arr - inner_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(ring, mode="L")


def render_glass_panel(
    width: int,
    height: int,
    theme: Theme,
    radius: int = 16,
    border_width: int = 2,
    gradient_strength: float = 0.18,
) -> np.ndarray:
    """
    Build a self-contained (height, width, 4) BGRA panel: rounded corners,
    a vertical gradient across theme.panel_fill, translucent alpha from
    theme.panel_fill_alpha, and a theme.panel_border stroke. Does NOT
    include the drop shadow — see render_panel_shadow / draw_glass_panel.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"panel width/height must be positive, got {width}x{height}")
    radius = max(0, min(radius, min(width, height) // 2))
    border_width = max(0, border_width)

    mask = np.asarray(_rounded_rect_mask(width, height, radius), dtype=np.float32) / 255.0

    top_color = _lighten(theme.panel_fill, gradient_strength)
    bottom_color = _darken(theme.panel_fill, gradient_strength)
    fill_bgr = _vertical_gradient(width, height, top_color, bottom_color).astype(np.float32)

    alpha = mask * float(theme.panel_fill_alpha)

    if border_width > 0:
        border_alpha = np.asarray(
            _rounded_rect_border_mask(width, height, radius, border_width), dtype=np.float32
        ) / 255.0
        border_bgr = np.empty_like(fill_bgr)
        border_bgr[:] = theme.panel_border
        b = border_alpha[:, :, None]
        fill_bgr = fill_bgr * (1.0 - b) + border_bgr * b
        alpha = np.maximum(alpha, border_alpha)  # border is drawn opaque-ish regardless of panel_fill_alpha

    panel = np.dstack([fill_bgr, alpha * 255.0]).astype(np.uint8)
    return panel


def render_panel_shadow(
    width: int,
    height: int,
    theme: Theme,
    radius: int = 16,
    blur_radius: int = 12,
    offset: Tuple[int, int] = (0, 6),
    strength: float = 0.5,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Build a blurred, theme-tinted drop-shadow patch, sized larger than the
    panel itself so the Gaussian blur has room to fall off softly rather
    than getting cut off at a hard edge.

    Returns (bgra_patch, (dx, dy)) where (dx, dy) is the offset from the
    panel's own top-left corner at which this patch should be blitted —
    already accounting for both the shadow's positional offset and the
    extra blur margin baked into the patch.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"panel width/height must be positive, got {width}x{height}")
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    radius = max(0, min(radius, min(width, height) // 2))
    blur_radius = max(0, blur_radius)

    margin = blur_radius * 2 + 1
    mask = _rounded_rect_mask(width, height, radius)
    canvas = Image.new("L", (width + margin * 2, height + margin * 2), 0)
    canvas.paste(mask, (margin, margin))
    if blur_radius > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur_radius))

    alpha = (np.asarray(canvas, dtype=np.float32) / 255.0) * strength
    bgr = np.empty((alpha.shape[0], alpha.shape[1], 3), dtype=np.float32)
    bgr[:] = theme.shadow

    bgra = np.dstack([bgr, alpha * 255.0]).astype(np.uint8)
    dx, dy = offset
    return bgra, (dx - margin, dy - margin)


def draw_glass_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    theme: Theme,
    radius: int = 16,
    border_width: int = 2,
    shadow_blur: int = 12,
    shadow_offset: Tuple[int, int] = (0, 6),
) -> np.ndarray:
    """
    Draw a themed glass panel (shadow, then fill+border) onto a COPY of
    `frame` at (x, y) = panel top-left. Never mutates the input, same
    convention as draw_guide_box/draw_text.
    """
    out = frame.copy()

    shadow_bgra, (sdx, sdy) = render_panel_shadow(
        width, height, theme, radius=radius, blur_radius=shadow_blur, offset=shadow_offset
    )
    _blit_bgra(out, shadow_bgra, x + sdx, y + sdy)

    panel_bgra = render_glass_panel(width, height, theme, radius=radius, border_width=border_width)
    _blit_bgra(out, panel_bgra, x, y)

    return out


# ==========================================================================
# Cinematic vignette
# ==========================================================================

def apply_vignette(frame: np.ndarray, strength: float) -> np.ndarray:
    """
    Darken `frame`'s edges with a radial falloff from center, strength 0
    (no effect) to 1 (corners driven to black). Returns a new array; never
    mutates the input.

    Distance is normalized so straight edge midpoints reach ~0.71 and the
    four corners reach exactly 1.0 — the classic "corners darker than
    edges" cinematic vignette shape, not a uniform frame border.
    """
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if frame.ndim != 3:
        raise ValueError(f"Expected an HxWxC frame, got shape {frame.shape}")
    if strength == 0.0:
        return frame.copy()

    h, w = frame.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    y_idx, x_idx = np.indices((h, w), dtype=np.float32)

    dist = np.sqrt(((x_idx - cx) / cx) ** 2 + ((y_idx - cy) / cy) ** 2) / np.sqrt(2.0)
    dist = np.clip(dist, 0.0, 1.0)

    darkness = 1.0 - strength * (dist ** 2)
    darkness = darkness[:, :, None]  # broadcast across channels

    out = frame.astype(np.float32) * darkness
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_theme_vignette(frame: np.ndarray, theme: Theme) -> np.ndarray:
    """Convenience wrapper: apply_vignette using the theme's own configured strength."""
    return apply_vignette(frame, theme.vignette_strength)


# ==========================================================================
# Similarity meter — the signature visual
# ==========================================================================
#
# A horizontal bar per known class showing its live cosine similarity to
# the current frame — from prototypes.py's all_similarities(), not just
# best_match()'s single winner. This is the whole point of the design: the
# actual ML decision is visible on screen, not just a final label, which
# is what makes this project read as a research/analysis tool rather than
# a game or an assistive aid (VisionPuzzle's rainbow game skeleton and
# SignSense's neural-scan hand mesh both show *input*; this shows the
# *decision*).
#
# Built from the same primitives as everything else in this file: pill
# shapes reuse _rounded_rect_mask (already built for glass panels), and
# rows are composited with the same _blit_bgra used for text and panels.
#
# Row ORDER is the caller's choice, deliberately: this function draws
# `entries` in the order given rather than silently sorting by score.
# Sorting by similarity descending puts the best match on top, which reads
# well, but it also means two classes with close, fluctuating scores can
# swap rows from frame to frame — visually jittery for a live meter. A
# stable, caller-chosen order (e.g. alphabetical, or "insertion order from
# enrollment") avoids that; ranking by score is still easy to do by sorting
# `entries` before calling this, if that's what a given screen wants.

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _solid_rounded_rect(width: int, height: int, radius: int, color: BGRColor, alpha: float) -> np.ndarray:
    """A flat-color, anti-aliased rounded-rect BGRA patch — the same shape
    machinery as render_glass_panel's fill, minus the gradient/border, used
    here for meter bar tracks and fills (a 'pill' shape when radius ==
    height // 2)."""
    width, height = max(1, width), max(1, height)
    radius = max(0, min(radius, min(width, height) // 2))
    mask = np.asarray(_rounded_rect_mask(width, height, radius), dtype=np.float32) / 255.0
    bgr = np.empty((height, width, 3), dtype=np.float32)
    bgr[:] = color
    a = mask * _clamp01(alpha)
    return np.dstack([bgr, a * 255.0]).astype(np.uint8)


def _truncate_to_width(cache: GlyphCache, text: str, weight: str, size: int, max_width: int) -> str:
    """Shorten `text` with a trailing ellipsis so it fits in `max_width`
    pixels — protects the layout against arbitrary user-entered class
    labels running into the bar rather than assuming short names."""
    width, _ = cache.measure_text(text, weight, size)
    if width <= max_width or not text:
        return text
    ellipsis = "…"
    for cut in range(len(text) - 1, 0, -1):
        candidate = text[:cut] + ellipsis
        w, _ = cache.measure_text(candidate, weight, size)
        if w <= max_width:
            return candidate
    return ellipsis


def similarity_meter_height(num_entries: int, bar_height: int = 14, row_gap: int = 10) -> int:
    """Total pixel height `draw_similarity_meter` will occupy for
    `num_entries` rows — lets a caller (e.g. a future HUD-assembly step
    sizing a glass panel around the meter) know the height up front without
    duplicating the row-spacing arithmetic."""
    if num_entries <= 0:
        return 0
    return num_entries * bar_height + (num_entries - 1) * row_gap


def draw_similarity_meter(
    frame: np.ndarray,
    x: int,
    y: int,
    entries: Sequence[Tuple[str, float]],
    theme: Theme,
    glyph_cache: GlyphCache,
    threshold: float = 0.5,
    width: int = 220,
    bar_height: int = 14,
    row_gap: int = 10,
    label_width: int = 70,
    value_width: int = 44,
    font_size: int = 14,
    font_weight: str = "regular",
    min_similarity: float = -1.0,
    max_similarity: float = 1.0,
    track_alpha: float = 0.35,
    fill_alpha: float = 0.9,
) -> np.ndarray:
    """
    Draw one row per (label, similarity) entry: the class name, a
    horizontal bar-track with a filled bar proportional to similarity, and
    the numeric value — ProtoVision's signature visual.

    Row layout, left to right: [label, fixed label_width] [bar track,
    fills the rest] [value text, fixed value_width]. (x, y) is the
    top-left of the whole stack; rows are drawn top to bottom in the order
    `entries` is given (see the module-level note above on why this
    function doesn't sort them itself).

    Bars for entries at/above `threshold` use theme.accent_known; bars
    below it use theme.accent_unknown — so at a glance you can see not
    just which class is winning, but whether anything is actually
    confident. `similarity` values are expected in cosine-similarity range
    (roughly [-1, 1]; that's the default min/max_similarity mapping), but
    values outside that range are clamped rather than producing a
    negative-width or overflowing bar.

    Returns a COPY of `frame` — never mutates the input, same convention as
    draw_text/draw_glass_panel/draw_guide_box.
    """
    if bar_height <= 0:
        raise ValueError(f"bar_height must be positive, got {bar_height}")
    if width <= label_width + value_width:
        raise ValueError(
            f"width ({width}) must be greater than label_width + value_width "
            f"({label_width + value_width}) or there's no room left for the bar"
        )
    if max_similarity <= min_similarity:
        raise ValueError("max_similarity must be greater than min_similarity")

    out = frame.copy()
    bar_x = x + label_width
    bar_width = width - label_width - value_width
    value_range = max_similarity - min_similarity

    row_y = y
    for label, similarity in entries:
        is_known = similarity >= threshold
        accent = theme.accent_known if is_known else theme.accent_unknown

        display_label = _truncate_to_width(glyph_cache, label, font_weight, font_size, label_width - 4)
        out = draw_text(
            out, glyph_cache, display_label, x, row_y,
            weight=font_weight, size=font_size, color=theme.text_primary,
        )

        track_bgra = _solid_rounded_rect(
            bar_width, bar_height, radius=bar_height // 2, color=theme.text_secondary, alpha=track_alpha
        )
        _blit_bgra(out, track_bgra, bar_x, row_y)

        frac = _clamp01((similarity - min_similarity) / value_range)
        fill_w = int(round(bar_width * frac))
        if fill_w > 0:
            fill_bgra = _solid_rounded_rect(
                fill_w, bar_height, radius=bar_height // 2, color=accent, alpha=fill_alpha
            )
            _blit_bgra(out, fill_bgra, bar_x, row_y)

        value_text = f"{similarity:.2f}"
        out = draw_text(
            out, glyph_cache, value_text, bar_x + bar_width + 6, row_y,
            weight=font_weight, size=font_size, color=theme.text_secondary,
        )

        row_y += bar_height + row_gap

    return out
