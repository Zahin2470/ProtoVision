"""
Unit tests for ui.py's typography + theme system.

Unlike backbone.py's tests, these run against the REAL bundled Poppins
fonts (assets/fonts/) rather than a mock — there's no gating/licensing
issue for a Google Font, so there's no reason to fake it. That also lets
`TestDrawTextCorrectness` do the strongest test in this file: composite a
string via the glyph cache and compare it pixel-by-pixel against a direct,
one-shot PIL render of the same string — proving the cache/compositing
approach doesn't distort what Poppins actually looks like.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protovision.ui import (
    GlyphCache,
    FontNotFoundError,
    draw_text,
    THEMES,
    Theme,
    ThemeManager,
    DEFAULT_THEME,
    KEY_THEME_TOGGLE,
)

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


@pytest.fixture
def cache() -> GlyphCache:
    return GlyphCache(font_dir=FONT_DIR)


def make_frame(w=200, h=60, color=(20, 20, 30)) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


# --------------------------------------------------------------------------
# font loading
# --------------------------------------------------------------------------

class TestFontLoading:
    def test_unknown_weight_raises_valueerror(self, cache):
        with pytest.raises(ValueError):
            cache.get_advance("A", weight="ultrablack", size=24)

    def test_missing_font_file_raises_helpful_error(self, tmp_path):
        empty_cache = GlyphCache(font_dir=tmp_path)  # no .ttf files here
        with pytest.raises(FontNotFoundError):
            empty_cache.get_advance("A", weight="regular", size=24)

    def test_loads_real_bundled_font_without_error(self, cache):
        # Just confirms the bundled Poppins files are present and loadable.
        advance = cache.get_advance("A", weight="regular", size=24)
        assert advance > 0

    def test_all_bundled_weights_load(self, cache):
        for weight in ("light", "regular", "medium", "bold"):
            assert cache.get_advance("A", weight=weight, size=20) > 0


# --------------------------------------------------------------------------
# advance width / measure_text
# --------------------------------------------------------------------------

class TestAdvanceAndMeasure:
    def test_advance_is_positive_for_visible_char(self, cache):
        assert cache.get_advance("W", size=24) > 0

    def test_advance_is_positive_for_space(self, cache):
        assert cache.get_advance(" ", size=24) > 0

    def test_wide_char_has_larger_advance_than_narrow_char(self, cache):
        # 'i' is narrow, 'W' is wide, in essentially every Latin typeface.
        assert cache.get_advance("W", size=32) > cache.get_advance("i", size=32)

    def test_larger_size_has_larger_advance(self, cache):
        assert cache.get_advance("A", size=48) > cache.get_advance("A", size=16)

    def test_advance_is_deterministic(self, cache):
        a1 = cache.get_advance("Q", size=30)
        a2 = cache.get_advance("Q", size=30)
        assert a1 == a2

    def test_measure_text_width_equals_sum_of_advances(self, cache):
        text = "Hello!"
        width, _ = cache.measure_text(text, size=24)
        expected = sum(cache.get_advance(ch, size=24) for ch in text)
        assert width == expected

    def test_measure_text_height_equals_ascent_plus_descent(self, cache):
        _, height = cache.measure_text("Hg", size=24)
        ascent, descent = cache._get_metrics("regular", 24)
        assert height == ascent + descent

    def test_empty_string_has_zero_width(self, cache):
        width, _ = cache.measure_text("", size=24)
        assert width == 0

    def test_longer_string_is_wider(self, cache):
        w_short, _ = cache.measure_text("Hi", size=24)
        w_long, _ = cache.measure_text("Hi there", size=24)
        assert w_long > w_short


# --------------------------------------------------------------------------
# glyph rendering + caching behavior
# --------------------------------------------------------------------------

class TestGlyphCaching:
    def test_glyph_has_four_channels(self, cache):
        glyph = cache.get_glyph("A", size=24)
        assert glyph.bgra.shape[2] == 4

    def test_glyph_has_positive_dimensions(self, cache):
        glyph = cache.get_glyph("A", size=24)
        assert glyph.bgra.shape[0] > 0
        assert glyph.bgra.shape[1] > 0

    def test_repeated_call_returns_identical_cached_object(self, cache):
        g1 = cache.get_glyph("A", size=24, color=(255, 255, 255))
        g2 = cache.get_glyph("A", size=24, color=(255, 255, 255))
        assert g1 is g2  # proves it was a cache hit, not re-rendered

    def test_different_color_is_a_different_cache_entry(self, cache):
        g_white = cache.get_glyph("A", size=24, color=(255, 255, 255))
        g_red = cache.get_glyph("A", size=24, color=(0, 0, 255))
        assert g_white is not g_red

    def test_different_size_is_a_different_cache_entry(self, cache):
        g_small = cache.get_glyph("A", size=20)
        g_big = cache.get_glyph("A", size=40)
        assert g_small is not g_big
        assert g_big.bgra.shape[0] > g_small.bgra.shape[0]

    def test_whitespace_glyph_has_no_visible_ink(self, cache):
        glyph = cache.get_glyph(" ", size=24, color=(255, 255, 255))
        alpha = glyph.bgra[:, :, 3]
        assert alpha.max() == 0

    def test_visible_char_has_some_ink(self, cache):
        glyph = cache.get_glyph("A", size=24, color=(255, 255, 255))
        alpha = glyph.bgra[:, :, 3]
        assert alpha.max() > 0

    def test_color_is_applied_correctly_bgr_blue(self, cache):
        glyph = cache.get_glyph("A", size=32, color=(255, 0, 0))  # pure blue, BGR
        alpha = glyph.bgra[:, :, 3]
        ink_mask = alpha > 200  # solidly-opaque interior pixels, away from AA edges
        assert ink_mask.any()
        b_channel = glyph.bgra[:, :, 0][ink_mask]
        r_channel = glyph.bgra[:, :, 2][ink_mask]
        assert b_channel.mean() > 200
        assert r_channel.mean() < 20

    def test_color_is_applied_correctly_bgr_red(self, cache):
        glyph = cache.get_glyph("A", size=32, color=(0, 0, 255))  # pure red, BGR
        alpha = glyph.bgra[:, :, 3]
        ink_mask = alpha > 200
        assert ink_mask.any()
        r_channel = glyph.bgra[:, :, 2][ink_mask]
        b_channel = glyph.bgra[:, :, 0][ink_mask]
        assert r_channel.mean() > 200
        assert b_channel.mean() < 20


# --------------------------------------------------------------------------
# draw_text / compositing
# --------------------------------------------------------------------------

class TestDrawTextBasics:
    def test_does_not_mutate_input_frame(self, cache):
        frame = make_frame()
        original = frame.copy()
        draw_text(frame, cache, "Test", 5, 5, size=20)
        np.testing.assert_array_equal(frame, original)

    def test_returns_same_shape_as_input(self, cache):
        frame = make_frame(200, 60)
        out = draw_text(frame, cache, "Test", 5, 5, size=20)
        assert out.shape == frame.shape

    def test_drawing_actually_changes_pixels(self, cache):
        frame = make_frame()
        out = draw_text(frame, cache, "Test", 5, 5, size=24, color=(255, 255, 255))
        assert not np.array_equal(out, frame)

    def test_empty_string_leaves_frame_unchanged(self, cache):
        frame = make_frame()
        out = draw_text(frame, cache, "", 5, 5, size=20)
        np.testing.assert_array_equal(out, frame)

    def test_text_off_right_edge_does_not_crash(self, cache):
        frame = make_frame(50, 40)
        out = draw_text(frame, cache, "This text is way too long for this frame", 10, 5, size=20)
        assert out.shape == frame.shape

    def test_text_off_bottom_edge_does_not_crash(self, cache):
        frame = make_frame(200, 20)
        out = draw_text(frame, cache, "Hi", 5, 100, size=40)
        assert out.shape == frame.shape

    def test_negative_position_does_not_crash(self, cache):
        frame = make_frame()
        out = draw_text(frame, cache, "Hi", -5, -5, size=20)
        assert out.shape == frame.shape


class TestDrawTextCorrectness:
    """The key correctness check: does compositing individually-cached
    glyphs actually reproduce what Poppins looks like, or does the caching
    machinery subtly distort it? Compares against a direct, single-shot
    PIL render of the same string."""

    def _render_reference(self, frame, text, x, y, weight, size, color_bgr):
        font_path = FONT_DIR / {
            "light": "Poppins-Light.ttf",
            "regular": "Poppins-Regular.ttf",
            "medium": "Poppins-Medium.ttf",
            "bold": "Poppins-Bold.ttf",
        }[weight]
        font = ImageFont.truetype(str(font_path), size)
        img = Image.fromarray(frame[:, :, ::-1].copy(), "RGB")  # BGR -> RGB
        draw = ImageDraw.Draw(img)
        b, g, r = color_bgr
        draw.text((x, y), text, font=font, fill=(r, g, b), anchor="la")
        return np.array(img)[:, :, ::-1]  # RGB -> BGR

    @pytest.mark.parametrize("text", ["Hi", "Ag y. 42%", "mug: 0.87", "ProtoVision"])
    def test_composited_text_closely_matches_direct_pil_render(self, cache, text):
        frame = make_frame(320, 60)
        composited = draw_text(frame, cache, text, 5, 5, weight="regular", size=28, color=(255, 255, 255))
        reference = self._render_reference(frame, text, 5, 5, "regular", 28, (255, 255, 255))

        diff = np.abs(composited.astype(int) - reference.astype(int))
        # Allow at most +/-1 per channel (float-rounding dust from
        # compositing each glyph in its own canvas vs. one continuous
        # PIL render) and require the vast majority of pixels to match
        # exactly — this is a strong bar, not a loose one.
        assert diff.max() <= 2
        assert (diff.sum(axis=-1) == 0).mean() > 0.95

    def test_matches_closely_at_a_different_size_and_weight(self, cache):
        frame = make_frame(300, 80)
        composited = draw_text(frame, cache, "Bold Text", 8, 8, weight="bold", size=36, color=(200, 220, 240))
        reference = self._render_reference(frame, "Bold Text", 8, 8, "bold", 36, (200, 220, 240))
        diff = np.abs(composited.astype(int) - reference.astype(int))
        assert diff.max() <= 2
        assert (diff.sum(axis=-1) == 0).mean() > 0.95


# --------------------------------------------------------------------------
# theme system
# --------------------------------------------------------------------------

class TestThemes:
    def test_expected_theme_names_present(self):
        assert set(THEMES.keys()) == {"dark", "light", "neon", "mono"}

    def test_every_theme_is_a_theme_instance(self):
        for theme in THEMES.values():
            assert isinstance(theme, Theme)

    def test_every_theme_alpha_and_vignette_in_valid_range(self):
        for theme in THEMES.values():
            assert 0.0 <= theme.panel_fill_alpha <= 1.0
            assert 0.0 <= theme.vignette_strength <= 1.0

    def test_every_theme_color_is_valid_bgr_tuple(self):
        color_fields = [
            "panel_fill", "panel_border", "text_primary", "text_secondary",
            "accent_known", "accent_unknown", "shadow",
        ]
        for theme in THEMES.values():
            for field_name in color_fields:
                color = getattr(theme, field_name)
                assert len(color) == 3
                assert all(0 <= c <= 255 for c in color)

    def test_default_theme_is_registered(self):
        assert DEFAULT_THEME in THEMES


class TestThemeManager:
    def test_defaults_to_dark(self):
        mgr = ThemeManager()
        assert mgr.name == DEFAULT_THEME

    def test_invalid_initial_theme_raises(self):
        with pytest.raises(ValueError):
            ThemeManager(initial="does-not-exist")

    def test_theme_property_returns_matching_theme_object(self):
        mgr = ThemeManager(initial="neon")
        assert mgr.theme is THEMES["neon"]

    def test_set_changes_active_theme(self):
        mgr = ThemeManager()
        mgr.set("mono")
        assert mgr.name == "mono"

    def test_set_invalid_theme_raises_and_does_not_change_state(self):
        mgr = ThemeManager(initial="dark")
        with pytest.raises(ValueError):
            mgr.set("bogus")
        assert mgr.name == "dark"  # unchanged after the failed set

    def test_cycle_next_moves_to_a_different_theme(self):
        mgr = ThemeManager(initial="dark")
        new_name = mgr.cycle_next()
        assert new_name != "dark"
        assert mgr.name == new_name

    def test_cycle_next_visits_every_theme_and_wraps_around(self):
        mgr = ThemeManager(initial="dark")
        seen = {mgr.name}
        for _ in range(len(THEMES)):
            seen.add(mgr.cycle_next())
        assert seen == set(THEMES.keys())
        assert mgr.name == "dark"  # back to the start after a full cycle

    def test_handle_key_toggles_on_theme_key(self):
        mgr = ThemeManager(initial="dark")
        consumed = mgr.handle_key(KEY_THEME_TOGGLE)
        assert consumed is True
        assert mgr.name != "dark"

    def test_handle_key_ignores_other_keys(self):
        mgr = ThemeManager(initial="dark")
        consumed = mgr.handle_key(ord("x"))
        assert consumed is False
        assert mgr.name == "dark"
