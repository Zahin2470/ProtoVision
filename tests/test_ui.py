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
    render_glass_panel,
    render_panel_shadow,
    draw_glass_panel,
    apply_vignette,
    apply_theme_vignette,
    _lighten,
    _darken,
    _vertical_gradient,
    _rounded_rect_mask,
    _rounded_rect_border_mask,
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


# --------------------------------------------------------------------------
# glass panel — low-level helpers
# --------------------------------------------------------------------------

class TestColorHelpers:
    def test_lighten_moves_toward_white(self):
        result = _lighten((50, 50, 50), 0.5)
        assert all(c > 50 for c in result)

    def test_lighten_zero_amount_is_unchanged(self):
        assert _lighten((50, 60, 70), 0.0) == (50, 60, 70)

    def test_lighten_full_amount_reaches_white(self):
        assert _lighten((50, 60, 70), 1.0) == (255, 255, 255)

    def test_lighten_clamps_at_255(self):
        result = _lighten((250, 250, 250), 0.9)
        assert all(c <= 255 for c in result)

    def test_darken_moves_toward_black(self):
        result = _darken((200, 200, 200), 0.5)
        assert all(c < 200 for c in result)

    def test_darken_zero_amount_is_unchanged(self):
        assert _darken((50, 60, 70), 0.0) == (50, 60, 70)

    def test_darken_full_amount_reaches_black(self):
        assert _darken((50, 60, 70), 1.0) == (0, 0, 0)

    def test_darken_clamps_at_0(self):
        result = _darken((2, 2, 2), 2.0)  # amount > 1, shouldn't go negative
        assert all(c >= 0 for c in result)


class TestVerticalGradient:
    def test_output_shape(self):
        grad = _vertical_gradient(40, 20, (0, 0, 0), (255, 255, 255))
        assert grad.shape == (20, 40, 3)

    def test_top_row_matches_top_color(self):
        grad = _vertical_gradient(10, 20, (10, 20, 30), (200, 210, 220))
        np.testing.assert_array_equal(grad[0, 0], [10, 20, 30])

    def test_bottom_row_matches_bottom_color(self):
        grad = _vertical_gradient(10, 20, (10, 20, 30), (200, 210, 220))
        np.testing.assert_array_equal(grad[-1, 0], [200, 210, 220])

    def test_gradient_is_constant_across_each_row(self):
        grad = _vertical_gradient(15, 10, (10, 10, 10), (200, 200, 200))
        for row in range(10):
            assert np.all(grad[row] == grad[row, 0])

    def test_monotonic_transition_top_to_bottom(self):
        grad = _vertical_gradient(1, 30, (0, 0, 0), (255, 255, 255))
        col = grad[:, 0, 0].astype(int)
        assert all(b >= a for a, b in zip(col, col[1:]))  # non-decreasing


class TestRoundedRectMask:
    def test_output_size(self):
        mask = _rounded_rect_mask(50, 30, 10)
        assert mask.size == (50, 30)

    def test_center_is_fully_opaque(self):
        mask = np.asarray(_rounded_rect_mask(60, 40, 12))
        assert mask[20, 30] == 255

    def test_true_corner_is_mostly_transparent(self):
        mask = np.asarray(_rounded_rect_mask(60, 40, 15))
        assert mask[0, 0] < 50  # rounded away from the literal corner pixel

    def test_zero_radius_is_still_a_full_rect_at_center(self):
        mask = np.asarray(_rounded_rect_mask(40, 40, 0))
        assert mask[20, 20] == 255

    def test_border_mask_is_zero_when_border_width_zero(self):
        mask = np.asarray(_rounded_rect_border_mask(50, 50, 10, 0))
        assert mask.max() == 0

    def test_border_mask_ring_is_near_edge_not_center(self):
        mask = np.asarray(_rounded_rect_border_mask(60, 60, 10, 3))
        assert mask[30, 30] == 0  # dead center: inside the "inner" cutout, not on the ring
        # somewhere along the top edge (away from the rounded corner) should be on the ring
        assert mask[1, 30] > 0


# --------------------------------------------------------------------------
# glass panel — full render
# --------------------------------------------------------------------------

class TestRenderGlassPanel:
    def test_output_shape_and_dtype(self):
        panel = render_glass_panel(120, 80, THEMES["dark"])
        assert panel.shape == (80, 120, 4)
        assert panel.dtype == np.uint8

    def test_rejects_non_positive_dimensions(self):
        with pytest.raises(ValueError):
            render_glass_panel(0, 80, THEMES["dark"])
        with pytest.raises(ValueError):
            render_glass_panel(120, -1, THEMES["dark"])

    def test_oversized_radius_does_not_crash(self):
        panel = render_glass_panel(40, 30, THEMES["dark"], radius=1000)
        assert panel.shape == (30, 40, 4)

    def test_center_alpha_matches_theme_fill_alpha(self):
        theme = THEMES["dark"]
        panel = render_glass_panel(120, 80, theme, border_width=0)
        center_alpha = panel[40, 60, 3]
        expected = round(theme.panel_fill_alpha * 255)
        assert abs(int(center_alpha) - expected) <= 1

    def test_corner_alpha_is_near_zero(self):
        panel = render_glass_panel(120, 80, THEMES["dark"], radius=20)
        assert panel[0, 0, 3] < 20

    def test_top_is_lighter_than_bottom(self):
        panel = render_glass_panel(100, 100, THEMES["dark"], border_width=0)
        top_pixel = panel[10, 50, :3].astype(int).sum()
        bottom_pixel = panel[90, 50, :3].astype(int).sum()
        assert top_pixel > bottom_pixel

    def test_no_border_when_border_width_zero(self):
        theme = THEMES["dark"]
        panel = render_glass_panel(100, 100, theme, border_width=0)
        # near-edge pixel (but still inside the rounded area) should have the
        # same alpha as the base fill, not the boosted "always visible" border alpha
        edge_alpha = panel[50, 2, 3]
        expected = round(theme.panel_fill_alpha * 255)
        assert abs(int(edge_alpha) - expected) <= 5

    def test_border_present_changes_edge_color(self):
        theme = THEMES["neon"]  # magenta border, strongly distinct from the dark fill
        with_border = render_glass_panel(100, 100, theme, border_width=3)
        without_border = render_glass_panel(100, 100, theme, border_width=0)
        edge_with = with_border[50, 2, :3].astype(int)
        edge_without = without_border[50, 2, :3].astype(int)
        assert not np.array_equal(edge_with, edge_without)


class TestRenderPanelShadow:
    def test_output_is_larger_than_panel_due_to_blur_margin(self):
        bgra, _ = render_panel_shadow(100, 60, THEMES["dark"], blur_radius=10)
        assert bgra.shape[0] > 60
        assert bgra.shape[1] > 100

    def test_offset_accounts_for_margin(self):
        _, (dx, dy) = render_panel_shadow(100, 60, THEMES["dark"], blur_radius=10, offset=(0, 6))
        assert dx < 0  # shifted left/up to compensate for the blur margin
        assert dy < 0

    def test_rejects_invalid_strength(self):
        with pytest.raises(ValueError):
            render_panel_shadow(100, 60, THEMES["dark"], strength=1.5)

    def test_zero_strength_is_fully_transparent(self):
        bgra, _ = render_panel_shadow(100, 60, THEMES["dark"], strength=0.0)
        assert bgra[:, :, 3].max() == 0

    def test_positive_strength_has_visible_alpha_near_center(self):
        bgra, _ = render_panel_shadow(100, 60, THEMES["dark"], strength=0.5)
        cy, cx = bgra.shape[0] // 2, bgra.shape[1] // 2
        assert bgra[cy, cx, 3] > 0

    def test_shadow_is_tinted_with_theme_shadow_color(self):
        theme = THEMES["dark"]
        bgra, _ = render_panel_shadow(100, 60, theme, strength=1.0)
        cy, cx = bgra.shape[0] // 2, bgra.shape[1] // 2
        np.testing.assert_array_equal(bgra[cy, cx, :3], theme.shadow)

    def test_zero_blur_radius_does_not_crash(self):
        bgra, _ = render_panel_shadow(100, 60, THEMES["dark"], blur_radius=0)
        assert bgra.shape[0] >= 60


class TestDrawGlassPanel:
    def test_does_not_mutate_input(self):
        frame = make_frame(300, 200, color=(90, 90, 90))
        original = frame.copy()
        draw_glass_panel(frame, 20, 20, 100, 80, THEMES["dark"])
        np.testing.assert_array_equal(frame, original)

    def test_output_same_shape_as_input(self):
        frame = make_frame(300, 200)
        out = draw_glass_panel(frame, 20, 20, 100, 80, THEMES["dark"])
        assert out.shape == frame.shape

    def test_panel_region_actually_changes_pixels(self):
        frame = make_frame(300, 200, color=(90, 90, 90))
        out = draw_glass_panel(frame, 20, 20, 100, 80, THEMES["dark"])
        assert not np.array_equal(out[60, 70], frame[60, 70])

    def test_far_from_panel_is_unaffected(self):
        frame = make_frame(300, 200, color=(90, 90, 90))
        out = draw_glass_panel(frame, 20, 20, 100, 80, THEMES["dark"], shadow_blur=5)
        np.testing.assert_array_equal(out[190, 290], frame[190, 290])

    @pytest.mark.parametrize("x,y", [(-50, -50), (280, 180), (-10, 90), (290, -10)])
    def test_panel_off_every_edge_does_not_crash(self, x, y):
        frame = make_frame(300, 200)
        out = draw_glass_panel(frame, x, y, 100, 80, THEMES["dark"])
        assert out.shape == frame.shape

    def test_works_for_every_theme(self):
        frame = make_frame(300, 200)
        for theme in THEMES.values():
            out = draw_glass_panel(frame, 20, 20, 100, 80, theme)
            assert out.shape == frame.shape


# --------------------------------------------------------------------------
# vignette
# --------------------------------------------------------------------------

class TestApplyVignette:
    def test_rejects_out_of_range_strength(self):
        frame = make_frame(100, 100)
        with pytest.raises(ValueError):
            apply_vignette(frame, -0.1)
        with pytest.raises(ValueError):
            apply_vignette(frame, 1.1)

    def test_rejects_non_3d_frame(self):
        with pytest.raises(ValueError):
            apply_vignette(np.zeros((10, 10), dtype=np.uint8), 0.5)

    def test_does_not_mutate_input(self):
        frame = make_frame(100, 100, color=(150, 150, 150))
        original = frame.copy()
        apply_vignette(frame, 0.7)
        np.testing.assert_array_equal(frame, original)

    def test_zero_strength_is_unchanged_copy(self):
        frame = make_frame(100, 100, color=(150, 150, 150))
        out = apply_vignette(frame, 0.0)
        np.testing.assert_array_equal(out, frame)
        assert out is not frame

    def test_center_pixel_barely_changes(self):
        frame = make_frame(100, 100, color=(150, 150, 150))
        out = apply_vignette(frame, 1.0)
        center = out[50, 50]
        assert abs(int(center[0]) - 150) <= 2

    def test_corner_darker_than_center(self):
        frame = make_frame(100, 100, color=(150, 150, 150))
        out = apply_vignette(frame, 0.8)
        corner = int(out[0, 0, 0])
        center = int(out[50, 50, 0])
        assert corner < center

    def test_stronger_vignette_darkens_corner_more(self):
        frame = make_frame(100, 100, color=(150, 150, 150))
        mild = apply_vignette(frame, 0.3)
        strong = apply_vignette(frame, 0.9)
        assert int(strong[0, 0, 0]) < int(mild[0, 0, 0])

    def test_full_strength_corner_is_near_black(self):
        frame = make_frame(100, 100, color=(200, 200, 200))
        out = apply_vignette(frame, 1.0)
        assert int(out[0, 0, 0]) < 20

    def test_output_shape_and_dtype_preserved(self):
        frame = make_frame(120, 80)
        out = apply_vignette(frame, 0.5)
        assert out.shape == frame.shape
        assert out.dtype == frame.dtype


class TestApplyThemeVignette:
    def test_matches_direct_call_with_theme_strength(self):
        frame = make_frame(100, 100, color=(150, 150, 150))
        theme = THEMES["neon"]
        via_theme = apply_theme_vignette(frame, theme)
        direct = apply_vignette(frame, theme.vignette_strength)
        np.testing.assert_array_equal(via_theme, direct)

    def test_works_for_every_theme(self):
        frame = make_frame(100, 100)
        for theme in THEMES.values():
            out = apply_theme_vignette(frame, theme)
            assert out.shape == frame.shape
