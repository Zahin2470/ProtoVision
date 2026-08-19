"""
Unit tests for capture.py's pure logic (guide box geometry, cropping,
drawing). `Camera` itself opens real hardware and is NOT tested here —
there's no webcam in this sandbox. See module docstring in capture.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protovision.capture import GuideBox, compute_guide_box, crop_guide_box, draw_guide_box


def make_frame(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# compute_guide_box
# --------------------------------------------------------------------------

class TestComputeGuideBox:
    def test_box_is_centered(self):
        box = compute_guide_box(640, 480, box_fraction=0.5)
        frame_cx, frame_cy = 320, 240
        box_cx = (box.x1 + box.x2) / 2
        box_cy = (box.y1 + box.y2) / 2
        assert abs(box_cx - frame_cx) <= 1
        assert abs(box_cy - frame_cy) <= 1

    def test_box_size_matches_fraction_of_shorter_side(self):
        box = compute_guide_box(640, 480, box_fraction=0.5)
        expected_side = int(480 * 0.5)  # shorter dimension is height
        assert box.width == expected_side
        assert box.height == expected_side

    def test_box_fits_inside_frame(self):
        for w, h in [(640, 480), (1920, 1080), (480, 640), (100, 100), (321, 241)]:
            box = compute_guide_box(w, h, box_fraction=0.7)
            assert box.x1 >= 0
            assert box.y1 >= 0
            assert box.x2 <= w
            assert box.y2 <= h

    def test_box_is_square(self):
        box = compute_guide_box(1920, 1080, box_fraction=0.5)
        assert box.width == box.height

    def test_larger_fraction_gives_larger_box(self):
        small_box = compute_guide_box(640, 480, box_fraction=0.3)
        big_box = compute_guide_box(640, 480, box_fraction=0.8)
        assert big_box.width > small_box.width

    def test_never_smaller_than_one_patch(self):
        box = compute_guide_box(20, 20, box_fraction=0.1)
        assert box.width >= 16
        assert box.height >= 16

    def test_never_larger_than_frame(self):
        box = compute_guide_box(20, 20, box_fraction=1.0)
        assert box.width <= 20
        assert box.height <= 20

    def test_rejects_fraction_out_of_range(self):
        with pytest.raises(ValueError):
            compute_guide_box(640, 480, box_fraction=0.0)
        with pytest.raises(ValueError):
            compute_guide_box(640, 480, box_fraction=1.5)

    def test_rejects_non_positive_dimensions(self):
        with pytest.raises(ValueError):
            compute_guide_box(0, 480)
        with pytest.raises(ValueError):
            compute_guide_box(640, -10)

    def test_works_on_tiny_odd_frame(self):
        # Regression guard: odd dimensions near the minimum shouldn't push
        # the box out of bounds via integer-division rounding.
        box = compute_guide_box(17, 17, box_fraction=1.0)
        assert box.x1 >= 0 and box.y1 >= 0
        assert box.x2 <= 17 and box.y2 <= 17

    def test_returns_guide_box_instance(self):
        box = compute_guide_box(640, 480)
        assert isinstance(box, GuideBox)


class TestGuideBoxProperties:
    def test_width_height(self):
        box = GuideBox(10, 20, 110, 170)
        assert box.width == 100
        assert box.height == 150

    def test_as_tuple(self):
        box = GuideBox(1, 2, 3, 4)
        assert box.as_tuple() == (1, 2, 3, 4)

    def test_is_immutable(self):
        box = GuideBox(1, 2, 3, 4)
        with pytest.raises(Exception):
            box.x1 = 99  # frozen dataclass


# --------------------------------------------------------------------------
# crop_guide_box
# --------------------------------------------------------------------------

class TestCropGuideBox:
    def test_crop_has_expected_shape(self):
        frame = make_frame(640, 480)
        box = GuideBox(100, 50, 300, 250)
        crop = crop_guide_box(frame, box)
        assert crop.shape == (200, 200, 3)

    def test_crop_pulls_correct_region(self):
        frame = make_frame(10, 10)
        frame[2:5, 3:7] = 255  # a distinct patch
        box = GuideBox(3, 2, 7, 5)
        crop = crop_guide_box(frame, box)
        assert np.all(crop == 255)

    def test_full_frame_box(self):
        frame = make_frame(64, 64)
        box = compute_guide_box(64, 64, box_fraction=1.0)
        crop = crop_guide_box(frame, box)
        assert crop.shape[0] <= 64 and crop.shape[1] <= 64

    def test_out_of_bounds_box_raises(self):
        frame = make_frame(100, 100)
        box = GuideBox(50, 50, 150, 150)  # extends past the frame
        with pytest.raises(ValueError):
            crop_guide_box(frame, box)

    def test_negative_origin_raises(self):
        frame = make_frame(100, 100)
        box = GuideBox(-10, 0, 50, 50)
        with pytest.raises(ValueError):
            crop_guide_box(frame, box)

    def test_zero_size_box_raises(self):
        frame = make_frame(100, 100)
        box = GuideBox(10, 10, 10, 10)
        with pytest.raises(ValueError):
            crop_guide_box(frame, box)

    def test_compute_then_crop_roundtrip_never_raises(self):
        # The realistic path: compute a box for a real frame size, then crop
        # that exact frame with it — should never hit the bounds check.
        for w, h in [(640, 480), (1280, 720), (100, 100), (321, 241)]:
            frame = make_frame(w, h)
            box = compute_guide_box(w, h, box_fraction=0.6)
            crop = crop_guide_box(frame, box)
            assert crop.size > 0


# --------------------------------------------------------------------------
# draw_guide_box
# --------------------------------------------------------------------------

class TestDrawGuideBox:
    def test_returns_same_shape(self):
        frame = make_frame(100, 100)
        box = compute_guide_box(100, 100)
        out = draw_guide_box(frame, box)
        assert out.shape == frame.shape

    def test_does_not_mutate_original_frame(self):
        frame = make_frame(100, 100)
        original = frame.copy()
        box = compute_guide_box(100, 100)
        draw_guide_box(frame, box)
        np.testing.assert_array_equal(frame, original)

    def test_actually_draws_something(self):
        frame = make_frame(100, 100)
        box = compute_guide_box(100, 100, box_fraction=0.5)
        out = draw_guide_box(frame, box, color=(0, 255, 0))
        assert not np.array_equal(out, frame)  # pixels changed somewhere
