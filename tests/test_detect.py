"""
Unit tests for detect.py. Everything here runs against real OpenCV calls
(cv2.Canny, cv2.findContours, etc.) on synthetic drawn frames — no camera,
no learned weights, nothing to mock. Same "test the real thing when it
doesn't need real hardware" approach used for the real Poppins fonts and
real pygame elsewhere in this project.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protovision.capture import GuideBox
from protovision.detect import (
    propose_regions,
    _boxes_overlap_fraction,
    _merge_overlapping_boxes,
)


def blank_frame(w=600, h=400, color=200):
    return np.full((h, w, 3), color, dtype=np.uint8)


def draw_shapes(frame, shapes):
    """shapes: list of ('rect', x1, y1, x2, y2, bgr) or ('circle', cx, cy, r, bgr)."""
    out = frame.copy()
    for shape in shapes:
        if shape[0] == "rect":
            _, x1, y1, x2, y2, color = shape
            cv2.rectangle(out, (x1, y1), (x2, y2), color, -1)
        elif shape[0] == "circle":
            _, cx, cy, r, color = shape
            cv2.circle(out, (cx, cy), r, color, -1)
    return out


# --------------------------------------------------------------------------
# _boxes_overlap_fraction
# --------------------------------------------------------------------------

class TestBoxesOverlapFraction:
    def test_identical_boxes_fully_overlap(self):
        box = GuideBox(10, 10, 50, 50)
        assert _boxes_overlap_fraction(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes_do_not_overlap(self):
        a = GuideBox(0, 0, 10, 10)
        b = GuideBox(100, 100, 110, 110)
        assert _boxes_overlap_fraction(a, b) == 0.0

    def test_touching_but_not_overlapping_edges(self):
        a = GuideBox(0, 0, 10, 10)
        b = GuideBox(10, 0, 20, 10)  # shares an edge, no area overlap
        assert _boxes_overlap_fraction(a, b) == 0.0

    def test_small_box_fully_inside_large_box(self):
        big = GuideBox(0, 0, 100, 100)
        small = GuideBox(40, 40, 60, 60)
        # intersection == small box's full area -> fraction of the SMALLER box == 1.0
        assert _boxes_overlap_fraction(big, small) == pytest.approx(1.0)

    def test_partial_overlap_is_between_zero_and_one(self):
        a = GuideBox(0, 0, 50, 50)
        b = GuideBox(25, 25, 75, 75)
        frac = _boxes_overlap_fraction(a, b)
        assert 0.0 < frac < 1.0

    def test_symmetric(self):
        a = GuideBox(0, 0, 50, 50)
        b = GuideBox(25, 25, 75, 75)
        assert _boxes_overlap_fraction(a, b) == pytest.approx(_boxes_overlap_fraction(b, a))


# --------------------------------------------------------------------------
# _merge_overlapping_boxes
# --------------------------------------------------------------------------

class TestMergeOverlappingBoxes:
    def test_empty_list(self):
        assert _merge_overlapping_boxes([]) == []

    def test_single_box_unchanged(self):
        box = GuideBox(0, 0, 10, 10)
        assert _merge_overlapping_boxes([box]) == [box]

    def test_disjoint_boxes_all_kept(self):
        a = GuideBox(0, 0, 10, 10)
        b = GuideBox(100, 100, 110, 110)
        result = _merge_overlapping_boxes([a, b])
        assert len(result) == 2

    def test_heavily_overlapping_boxes_merged_to_one(self):
        big = GuideBox(0, 0, 100, 100)
        nested = GuideBox(10, 10, 90, 90)  # fully inside `big`, overlap fraction 1.0
        result = _merge_overlapping_boxes([big, nested], overlap_threshold=0.5)
        assert len(result) == 1

    def test_kept_box_is_the_larger_one(self):
        big = GuideBox(0, 0, 100, 100)
        nested = GuideBox(10, 10, 90, 90)
        result = _merge_overlapping_boxes([nested, big], overlap_threshold=0.5)  # order shouldn't matter
        assert result == [big]

    def test_overlap_below_threshold_keeps_both(self):
        a = GuideBox(0, 0, 50, 50)
        b = GuideBox(45, 45, 95, 95)  # small corner overlap only
        result = _merge_overlapping_boxes([a, b], overlap_threshold=0.9)
        assert len(result) == 2


# --------------------------------------------------------------------------
# propose_regions — validation
# --------------------------------------------------------------------------

class TestProposeRegionsValidation:
    def test_rejects_non_3d_frame(self):
        with pytest.raises(ValueError):
            propose_regions(np.zeros((10, 10), dtype=np.uint8))

    def test_rejects_min_area_greater_than_max_area(self):
        with pytest.raises(ValueError):
            propose_regions(blank_frame(), min_area_fraction=0.5, max_area_fraction=0.1)

    def test_rejects_min_area_equal_to_max_area(self):
        with pytest.raises(ValueError):
            propose_regions(blank_frame(), min_area_fraction=0.2, max_area_fraction=0.2)

    def test_rejects_out_of_range_fractions(self):
        with pytest.raises(ValueError):
            propose_regions(blank_frame(), min_area_fraction=-0.1)
        with pytest.raises(ValueError):
            propose_regions(blank_frame(), max_area_fraction=1.5)

    def test_rejects_non_positive_max_regions(self):
        with pytest.raises(ValueError):
            propose_regions(blank_frame(), max_regions=0)


# --------------------------------------------------------------------------
# propose_regions — real detection on synthetic frames
# --------------------------------------------------------------------------

class TestProposeRegionsDetection:
    def test_blank_frame_finds_nothing(self):
        assert propose_regions(blank_frame()) == []

    def test_single_object_finds_one_region(self):
        frame = draw_shapes(blank_frame(), [("rect", 50, 50, 150, 150, (0, 0, 200))])
        regions = propose_regions(frame)
        assert len(regions) == 1

    def test_single_object_region_roughly_matches_drawn_area(self):
        frame = draw_shapes(blank_frame(), [("rect", 50, 50, 150, 150, (0, 0, 200))])
        regions = propose_regions(frame)
        box = regions[0]
        # allow some slack for anti-aliasing/edge dilation growing the box slightly
        assert 80 <= box.width <= 120
        assert 80 <= box.height <= 120

    def test_three_separated_objects_find_three_regions(self):
        frame = draw_shapes(blank_frame(), [
            ("rect", 50, 50, 150, 150, (0, 0, 200)),
            ("circle", 350, 100, 60, (0, 150, 0)),
            ("rect", 400, 250, 550, 350, (200, 100, 0)),
        ])
        regions = propose_regions(frame)
        assert len(regions) == 3

    def test_regions_sorted_largest_first(self):
        frame = draw_shapes(blank_frame(), [
            ("rect", 50, 50, 100, 100, (0, 0, 200)),      # small: 50x50
            ("rect", 300, 50, 500, 250, (0, 150, 0)),      # large: 200x200
        ])
        regions = propose_regions(frame)
        areas = [r.width * r.height for r in regions]
        assert areas == sorted(areas, reverse=True)

    def test_max_regions_caps_the_count(self):
        shapes = [("rect", i * 90, 20, i * 90 + 60, 80, (0, 0, 200)) for i in range(6)]
        frame = draw_shapes(blank_frame(w=700), shapes)
        regions = propose_regions(frame, max_regions=2)
        assert len(regions) <= 2

    def test_tiny_speck_filtered_out_by_min_area(self):
        frame = draw_shapes(blank_frame(), [("rect", 100, 100, 103, 103, (0, 0, 200))])  # 3x3 speck
        regions = propose_regions(frame, min_area_fraction=0.01)
        assert regions == []

    def test_huge_object_filtered_out_by_max_area(self):
        # a rectangle covering most of the frame should be excluded by the
        # default max_area_fraction (background, not an "object")
        frame = draw_shapes(blank_frame(w=600, h=400), [("rect", 5, 5, 595, 395, (0, 0, 200))])
        regions = propose_regions(frame, max_area_fraction=0.5)
        assert regions == []

    def test_overlapping_objects_do_not_produce_duplicate_regions(self):
        frame = draw_shapes(blank_frame(), [
            ("rect", 50, 50, 200, 200, (0, 0, 200)),
            ("rect", 60, 60, 190, 190, (0, 150, 0)),  # nested inside the first
        ])
        regions = propose_regions(frame)
        # the two nested rectangles' combined edges shouldn't produce more
        # than a couple of heavily-overlapping boxes after merging
        assert len(regions) <= 2

    def test_returns_guidebox_instances(self):
        frame = draw_shapes(blank_frame(), [("rect", 50, 50, 150, 150, (0, 0, 200))])
        regions = propose_regions(frame)
        assert all(isinstance(r, GuideBox) for r in regions)

    def test_regions_stay_within_frame_bounds(self):
        frame = draw_shapes(blank_frame(w=600, h=400), [
            ("rect", 50, 50, 150, 150, (0, 0, 200)),
            ("circle", 350, 100, 60, (0, 150, 0)),
        ])
        regions = propose_regions(frame)
        for r in regions:
            assert 0 <= r.x1 < r.x2 <= 600
            assert 0 <= r.y1 < r.y2 <= 400

    def test_noisy_background_does_not_produce_false_positives(self):
        rng = np.random.default_rng(0)
        h, w = 400, 600
        base = np.full((h, w), 190, dtype=np.uint8)
        noise = rng.integers(-8, 8, size=(h, w))
        gray_bg = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
        frame = np.dstack([gray_bg, gray_bg, gray_bg]).astype(np.uint8)
        frame = draw_shapes(frame, [
            ("rect", 50, 50, 150, 150, (0, 0, 200)),
            ("circle", 350, 100, 60, (0, 150, 0)),
        ])
        regions = propose_regions(frame)
        assert len(regions) == 2  # exactly the two real objects, no noise-triggered extras

    def test_custom_max_regions_of_one_returns_the_largest(self):
        frame = draw_shapes(blank_frame(), [
            ("rect", 50, 50, 100, 100, (0, 0, 200)),      # small
            ("rect", 300, 50, 500, 250, (0, 150, 0)),      # large
        ])
        regions = propose_regions(frame, max_regions=1)
        assert len(regions) == 1
        assert regions[0].width * regions[0].height > 100 * 100
