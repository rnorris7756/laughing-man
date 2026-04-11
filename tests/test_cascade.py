"""Tests for cascade ROI helpers."""

from __future__ import annotations

from laughing_man.cascade import (
    expand_box_with_margin,
    translate_box_from_crop,
    union_boxes,
)
from laughing_man.roi import blend_detection_centers


def test_expand_box_with_margin_clamps() -> None:
    x, y, w, h = expand_box_with_margin(100.0, 100.0, 50.0, 50.0, 0.2, 640, 480)
    assert x >= 0 and y >= 0
    assert w >= 1 and h >= 1
    assert x + w <= 640
    assert y + h <= 480


def test_translate_box_from_crop() -> None:
    assert translate_box_from_crop(10, 20, 30, 40, 100, 200) == (110, 220, 30, 40)


def test_union_boxes() -> None:
    assert union_boxes((0, 0, 10, 10), (5, 5, 10, 10)) == (0, 0, 15, 15)


def test_blend_detection_centers() -> None:
    cx, cy = blend_detection_centers(10.0, 20.0, 30.0, 40.0, 0.5)
    assert cx == 20.0 and cy == 30.0


def test_blend_detection_centers_extremes() -> None:
    assert blend_detection_centers(0.0, 0.0, 100.0, 200.0, 1.0) == (0.0, 0.0)
    assert blend_detection_centers(0.0, 0.0, 100.0, 200.0, 0.0) == (100.0, 200.0)
