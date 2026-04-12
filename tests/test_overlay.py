"""Tests for overlay cache helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image

from laughing_man.overlay import build_overlay_rgb_cache


def test_build_overlay_rgb_cache_empty() -> None:
    """Empty cache yields an empty list."""
    assert build_overlay_rgb_cache([]) == []


def test_build_overlay_rgb_cache_shared_static() -> None:
    """Custom static overlay: one PIL image in every slot shares one ndarray."""
    img = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    cache: list[list[Image.Image]] = [[img], [img], [img]]
    out = build_overlay_rgb_cache(cache)
    assert len(out) == 3
    assert out[0] is out[1] is out[2]
    assert out[0].shape == (8, 8, 3)
    assert out[0].dtype == np.uint8


def test_build_overlay_rgb_cache_distinct_rotations() -> None:
    """Different PIL frames per slot produce distinct arrays."""
    a = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    b = Image.new("RGBA", (4, 4), (0, 255, 0, 255))
    cache: list[list[Image.Image]] = [[a], [b]]
    out = build_overlay_rgb_cache(cache)
    assert len(out) == 2
    assert out[0] is not out[1]
    assert np.any(out[0] != out[1])
