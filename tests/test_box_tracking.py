"""Tests for Kalman box filter and optical flow helper."""

from __future__ import annotations

import numpy as np

from laughing_man.box_tracking import BoxKalman, optical_flow_center_shift


def test_box_kalman_smooths_sequence() -> None:
    k = BoxKalman()
    z0 = np.array([100.0, 200.0, 80.0, 90.0])
    z1 = np.array([110.0, 210.0, 85.0, 92.0])
    o0 = k.update(z0)
    o1 = k.update(z1)
    assert np.allclose(o0, z0)
    assert o1[0] > z0[0] and o1[0] < z1[0]


def test_optical_flow_center_shift_translated_patch() -> None:
    h, w = 120, 160
    rng = np.random.default_rng(0)
    prev = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    cur = np.zeros_like(prev)
    cur[2:, 3:] = prev[:-2, :-3]
    shift = optical_flow_center_shift(prev, cur, 50.0, 40.0, 60.0, 40.0)
    assert shift is not None
    dx, dy = shift
    assert 2.4 < dx < 3.6
    assert 1.6 < dy < 2.4
