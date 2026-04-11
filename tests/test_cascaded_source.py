"""Tests for CascadedFaceBoxSource."""

from __future__ import annotations

import numpy as np

from laughing_man.cascade import CascadedFaceBoxSource
from laughing_man.protocols import FaceBoxSource
from laughing_man.roi import RoiState


class _FixedBoxSource:
    """Returns a fixed box in crop coordinates (for testing mapping)."""

    def __init__(self, box: tuple[int, int, int, int] | None) -> None:
        self._box = box

    def face_box(self, frame: np.ndarray, timestamp_ms: int) -> tuple[int, int, int, int] | None:
        return self._box


def test_cascade_maps_detection_to_full_frame() -> None:
    state = RoiState()
    state.prev = (100.0, 100.0, 80.0, 80.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    inner: FaceBoxSource = _FixedBoxSource((5, 6, 20, 22))
    src = CascadedFaceBoxSource(inner, margin=0.25, roi_state=state)
    out = src.face_box(frame, 0)
    assert out is not None
    assert out == (85, 86, 20, 22)


def test_cascade_falls_back_when_no_prev() -> None:
    state = RoiState()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    inner: FaceBoxSource = _FixedBoxSource((10, 10, 30, 40))
    src = CascadedFaceBoxSource(inner, margin=0.5, roi_state=state)
    assert src.face_box(frame, 0) == (10, 10, 30, 40)
