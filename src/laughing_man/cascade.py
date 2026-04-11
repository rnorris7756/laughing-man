"""ROI expansion and crop-based second-stage detection helpers."""

from __future__ import annotations

import numpy as np

from laughing_man.protocols import FaceBoxSource
from laughing_man.roi import RoiState


def expand_box_with_margin(
    x: float,
    y: float,
    w: float,
    h: float,
    margin: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """
    Expand an axis-aligned box by a margin fraction of its size, then clamp.

    Parameters
    ----------
    x, y, w, h
        Source rectangle in pixel coordinates.
    margin
        Fraction of ``w`` / ``h`` added on each side (e.g. ``0.2`` adds 20% of width
        to left and right).
    frame_w, frame_h
        Frame bounds.

    Returns
    -------
    tuple[int, int, int, int]
        Integer ``(x0, y0, cw, ch)`` crop rectangle fully inside the frame.
    """
    if margin <= 0:
        ix, iy = int(round(x)), int(round(y))
        iw = max(1, int(round(w)))
        ih = max(1, int(round(h)))
        return _clamp_rect(ix, iy, iw, ih, frame_w, frame_h)

    mx = w * margin
    my = h * margin
    x0 = x - mx
    y0 = y - my
    x1 = x + w + mx
    y1 = y + h + my
    x0 = max(0.0, min(x0, float(frame_w - 1)))
    y0 = max(0.0, min(y0, float(frame_h - 1)))
    x1 = max(x0 + 1.0, min(x1, float(frame_w)))
    y1 = max(y0 + 1.0, min(y1, float(frame_h)))
    ix0 = int(round(x0))
    iy0 = int(round(y0))
    cw = max(1, int(round(x1 - x0)))
    ch = max(1, int(round(y1 - y0)))
    return _clamp_rect(ix0, iy0, cw, ch, frame_w, frame_h)


def _clamp_rect(
    x: int, y: int, w: int, h: int, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if x + w > frame_w:
        w = frame_w - x
    if y + h > frame_h:
        h = frame_h - y
    w = max(1, w)
    h = max(1, h)
    return (x, y, w, h)


def translate_box_from_crop(
    x: int,
    y: int,
    w: int,
    h: int,
    crop_x0: int,
    crop_y0: int,
) -> tuple[int, int, int, int]:
    """
    Map a box from crop coordinates into full-frame coordinates.

    Parameters
    ----------
    x, y, w, h
        Detection inside the crop.
    crop_x0, crop_y0
        Top-left of the crop in the full frame.

    Returns
    -------
    tuple[int, int, int, int]
        ``(x, y, w, h)`` in full-frame pixel coordinates.
    """
    return (x + crop_x0, y + crop_y0, w, h)


def union_boxes(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Axis-aligned union of two integer boxes.

    Parameters
    ----------
    a, b
        ``(x, y, w, h)``.

    Returns
    -------
    tuple[int, int, int, int]
        Smallest enclosing rectangle.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


class CascadedFaceBoxSource:
    """
    Run a face detector on an expanded crop from the previous smoothed ROI.

    Falls back to full-frame detection when there is no prior box or the crop
    yields no face. Intended for backends that tolerate variable input size
    (e.g. OpenCV YuNet), not MediaPipe BlazeFace VIDEO mode.

    Parameters
    ----------
    inner
        Wrapped :class:`~laughing_man.protocols.FaceBoxSource`.
    margin
        Per-side expansion as a fraction of the previous box width/height.
    roi_state
        Shared :class:`~laughing_man.roi.RoiState` updated by ``smooth_and_draw``.
    """

    def __init__(
        self,
        inner: FaceBoxSource,
        margin: float,
        roi_state: RoiState,
    ) -> None:
        self._inner = inner
        self._margin = margin
        self._state = roi_state

    def face_box(
        self, frame: np.ndarray, timestamp_ms: int
    ) -> tuple[int, int, int, int] | None:
        """Run detection on a crop from the previous box, or full frame."""
        frame_h, frame_w = frame.shape[:2]
        prev = self._state.prev
        if self._margin <= 0 or prev is None:
            return self._inner.face_box(frame, timestamp_ms)

        px, py, pw, ph = prev
        x0, y0, cw, ch = expand_box_with_margin(
            px, py, pw, ph, self._margin, frame_w, frame_h
        )
        crop = frame[y0 : y0 + ch, x0 : x0 + cw]
        if crop.size == 0:
            return self._inner.face_box(frame, timestamp_ms)

        det = self._inner.face_box(crop, timestamp_ms)
        if det is None:
            return self._inner.face_box(frame, timestamp_ms)
        dx, dy, dw, dh = det
        return translate_box_from_crop(dx, dy, dw, dh, x0, y0)
