"""Kalman filtering and optical flow helpers for face box stabilization."""

from __future__ import annotations

import cv2
import numpy as np


def build_cv_transition_matrix(dt: float = 1.0) -> np.ndarray:
    """
    Constant-velocity transition matrix for state ``[cx, cy, w, h, vx, vy, vw, vh]``.

    Parameters
    ----------
    dt
        Time step (one frame).

    Returns
    -------
    numpy.ndarray
        ``(8, 8)`` float32.
    """
    f = np.eye(8, dtype=np.float32)
    f[0, 4] = dt
    f[1, 5] = dt
    f[2, 6] = dt
    f[3, 7] = dt
    return f


class BoxKalman:
    """
    OpenCV Kalman filter on face box center and size (axis-aligned).

    State: ``cx, cy, w, h, vx, vy, vw, vh``. Measurement: ``cx, cy, w, h``.
    """

    def __init__(
        self,
        *,
        process_noise: float = 1e-2,
        measurement_noise: float = 4e-1,
    ) -> None:
        self._kf = cv2.KalmanFilter(8, 4)
        self._kf.transitionMatrix = build_cv_transition_matrix(1.0)
        self._kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self._kf.processNoiseCov = np.eye(8, dtype=np.float32) * process_noise
        self._kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * measurement_noise
        self._kf.errorCovPost = np.eye(8, dtype=np.float32) * 0.1
        self._inited = False

    def reset(self) -> None:
        """Clear filter state so the next update re-initializes from a measurement."""
        self._inited = False

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Incorporate a measurement and return smoothed ``[cx, cy, w, h]``.

        Parameters
        ----------
        z
            Shape ``(4,)``: ``cx, cy, w, h``.

        Returns
        -------
        numpy.ndarray
            Posterior ``(cx, cy, w, h)``.
        """
        zm = z.reshape(4, 1).astype(np.float32)
        if not self._inited:
            self._kf.statePost = np.zeros((8, 1), dtype=np.float32)
            self._kf.statePost[:4] = zm
            self._kf.statePost[4:] = 0.0
            self._inited = True
            return z.copy()

        self._kf.predict()
        self._kf.correct(zm)
        return self._kf.statePost[:4, 0].copy()


def optical_flow_center_shift(
    prev_gray: np.ndarray,
    cur_gray: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    margin: float = 0.1,
    grid: int = 5,
) -> tuple[float, float] | None:
    """
    Estimate median displacement of features between frames inside a rectangle.

    Parameters
    ----------
    prev_gray, cur_gray
        Consecutive frames, same shape, uint8 single channel.
    x, y, w, h
        Region of interest in the **current** frame coordinate system; the same
        coordinates apply to ``prev_gray`` when motion is small.
    margin
        Expand the ROI by this fraction of ``w`` and ``h`` before sampling.
    grid
        ``grid x grid`` points on a regular lattice inside the ROI.

    Returns
    -------
    tuple[float, float] | None
        ``(dx, dy)`` median displacement, or None if too few inliers.
    """
    fh, fw = cur_gray.shape[:2]
    mx = w * margin
    my = h * margin
    x0 = max(0, int(round(x - mx)))
    y0 = max(0, int(round(y - my)))
    x1 = min(fw, int(round(x + w + mx)))
    y1 = min(fh, int(round(y + h + my)))
    if x1 <= x0 + 3 or y1 <= y0 + 3:
        return None

    roi_prev = prev_gray[y0:y1, x0:x1]
    roi_cur = cur_gray[y0:y1, x0:x1]
    rh, rw = roi_prev.shape[:2]
    if rh < 5 or rw < 5:
        return None

    xs = np.linspace(2, rw - 3, grid, dtype=np.float32)
    ys = np.linspace(2, rh - 3, grid, dtype=np.float32)
    pts = np.array([[xv, yv] for yv in ys for xv in xs], dtype=np.float32)
    p0 = pts.reshape(-1, 1, 2)

    p1, st, _ = cv2.calcOpticalFlowPyrLK(  # ty: ignore[no-matching-overload]
        roi_prev,
        roi_cur,
        p0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if p1 is None or st is None:
        return None
    good = st.ravel() == 1
    if int(np.sum(good)) < max(5, grid * grid // 4):
        return None
    d = (p1 - p0).reshape(-1, 2)[good]
    dx = float(np.median(d[:, 0]))
    dy = float(np.median(d[:, 1]))
    return (dx, dy)
