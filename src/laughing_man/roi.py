"""Face ROI smoothing, privacy pass, and overlay compositing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from laughing_man.box_tracking import BoxKalman, optical_flow_center_shift
from laughing_man.constants import (
    FADEOUT_LIM,
    FLOW_CENTER_MIX,
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_PROCESS_NOISE,
    MAIN_WIN_NAME,
    OVERLAY_VERTICAL_SHIFT,
    ROI_SCALER,
)
from laughing_man.protocols import PrivacyEffect


@dataclass
class RoiState:
    """Smoothed face bounding box in float coordinates."""

    prev: tuple[float, float, float, float] | None = None
    last_face_h: float | None = None
    no_face_streak: int = 0
    kalman: BoxKalman | None = None
    prev_gray: np.ndarray | None = None


def blend_detection_centers(
    prev_cx: float,
    prev_cy: float,
    det_cx: float,
    det_cy: float,
    center_lambda: float,
) -> tuple[float, float]:
    """
    Exponential moving average on the 2D box center (horizontal and vertical).

    Parameters
    ----------
    prev_cx, prev_cy
        Center of the previous smoothed ROI.
    det_cx, det_cy
        Center of the current raw detection.
    center_lambda
        Weight on the previous center; ``(1 - center_lambda)`` on the detection.

    Returns
    -------
    tuple[float, float]
        Blended ``(cx, cy)``.
    """
    cx = center_lambda * prev_cx + (1.0 - center_lambda) * det_cx
    cy = center_lambda * prev_cy + (1.0 - center_lambda) * det_cy
    return (cx, cy)


def clamp_roi(
    x: float,
    y: float,
    w: float,
    h: float,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float, float]:
    """Clamp a rectangle so it stays inside the frame."""
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if x + w > frame_w:
        w = float(frame_w - x)
    if y + h > frame_h:
        h = float(frame_h - y)
    return (x, y, w, h)


def resize_overlay_to_face_roi(
    overlay_rgb: np.ndarray,
    mask_l: np.ndarray,
    aw: int,
    ah: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map the square overlay cache to the face ROI using a relative scale factor.

    The cache is resized to roughly ``(aw * scale, ah * scale)``, then
    center-cropped or center-padded with zeros to ``(aw, ah)``. Values above
    ``1.0`` zoom into the center of the artwork; values below ``1.0`` shrink it
    with transparent margins (mask zero).

    Parameters
    ----------
    overlay_rgb
        RGB uint8, square, same size as ``mask_l`` (typically ``min_dim``).
    mask_l
        Single-channel uint8 mask, same spatial size as ``overlay_rgb``.
    aw
        Face ROI width in pixels.
    ah
        Face ROI height in pixels.
    scale
        Relative size vs the face box; ``1.0`` matches prior behavior.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(overlay_rgb, mask_l)`` both shaped ``(ah, aw)``.
    """
    if scale <= 0:
        scale = 1.0
    tw = max(1, int(round(aw * scale)))
    th = max(1, int(round(ah * scale)))
    ov = cv2.resize(overlay_rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
    mk = cv2.resize(mask_l, (tw, th), interpolation=cv2.INTER_LINEAR)
    if tw == aw and th == ah:
        return ov, mk
    if tw >= aw and th >= ah:
        x0 = (tw - aw) // 2
        y0 = (th - ah) // 2
        return ov[y0 : y0 + ah, x0 : x0 + aw], mk[y0 : y0 + ah, x0 : x0 + aw]
    ov_pad = np.zeros((ah, aw, 3), dtype=np.uint8)
    mk_pad = np.zeros((ah, aw), dtype=np.uint8)
    x0 = (aw - tw) // 2
    y0 = (ah - th) // 2
    ov_pad[y0 : y0 + th, x0 : x0 + tw] = ov
    mk_pad[y0 : y0 + th, x0 : x0 + tw] = mk
    return ov_pad, mk_pad


def _clear_roi_tracking(state: RoiState) -> None:
    state.prev = None
    state.last_face_h = None
    state.kalman = None
    state.prev_gray = None


def smooth_and_draw(
    frame: np.ndarray,
    raw_face: tuple[int, int, int, int] | None,
    state: RoiState,
    overlay_rgb: np.ndarray,
    mask_l: np.ndarray,
    *,
    privacy_effect: PrivacyEffect,
    center_lambda: float,
    size_lambda: float,
    no_face_blur_frames: int,
    show_preview: bool,
    overlay_scale: float,
    roi_motion: Literal["ema", "kalman", "kalman_flow"] = "ema",
) -> None:
    """
    Apply full-frame privacy effect after consecutive no-face frames, temporal
    smoothing to the face box when a face is present, then alpha-composite the
    overlay on the face (or on the last known box during the debounce window).

    Parameters
    ----------
    frame
        BGR image (modified in place in the face region).
    raw_face
        Largest face box from the active detector this frame, or None if none qualify.
    state
        Tracks the smoothed ROI between frames.
    overlay_rgb
        RGB uint8 image, same size as the face ROI when drawn.
    mask_l
        Single-channel uint8 mask (0–255), same spatial size as ``overlay_rgb``.
    privacy_effect
        Full-frame treatment when no face is detected (after debounce).
    center_lambda
        Low-pass on the **horizontal and vertical** center of the box (higher =
        stickier position; reduces per-frame jitter when sitting still).
    size_lambda
        Low-pass on **width and height** (higher = less size jitter from the
        detector; raise above ``center_lambda`` if width/height jitter dominates.
    no_face_blur_frames
        Require this many consecutive frames with no face before applying privacy
        blur; until then the last smoothed overlay position is held fixed.
    show_preview
        If True, show frames in an OpenCV window.
    overlay_scale
        Resize the overlay relative to the face ROI before blending (see
        ``--scale``). ``1.0`` preserves the previous mapping from the square cache
        to the face box.
    roi_motion
        ``ema`` — exponential moving average (``--roi-lambda`` / ``--size-lambda``).
        ``kalman`` — constant-velocity Kalman on ``(cx, cy, w, h)``.
        ``kalman_flow`` — Kalman with sparse LK optical flow to blend the measured
        center toward motion from the previous frame.
    """
    frame_h, frame_w = frame.shape[:2]

    if raw_face is not None:
        state.no_face_streak = 0
    else:
        state.no_face_streak += 1

    apply_privacy = state.no_face_streak >= no_face_blur_frames
    privacy_effect.apply(frame, 1.0 if apply_privacy else 0.0)

    r: tuple[int, int, int, int] | None = None

    try:
        if raw_face is not None:
            x, y, w, h = raw_face
            face_h = float(h)
            state.last_face_h = face_h
            y_shift = OVERLAY_VERTICAL_SHIFT * face_h
            det_cx = float(x) + float(w) / 2.0
            det_cy = float(y) + float(h) / 2.0
            w_det = float(w) * ROI_SCALER
            h_det = float(h) * ROI_SCALER

            if roi_motion == "ema":
                if state.prev is None:
                    sy = float(y) - y_shift
                    state.prev = (float(x), sy, float(w), float(h))
                    r = (x, int(round(sy)), w, h)
                else:
                    px, py, pw, ph = state.prev
                    new_w = size_lambda * pw + (1.0 - size_lambda) * w_det
                    new_h = size_lambda * ph + (1.0 - size_lambda) * h_det
                    prev_cx = px + pw / 2.0
                    prev_cy = py + ph / 2.0
                    roi_cx, roi_cy = blend_detection_centers(prev_cx, prev_cy, det_cx, det_cy, center_lambda)
                    new_x = roi_cx - new_w / 2.0
                    new_y = roi_cy - new_h / 2.0 - y_shift
                    new_x, new_y, new_w, new_h = clamp_roi(new_x, new_y, new_w, new_h, frame_w, frame_h)
                    state.prev = (new_x, new_y, new_w, new_h)
                    ix, iy = int(round(new_x)), int(round(new_y))
                    iw, ih = int(round(new_w)), int(round(new_h))
                    if iw < FADEOUT_LIM or ih < FADEOUT_LIM:
                        _clear_roi_tracking(state)
                    else:
                        r = (ix, iy, iw, ih)

            else:
                if state.kalman is None:
                    state.kalman = BoxKalman(
                        process_noise=KALMAN_PROCESS_NOISE,
                        measurement_noise=KALMAN_MEASUREMENT_NOISE,
                    )

                meas_cx = det_cx
                meas_cy = det_cy
                if roi_motion == "kalman_flow" and state.prev_gray is not None and state.prev is not None:
                    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    px, py, pw, ph = state.prev
                    shift = optical_flow_center_shift(
                        state.prev_gray,
                        cur_gray,
                        px,
                        py,
                        pw,
                        ph,
                    )
                    if shift is not None:
                        dx, dy = shift
                        prev_cx = px + pw / 2.0
                        prev_cy = py + ph / 2.0
                        mix = FLOW_CENTER_MIX
                        meas_cx = (1.0 - mix) * det_cx + mix * (prev_cx + dx)
                        meas_cy = (1.0 - mix) * det_cy + mix * (prev_cy + dy)

                z = np.array([meas_cx, meas_cy, w_det, h_det], dtype=np.float64)
                cx, cy, kw, kh = state.kalman.update(z)
                new_x = float(cx) - kw / 2.0
                new_y = float(cy) - kh / 2.0 - y_shift
                new_x, new_y, new_w, new_h = clamp_roi(new_x, new_y, float(kw), float(kh), frame_w, frame_h)
                state.prev = (new_x, new_y, new_w, new_h)
                ix, iy = int(round(new_x)), int(round(new_y))
                iw, ih = int(round(new_w)), int(round(new_h))
                if iw < FADEOUT_LIM or ih < FADEOUT_LIM:
                    _clear_roi_tracking(state)
                else:
                    r = (ix, iy, iw, ih)

        elif not apply_privacy and state.prev is not None:
            px, py, pw, ph = state.prev
            new_x, new_y, new_w, new_h = clamp_roi(px, py, pw, ph, frame_w, frame_h)
            ix, iy = int(round(new_x)), int(round(new_y))
            iw, ih = int(round(new_w)), int(round(new_h))
            if iw >= FADEOUT_LIM and ih >= FADEOUT_LIM:
                r = (ix, iy, iw, ih)

        if r is None:
            if show_preview:
                cv2.imshow(MAIN_WIN_NAME, frame)
            return

        rx, ry, rw, rh = r
        face_roi = frame[ry : ry + rh, rx : rx + rw]
        if face_roi.size == 0:
            if show_preview:
                cv2.imshow(MAIN_WIN_NAME, frame)
            return

        ah, aw = face_roi.shape[:2]
        ov, mk = resize_overlay_to_face_roi(overlay_rgb, mask_l, aw, ah, overlay_scale)

        overlay_bgr = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR).astype(np.float32)
        mask_f = (mk.astype(np.float32) / 255.0)[..., np.newaxis]
        base = face_roi.astype(np.float32)
        blended = base * (1.0 - mask_f) + overlay_bgr * mask_f
        face_roi[:] = np.clip(blended, 0, 255).astype(np.uint8)

        if show_preview:
            cv2.imshow(MAIN_WIN_NAME, frame)
    finally:
        if roi_motion == "kalman_flow":
            state.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).copy()
        elif state.prev_gray is not None:
            state.prev_gray = None
