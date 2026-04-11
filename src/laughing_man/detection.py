"""MediaPipe BlazeFace face detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from laughing_man.constants import (
    MIN_DETECTION_CONFIDENCE,
    MIN_FACE_SIZE,
    MIN_SUPPRESSION_THRESHOLD,
)


def pick_largest_face(
    detections: list,
    min_w: int,
    min_h: int,
) -> tuple[int, int, int, int] | None:
    """Choose the largest detection meeting minimum size."""
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    for det in detections:
        bb = det.bounding_box
        x, y, w, h = bb.origin_x, bb.origin_y, bb.width, bb.height
        if w < min_w or h < min_h:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    return best


def mediapipe_detect_face(
    detector: vision.FaceDetector,
    frame: np.ndarray,
    timestamp_ms: int,
) -> tuple[int, int, int, int] | None:
    """
    Run BlazeFace on this frame and return the largest face box, if any.

    Parameters
    ----------
    detector
        MediaPipe FaceDetector (VIDEO running mode).
    frame
        BGR image.
    timestamp_ms
        Monotonic time for ``detect_for_video``.

    Returns
    -------
    tuple[int, int, int, int] | None
        ``(x, y, w, h)`` in pixels, or None if no face passes the minimum size.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    min_w, min_h = MIN_FACE_SIZE
    return pick_largest_face(list(result.detections), min_w, min_h)


def face_detector_options(model_path: Path, *, use_gpu: bool) -> vision.FaceDetectorOptions:
    """Build FaceDetectorOptions for CPU or TFLite GPU delegate."""
    delegate = python.BaseOptions.Delegate.GPU if use_gpu else python.BaseOptions.Delegate.CPU
    base_options = python.BaseOptions(
        model_asset_path=str(model_path),
        delegate=delegate,
    )
    return vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_suppression_threshold=MIN_SUPPRESSION_THRESHOLD,
    )


class BlazeFaceFaceBoxSource:
    """
    :class:`~laughing_man.protocols.FaceBoxSource` using MediaPipe BlazeFace (VIDEO).
    """

    def __init__(self, detector: vision.FaceDetector) -> None:
        self._detector = detector

    def face_box(self, frame: np.ndarray, timestamp_ms: int) -> tuple[int, int, int, int] | None:
        return mediapipe_detect_face(self._detector, frame, timestamp_ms)
