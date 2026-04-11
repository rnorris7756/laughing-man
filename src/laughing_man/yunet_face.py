"""OpenCV YuNet face detection (FaceDetectorYN)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from laughing_man.constants import MIN_FACE_SIZE


def create_yunet_detector(
    model_path: Path,
    frame_w: int,
    frame_h: int,
    *,
    score_threshold: float = 0.65,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> cv2.FaceDetectorYN:
    """
    Build a YuNet detector for the given frame size.

    Parameters
    ----------
    model_path
        Path to ``face_detection_yunet_2023mar.onnx``.
    frame_w, frame_h
        Initial input size; :meth:`YuNetFaceBoxSource.face_box` updates this each
        frame for variable resolutions (e.g. cascaded crops).
    score_threshold
        Minimum detection score in ``[0, 1]``.
    nms_threshold
        NMS IoU threshold.
    top_k
        Max faces before NMS.

    Returns
    -------
    cv2.FaceDetectorYN
        OpenCV face detector instance.
    """
    det = cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (max(1, frame_w), max(1, frame_h)),
        score_threshold,
        nms_threshold,
        top_k,
    )
    return det


def pick_largest_yunet_face(
    faces: np.ndarray | None,
    min_w: int,
    min_h: int,
) -> tuple[int, int, int, int] | None:
    """
    Choose the largest YuNet row as ``(x, y, w, h)``.

    Parameters
    ----------
    faces
        ``Nx15`` array from :meth:`cv2.FaceDetectorYN.detect`, or None.
    min_w, min_h
        Minimum box dimensions.

    Returns
    -------
    tuple[int, int, int, int] | None
        Integer pixel box or None.
    """
    if faces is None or faces.size == 0:
        return None
    best: tuple[int, int, int, int] | None = None
    best_area = 0
    n = faces.shape[0]
    for i in range(n):
        row = faces[i]
        x, y = int(round(row[0])), int(round(row[1]))
        w, h = int(round(row[2])), int(round(row[3]))
        if w < min_w or h < min_h:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    return best


class YuNetFaceBoxSource:
    """
    :class:`~laughing_man.protocols.FaceBoxSource` using OpenCV YuNet.

    ``timestamp_ms`` is ignored (YuNet has no VIDEO temporal state).
    """

    def __init__(self, detector: cv2.FaceDetectorYN) -> None:
        self._detector = detector

    def face_box(self, frame: np.ndarray, timestamp_ms: int) -> tuple[int, int, int, int] | None:
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame)
        min_w, min_h = MIN_FACE_SIZE
        return pick_largest_yunet_face(faces, min_w, min_h)
