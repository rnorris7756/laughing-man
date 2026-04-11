"""Webcam capture helpers."""

from __future__ import annotations

import cv2
import typer
from loguru import logger


def open_webcam(camera_index: int) -> cv2.VideoCapture:
    """
    Open a camera by index.

    Temporarily silences OpenCV's verbose FFmpeg/backend logging on failure so
    the process can exit with a single clear message instead of a multi-line
    OpenCV stack trace.

    Parameters
    ----------
    camera_index
        ``VideoCapture`` device index (typically ``0`` for the default webcam).

    Returns
    -------
    cv2.VideoCapture
        An opened capture device.

    Raises
    ------
    typer.Exit
        If the device cannot be opened.
    """
    prev_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    try:
        try:
            cap = cv2.VideoCapture(camera_index)
        except Exception as e:
            logger.error(camera_open_failed_message(camera_index, detail=str(e)))
            raise typer.Exit(code=1) from e
    finally:
        cv2.utils.logging.setLogLevel(prev_level)

    if not cap.isOpened():
        logger.error(camera_open_failed_message(camera_index))
        raise typer.Exit(code=1)
    return cap


def camera_open_failed_message(camera_index: int, detail: str | None = None) -> str:
    """
    Build a user-facing message when the webcam cannot be opened.

    Parameters
    ----------
    camera_index
        Index that was passed to ``VideoCapture``.
    detail
        Optional exception text from OpenCV (shown on one line when present).

    Returns
    -------
    str
        Full message for stderr.
    """
    lines = [
        f"ERROR: Could not open the webcam (camera index {camera_index}).",
        "",
        "No usable video capture device was found at that index. Typical causes:",
        "  • No camera is connected, or the wrong device index was chosen.",
        "  • Another application is using the camera; close it and try again.",
        "  • On Linux, you may need permission to access /dev/video* (e.g. be in the "
        '"video" group, or adjust permissions).',
    ]
    if detail:
        lines.extend(["", f"Technical detail: {detail}"])
    return "\n".join(lines)
