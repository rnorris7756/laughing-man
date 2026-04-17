"""Webcam capture helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import cv2
import typer
from loguru import logger

from laughing_man.constants import DEFAULT_CAMERA


def parse_camera_token(value: str) -> int | str:
    """
    Parse a ``--camera`` value that is not ``auto``.

    Parameters
    ----------
    value
        Non-empty string: decimal index or ``/dev/video*`` path.

    Returns
    -------
    int | str
        Integer index or device path for :class:`cv2.VideoCapture`.
    """
    s = value.strip()
    if s.startswith("/"):
        return s
    if s.isdigit():
        return int(s)
    logger.error(
        "Invalid --camera value {!r}: expected a non-negative integer, a /dev/video* path, or 'auto'.",
        value,
    )
    raise typer.Exit(code=1)


def _linux_list_video_basenames() -> list[str]:
    """Return sorted ``videoN`` names under ``/sys/class/video4linux``, if any."""
    root = Path("/sys/class/video4linux")
    if not root.is_dir():
        return []
    out: list[tuple[int, str]] = []
    for p in root.iterdir():
        if not p.name.startswith("video"):
            continue
        suffix = p.name[5:]
        if not suffix.isdigit():
            continue
        out.append((int(suffix), p.name))
    out.sort(key=lambda x: x[0])
    return [name for _, name in out]


def linux_is_v4l2_loopback(video_basename: str) -> bool:
    """
    Return True if sysfs reports this node is driven by v4l2loopback.

    Opening v4l2loopback devices for *capture* can hang or crash native code
    (no Python traceback). Callers should skip these when choosing an input.

    Parameters
    ----------
    video_basename
        e.g. ``video0`` (not a full ``/dev/...`` path).

    Returns
    -------
    bool
        True when the driver symlink resolves to ``v4l2loopback``.
    """
    driver = Path("/sys/class/video4linux") / video_basename / "device" / "driver"
    try:
        return driver.resolve().name == "v4l2loopback"
    except OSError:
        return False


def linux_non_loopback_indices() -> list[int]:
    """
    List numeric indices for V4L2 nodes that are not v4l2loopback outputs.

    Returns
    -------
    list[int]
        Sorted indices (e.g. ``[0, 2]`` if ``video1`` is loopback). Empty if
        sysfs is missing or no nodes exist.
    """
    result: list[int] = []
    for name in _linux_list_video_basenames():
        if linux_is_v4l2_loopback(name):
            continue
        result.append(int(name[5:]))
    return result


def _device_basename_for_sysfs(device: int | str) -> str | None:
    if isinstance(device, int):
        return f"video{device}"
    s = str(device)
    if s.startswith("/dev/video"):
        return Path(s).name
    return None


def _resolve_candidates(camera: str) -> list[int | str]:
    """
    Build ordered capture candidates for ``open_webcam``.

    ``auto`` on Linux prefers sysfs so v4l2loopback output nodes are never tried.
    """
    key = camera.strip().lower()
    if key == "auto":
        if sys.platform == "linux":
            basenames = _linux_list_video_basenames()
            indices = linux_non_loopback_indices()
            if basenames and not indices:
                logger.error(
                    "Only v4l2loopback (virtual output) V4L2 devices were found; "
                    "there is no physical webcam to capture from. If you need "
                    "loopback for --virtual-cam, load it on a dedicated node so "
                    "your real camera still appears, e.g.\n"
                    "  sudo modprobe v4l2loopback devices=1 video_nr=10 "
                    'card_label="LaughingMan"',
                )
                raise typer.Exit(code=1)
            if indices:
                return cast(list[int | str], indices)
            logger.warning(
                "Could not enumerate V4L2 devices from sysfs; falling back to "
                "camera index 0. If the app crashes or misbehaves, set --camera "
                "to your real webcam (e.g. /dev/video1) and load v4l2loopback "
                "on a high video_nr (e.g. video_nr=10).",
            )
            return [0]
        return [0]
    dev = parse_camera_token(camera)
    return [dev]


def _refuse_loopback_before_open(device: int | str) -> None:
    if sys.platform != "linux":
        return
    base = _device_basename_for_sysfs(device)
    if base is None:
        return
    if linux_is_v4l2_loopback(base):
        logger.error(
            "Refusing to open {} as a capture device: it is a v4l2loopback "
            "(virtual output) node. That is not a webcam; opening it for read "
            "can crash OpenCV.\n\n"
            "Use --camera auto (default on Linux), point --camera at your real "
            "webcam (often /dev/video1 when loopback took video0), or load the "
            "module on a dedicated number, e.g.\n"
            "  sudo modprobe v4l2loopback devices=1 video_nr=10 "
            'card_label="LaughingMan"',
            device if isinstance(device, str) else f"index {device}",
        )
        raise typer.Exit(code=1)


def _validate_capture(cap: cv2.VideoCapture) -> bool:
    """Return True if one frame can be read with non-zero dimensions."""
    prev_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    try:
        ok, frame = cap.read()
    finally:
        cv2.utils.logging.setLogLevel(prev_level)
    if not ok or frame is None or frame.size == 0:
        return False
    h, w = frame.shape[:2]
    if w < 2 or h < 2:
        return False
    return True


def _try_open_one(device: int | str) -> cv2.VideoCapture | None:
    _refuse_loopback_before_open(device)
    prev_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    cap: cv2.VideoCapture | None = None
    try:
        cap = cv2.VideoCapture(device)
    except Exception:
        cap = None
    finally:
        cv2.utils.logging.setLogLevel(prev_level)
    if cap is None or not cap.isOpened():
        return None
    if not _validate_capture(cap):
        cap.release()
        return None
    return cap


def open_webcam(camera: str = DEFAULT_CAMERA) -> cv2.VideoCapture:
    """
    Open a camera by ``--camera`` value.

    On Linux, ``auto`` skips v4l2loopback devices (virtual outputs) so index 0
    is not mistaken for a webcam after ``modprobe v4l2loopback``.

    Temporarily silences OpenCV's verbose FFmpeg/backend logging on failure so
    the process can exit with a single clear message instead of a multi-line
    OpenCV stack trace.

    Parameters
    ----------
    camera
        ``auto``, a decimal index, or ``/dev/video*``.

    Returns
    -------
    cv2.VideoCapture
        An opened capture device that has returned at least one valid frame.

    Raises
    ------
    typer.Exit
        If no usable device is found or the user chose a loopback node.
    """
    key = camera.strip() or "auto"
    candidates = _resolve_candidates(key)

    for device in candidates:
        cap = _try_open_one(device)
        if cap is not None:
            if key.lower() == "auto":
                logger.info("Using camera {}.", device)
            return cap

    logger.error(camera_open_failed_message(key))
    raise typer.Exit(code=1)


def camera_open_failed_message(camera: str, detail: str | None = None) -> str:
    """
    Build a user-facing message when the webcam cannot be opened.

    Parameters
    ----------
    camera
        Value passed to ``open_webcam`` (e.g. ``auto`` or ``1``).
    detail
        Optional exception text from OpenCV (shown on one line when present).

    Returns
    -------
    str
        Full message for stderr.
    """
    lines = [
        f"ERROR: Could not open a working webcam (--camera {camera!r}).",
        "",
        "No usable video capture device was found. Typical causes:",
        "  • No camera is connected, or the wrong device was chosen.",
        "  • Another application is using the camera; close it and try again.",
        "  • On Linux, v4l2loopback may own /dev/video0; use --camera auto "
        "(default) or --camera /dev/video1, or load the module with "
        "video_nr=10 so your real webcam stays on a low index.",
        "  • On Linux, you may need permission to access /dev/video* (e.g. be "
        'in the "video" group, or adjust permissions).',
    ]
    if detail:
        lines.extend(["", f"Technical detail: {detail}"])
    return "\n".join(lines)
