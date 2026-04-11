#
# Laughing Man webcam overlay (Ghost in the Shell style).
# MediaPipe BlazeFace + OpenCV + Pillow.
#
# Original face-detection demo: Jouni Paulus, 2010 (pyOpenCV + PIL).
#

from __future__ import annotations

import os
import select
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
import typer
from loguru import logger
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageChops, ImageDraw
from pyvirtualcam import PixelFormat

# Haar cascades are poor for pose; BlazeFace is a better default. See:
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector

BLAZE_FACE_SHORT_RANGE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
BLAZE_FACE_FULL_RANGE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
)

MODEL_ENV = "LAUGHING_MAN_FACE_MODEL"

STABLE_IMAGE_NAME = "limg.png"
ROT_IMAGE_NAME = "ltext.png"

MIN_FACE_SIZE = (100, 100)
MIN_DETECTION_CONFIDENCE = 0.45
MIN_SUPPRESSION_THRESHOLD = 0.3

ROT_RESO = 5
DEFAULT_ROI_LAMBDA = 0.55
DEFAULT_SIZE_LAMBDA = 0.55
FADEOUT_LIM = 50
# Full-frame privacy blur at fade strength 1.0 (Gaussian sigma, px).
PRIVACY_BLUR_MAX_SIGMA = 28.0
# Consecutive no-face frames required before privacy blur (debounce detector flicker).
DEFAULT_NO_FACE_BLUR_FRAMES = 3
# Face box → overlay scale (1.3 baseline × 1.08).
ROI_SCALER = 1.3 * 1.08
# Shift overlay up by this fraction of **detector** face height (y grows downward).
OVERLAY_VERTICAL_SHIFT = 0.10

KEY_WAIT_DELAY_MS = 25
MAIN_WIN_NAME = "Laughing Man (OpenCV)"
TMP_OFFSET = 0.1
# Step for interactive --roi-lambda / --size-lambda tuning (terminal or preview arrows).
LAMBDA_TUNE_STEP = 0.05
# waitKeyEx codes for arrow keys: GTK/Qt (Linux, macOS) and common Windows highgui values.
_ARROW_KEYS_ROI_UP = frozenset({65362, 2490368})
_ARROW_KEYS_ROI_DOWN = frozenset({65364, 2621440})
_ARROW_KEYS_SIZE_LEFT = frozenset({65361, 2424832})
_ARROW_KEYS_SIZE_RIGHT = frozenset({65363, 2555904})
# Preview window quit chords delivered as key events (see ``_should_quit_preview``).
_QUIT_PREVIEW_KEY_CODES = frozenset(
    {
        3,  # Ctrl+C (ETX) when the preview window has focus (terminal Ctrl+C uses SIGINT)
        17,  # Ctrl+Q (ASCII control-Q; common on GTK/X11 and Windows highgui)
    }
)

CAMERA_INDEX = 0


def _configure_logging(*, debug: bool) -> None:
    """
    Configure loguru: single stderr sink, INFO unless ``debug`` is True.

    Parameters
    ----------
    debug
        If True, emit DEBUG-level messages (e.g. overlay prefill diagnostics).
    """
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")


def _open_webcam(camera_index: int) -> cv2.VideoCapture:
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
            logger.error(_camera_open_failed_message(camera_index, detail=str(e)))
            raise typer.Exit(code=1) from e
    finally:
        cv2.utils.logging.setLogLevel(prev_level)

    if not cap.isOpened():
        logger.error(_camera_open_failed_message(camera_index))
        raise typer.Exit(code=1)
    return cap


def _camera_open_failed_message(camera_index: int, detail: str | None = None) -> str:
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


app = typer.Typer(
    help="Webcam Laughing Man face overlay (MediaPipe BlazeFace + OpenCV).",
    add_completion=False,
    no_args_is_help=True,
)


@dataclass
class RoiState:
    """Smoothed face bounding box in float coordinates."""

    prev: tuple[float, float, float, float] | None = None
    last_face_h: float | None = None
    no_face_streak: int = 0


def _cache_dir() -> Path:
    """Return XDG cache dir for downloaded models."""
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".cache"
    return root / "laughing-man"


def _default_model_path(full_range: bool) -> Path:
    """Default cache path for the bundled BlazeFace variant."""
    name = (
        "blaze_face_full_range.tflite" if full_range else "blaze_face_short_range.tflite"
    )
    return _cache_dir() / name


def _resolve_model(full_range: bool) -> tuple[Path, str | None]:
    """
    Return (path, download_url or None).

    If ``LAUGHING_MAN_FACE_MODEL`` is set, that path is used and no download URL
    is applied (the file must already exist).
    """
    env = os.environ.get(MODEL_ENV, "").strip()
    if env:
        return Path(env), None
    url = BLAZE_FACE_FULL_RANGE_URL if full_range else BLAZE_FACE_SHORT_RANGE_URL
    return _default_model_path(full_range), url


def _ensure_blaze_face_model(path: Path, url: str | None) -> None:
    """Download BlazeFace if a URL is known and the file is missing."""
    if path.exists():
        return
    if url is None:
        logger.error(
            "Face model not found at {} ({} is set; place a .tflite there).",
            path,
            MODEL_ENV,
        )
        raise typer.Exit(code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    logger.info("Downloading face detector model to {} ...", path)
    try:
        urllib.request.urlretrieve(url, tmp)
    except OSError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        logger.error("Could not download model: {}", e)
        raise typer.Exit(code=1) from e
    tmp.replace(path)


def _pick_largest_face(
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
    return _pick_largest_face(list(result.detections), min_w, min_h)


def _lambda_deltas_from_arrow_key(key: int) -> tuple[float, float] | None:
    """
    Map an OpenCV ``waitKeyEx`` code to tuning deltas for roi-lambda and size-lambda.

    Up/down adjust roi-lambda; left/right adjust size-lambda. Returns ``None`` if
    ``key`` is not a recognized arrow code.

    Parameters
    ----------
    key
        Value from :func:`cv2.waitKeyEx` (non-negative when a key was pressed).

    Returns
    -------
    tuple[float, float] | None
        ``(delta_roi_lambda, delta_size_lambda)``, each 0 if that axis does not
        apply, or ``None`` if ``key`` is not an arrow.
    """
    if key in _ARROW_KEYS_ROI_UP:
        return (LAMBDA_TUNE_STEP, 0.0)
    if key in _ARROW_KEYS_ROI_DOWN:
        return (-LAMBDA_TUNE_STEP, 0.0)
    if key in _ARROW_KEYS_SIZE_LEFT:
        return (0.0, -LAMBDA_TUNE_STEP)
    if key in _ARROW_KEYS_SIZE_RIGHT:
        return (0.0, LAMBDA_TUNE_STEP)
    return None


def _stdin_interactive_tuning_available() -> bool:
    """
    Return True if arrow / quit tuning can be read from the controlling terminal.

    Requires an interactive TTY on stdin and a supported platform API
    (POSIX ``termios`` or Windows ``msvcrt``).
    """
    if not sys.stdin.isatty():
        return False
    if sys.platform == "win32":
        return True
    try:
        import termios  # noqa: F401
    except ImportError:
        return False
    return True


def _terminal_stdin_tune_loop(
    listener_stop: threading.Event,
    user_quit: threading.Event,
    apply_deltas: Callable[[float, float], None],
) -> None:
    """
    Background loop: read arrow keys and Ctrl+Q from the terminal.

    Parameters
    ----------
    listener_stop
        When set, exit the loop and return (main loop is shutting down).
    user_quit
        Set when the user presses Ctrl+Q (ASCII DC1) in the terminal reader.
    apply_deltas
        ``(delta_roi_lambda, delta_size_lambda)`` — same semantics as
        :func:`_lambda_deltas_from_arrow_key` results.
    """
    try:
        if sys.platform == "win32":
            _windows_console_tune_loop(listener_stop, user_quit, apply_deltas)
        else:
            _posix_stdin_tune_loop(listener_stop, user_quit, apply_deltas)
    except Exception as e:
        logger.warning("Terminal key listener stopped ({}).", e)


def _posix_stdin_tune_loop(
    listener_stop: threading.Event,
    user_quit: threading.Event,
    apply_deltas: Callable[[float, float], None],
) -> None:
    """POSIX ``termios`` cbreak reader for CSI/SS3 arrows and Ctrl+Q."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not listener_stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.12)
            if not r:
                continue
            b0 = os.read(fd, 1)
            if not b0:
                break
            if b0 == b"\x1b":
                r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r2:
                    continue
                b1 = os.read(fd, 1)
                if b1 == b"[":
                    r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not r3:
                        continue
                    b2 = os.read(fd, 1)
                    if b2 == b"A":
                        apply_deltas(LAMBDA_TUNE_STEP, 0.0)
                    elif b2 == b"B":
                        apply_deltas(-LAMBDA_TUNE_STEP, 0.0)
                    elif b2 == b"C":
                        apply_deltas(0.0, LAMBDA_TUNE_STEP)
                    elif b2 == b"D":
                        apply_deltas(0.0, -LAMBDA_TUNE_STEP)
                elif b1 == b"O":
                    r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not r3:
                        continue
                    b2 = os.read(fd, 1)
                    if b2 == b"A":
                        apply_deltas(LAMBDA_TUNE_STEP, 0.0)
                    elif b2 == b"B":
                        apply_deltas(-LAMBDA_TUNE_STEP, 0.0)
                    elif b2 == b"C":
                        apply_deltas(0.0, LAMBDA_TUNE_STEP)
                    elif b2 == b"D":
                        apply_deltas(0.0, -LAMBDA_TUNE_STEP)
                continue
            if b0 == b"\x11":
                user_quit.set()
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _windows_console_tune_loop(
    listener_stop: threading.Event,
    user_quit: threading.Event,
    apply_deltas: Callable[[float, float], None],
) -> None:
    """Windows console: extended-prefix arrow keys and Ctrl+Q."""
    import msvcrt

    while not listener_stop.is_set():
        if msvcrt.kbhit():
            c = msvcrt.getch()
            if c in (b"\xe0", b"\x00"):
                c2 = msvcrt.getch()
                if c2 == b"H":
                    apply_deltas(LAMBDA_TUNE_STEP, 0.0)
                elif c2 == b"P":
                    apply_deltas(-LAMBDA_TUNE_STEP, 0.0)
                elif c2 == b"K":
                    apply_deltas(0.0, -LAMBDA_TUNE_STEP)
                elif c2 == b"M":
                    apply_deltas(0.0, LAMBDA_TUNE_STEP)
            elif c == b"\x11":
                user_quit.set()
                return
        else:
            listener_stop.wait(0.05)


def _should_quit_preview(key: int) -> bool:
    """
    Return True if this OpenCV ``waitKeyEx`` code should exit the preview loop.

    Handles Ctrl+C and Ctrl+Q as control-character key events when the preview
    window has focus (ASCII 3 and 17). When the terminal has focus, Ctrl+C
    still stops the process via ``KeyboardInterrupt``. Cmd+Q on macOS depends
    on the OpenCV GUI backend and may not be reported as a distinct code; use
    Ctrl+Q or terminal Ctrl+C if needed. Other keys are ignored unless handled
    elsewhere (e.g. arrow tuning).

    Parameters
    ----------
    key
        Value from :func:`cv2.waitKeyEx`, or ``-1`` if no key was pressed.

    Returns
    -------
    bool
        True if the user chose to quit from the preview window.
    """
    return key >= 0 and key in _QUIT_PREVIEW_KEY_CODES


def _clamp_roi(
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


def _apply_privacy_blur(frame: np.ndarray, amount: float) -> None:
    """
    Blend ``frame`` toward a heavily Gaussian-blurred copy (full-frame privacy).

    Parameters
    ----------
    frame
        BGR uint8 image, updated in place.
    amount
        Blend weight for the blurred layer in ``[0, 1]`` (0 = unchanged, 1 = full blur).
    """
    if amount <= 1e-6:
        return
    blurred = cv2.GaussianBlur(frame, (0, 0), PRIVACY_BLUR_MAX_SIGMA)
    cv2.addWeighted(frame, 1.0 - amount, blurred, amount, 0, dst=frame)


def _resize_overlay_to_face_roi(
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


def smooth_and_draw(
    frame: np.ndarray,
    raw_face: tuple[int, int, int, int] | None,
    state: RoiState,
    overlay_rgb: np.ndarray,
    mask_l: np.ndarray,
    *,
    center_lambda: float,
    size_lambda: float,
    no_face_blur_frames: int,
    show_preview: bool,
    overlay_scale: float,
) -> None:
    """
    Apply full-frame privacy blur after consecutive no-face frames, temporal
    smoothing to the face box when a face is present, then alpha-composite the
    overlay on the face (or on the last known box during the debounce window).

    Parameters
    ----------
    frame
        BGR image (modified in place in the face region).
    raw_face
        Largest face box from BlazeFace this frame, or None if no qualifying face.
    state
        Tracks the smoothed ROI between frames.
    overlay_rgb
        RGB uint8 image, same size as the face ROI when drawn.
    mask_l
        Single-channel uint8 mask (0–255), same spatial size as ``overlay_rgb``.
    center_lambda
        Low-pass on the **horizontal** center of the box only (higher = stickier
        left-right position). Vertical placement follows the current detection each
        frame so tuning this does not shift the overlay up or down.
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
    """
    frame_h, frame_w = frame.shape[:2]

    if raw_face is not None:
        state.no_face_streak = 0
    else:
        state.no_face_streak += 1

    apply_blur = state.no_face_streak >= no_face_blur_frames
    _apply_privacy_blur(frame, 1.0 if apply_blur else 0.0)

    r: tuple[int, int, int, int] | None = None

    if raw_face is not None:
        x, y, w, h = raw_face
        face_h = float(h)
        state.last_face_h = face_h
        y_shift = OVERLAY_VERTICAL_SHIFT * face_h
        if state.prev is None:
            sy = float(y) - y_shift
            state.prev = (float(x), sy, float(w), float(h))
            r = (x, int(round(sy)), w, h)
        else:
            px, py, pw, ph = state.prev
            w_det = w * ROI_SCALER
            h_det = h * ROI_SCALER
            new_w = size_lambda * pw + (1.0 - size_lambda) * w_det
            new_h = size_lambda * ph + (1.0 - size_lambda) * h_det
            roi_cx = center_lambda * (px + pw / 2.0) + (1.0 - center_lambda) * (
                x + w / 2.0
            )
            roi_cy = float(y) + float(h) / 2.0
            new_x = roi_cx - new_w / 2.0
            new_y = roi_cy - new_h / 2.0 - y_shift
            new_x, new_y, new_w, new_h = _clamp_roi(
                new_x, new_y, new_w, new_h, frame_w, frame_h
            )
            state.prev = (new_x, new_y, new_w, new_h)
            ix, iy = int(round(new_x)), int(round(new_y))
            iw, ih = int(round(new_w)), int(round(new_h))
            if iw < FADEOUT_LIM or ih < FADEOUT_LIM:
                state.prev = None
                state.last_face_h = None
            else:
                r = (ix, iy, iw, ih)

    elif not apply_blur and state.prev is not None:
        px, py, pw, ph = state.prev
        new_x, new_y, new_w, new_h = _clamp_roi(px, py, pw, ph, frame_w, frame_h)
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
    ov, mk = _resize_overlay_to_face_roi(overlay_rgb, mask_l, aw, ah, overlay_scale)

    overlay_bgr = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR).astype(np.float32)
    mask_f = (mk.astype(np.float32) / 255.0)[..., np.newaxis]
    base = face_roi.astype(np.float32)
    blended = base * (1.0 - mask_f) + overlay_bgr * mask_f
    face_roi[:] = np.clip(blended, 0, 255).astype(np.uint8)

    if show_preview:
        cv2.imshow(MAIN_WIN_NAME, frame)


def _face_detector_options(
    model_path: Path, *, use_gpu: bool
) -> vision.FaceDetectorOptions:
    """Build FaceDetectorOptions for CPU or TFLite GPU delegate."""
    delegate = (
        python.BaseOptions.Delegate.GPU
        if use_gpu
        else python.BaseOptions.Delegate.CPU
    )
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


def _build_rotated_overlay_frame(
    *,
    rot_angle: int,
    st_img: Image.Image,
    st_alpha: Image.Image,
    rot_img: Image.Image,
    rot_alpha: Image.Image,
    im_sz: tuple[int, int],
    tmp_offset: float,
    min_dim: int,
    custom_static_overlay: bool = False,
) -> Image.Image:
    """
    Composite stable art, rotating text, and masks; resize to the square ROI size.

    When ``custom_static_overlay`` is True (``--image`` with a user PNG), the
    stock white-ellipse backdrop and rotating layer are skipped: the overlay is
    only ``st_img`` composited over black using ``st_alpha``, so transparent
    pixels stay dark and do not pick up a white disk behind the art.

    Parameters
    ----------
    rot_angle
        Rotation in degrees (typically a multiple of ``ROT_RESO``). Ignored when
        ``custom_static_overlay`` is True.
    st_img, st_alpha, rot_img, rot_alpha
        Source layers (``st_*`` static, ``rot_*`` rotated each frame).
    im_sz
        ``(width, height)`` of ``st_img``.
    tmp_offset
        Ellipse inset factor (same as ``TMP_OFFSET``).
    min_dim
        Edge length of the square overlay after resize.
    custom_static_overlay
        If True, build a single-layer composite for user-supplied overlay art.

    Returns
    -------
    Image.Image
        RGB ``Image`` ready for ``numpy.asarray(..., dtype=uint8)`` in the loop.
    """
    if custom_static_overlay:
        backing = Image.new("RGB", st_img.size, (0, 0, 0))
        tmp_img = Image.composite(st_img, backing, st_alpha)
        return tmp_img.resize((min_dim, min_dim))

    comb_img = Image.new("RGB", st_img.size)
    draw_img = ImageDraw.Draw(comb_img)
    draw_img.ellipse(
        (
            im_sz[0] * tmp_offset,
            im_sz[1] * tmp_offset,
            im_sz[0] * (1 - tmp_offset),
            im_sz[1] * (1 - tmp_offset),
        ),
        fill="white",
    )
    rot_nearest = rot_img.rotate(rot_angle, Image.Resampling.NEAREST)
    rot_a_nearest = rot_alpha.rotate(rot_angle, Image.Resampling.NEAREST)
    tmp_img = Image.composite(
        st_img,
        Image.composite(rot_nearest, comb_img, rot_a_nearest),
        st_alpha,
    )
    return tmp_img.resize((min_dim, min_dim))


def _load_overlay_images(
    overlay_image: Path | None,
) -> tuple[Image.Image, Image.Image]:
    """
    Load stable and rotating overlay layers.

    When ``overlay_image`` is set, that file is the stable layer (RGBA); the
    rotating layer is a same-size fully transparent image so the overlay does
    not spin arbitrary artwork.

    Parameters
    ----------
    overlay_image
        Path to a PNG/JPEG/etc. to use instead of bundled ``limg.png`` /
        ``ltext.png``. If None, bundled assets are loaded from the current
        working directory.

    Returns
    -------
    tuple[Image.Image, Image.Image]
        ``(st_img, rot_img)`` as RGBA :class:`PIL.Image.Image` instances.
    """
    if overlay_image is not None:
        p = overlay_image.expanduser().resolve()
        if not p.is_file():
            logger.error("Overlay image not found: {}", p)
            raise typer.Exit(code=1)
        try:
            st_img = Image.open(p).convert("RGBA")
        except OSError as e:
            logger.error("Could not read overlay image {}: {}", p, e)
            raise typer.Exit(code=1) from e
        st_img.load()
        rot_img = Image.new("RGBA", st_img.size, (0, 0, 0, 0))
        return st_img, rot_img

    st_img = Image.open(STABLE_IMAGE_NAME)
    rot_img = Image.open(ROT_IMAGE_NAME)
    st_img.load()
    rot_img.load()
    return st_img, rot_img


def run_overlay(
    *,
    full_range: bool,
    use_gpu: bool,
    roi_lambda: float,
    size_lambda: float,
    no_face_blur_frames: int,
    virtual_cam: bool,
    v4l2_device: str | None,
    virtual_fps: float,
    show_preview: bool,
    overlay_image: Path | None,
    overlay_scale: float,
) -> None:
    """
    Run webcam capture with Laughing Man overlay.

    Parameters
    ----------
    full_range
        If True, use BlazeFace full-range (smaller/distant faces). Otherwise
        short-range (typical desk webcam).
    use_gpu
        If True, request MediaPipe's TFLite **GPU** delegate (not the same as
        "use CUDA" in PyTorch). Falls back to CPU if creation fails.
    roi_lambda
        Temporal smoothing for **horizontal** overlay position (0 = snap to raw).
        Vertical placement follows the detector each frame.
    size_lambda
        Temporal smoothing for **box width/height** (reduces detector size jitter).
    no_face_blur_frames
        Consecutive no-face frames before privacy blur; debounces flicker.
    virtual_cam
        If True, send composited frames to a virtual camera (Linux: v4l2loopback;
        Windows/macOS: OBS Virtual Camera when available) so other apps can
        capture it.
    v4l2_device
        Path to the loopback device (e.g. ``/dev/video10``). If None, the first
        available device is used.
    virtual_fps
        Target frame rate for the virtual camera (used when ``virtual_cam``).
    show_preview
        If True, show the OpenCV preview window. Disable with ``--no-preview``
        when streaming only to the virtual device.
    overlay_image
        If set, use this file as the face overlay instead of ``limg.png`` /
        ``ltext.png`` from the current working directory.
    overlay_scale
        Scale factor for mapping overlay art onto the face ROI (see ``--scale``).
    """
    if not virtual_cam and not show_preview:
        logger.error(
            "Need at least one of --virtual-cam or preview (omit --no-preview)."
        )
        raise typer.Exit(code=1)

    model_path, model_url = _resolve_model(full_range)
    _ensure_blaze_face_model(model_path, model_url)

    delegate_label: str

    def create_face_detector() -> vision.FaceDetector:
        """Open FaceDetector, falling back from GPU to CPU on failure."""
        nonlocal delegate_label
        opts = _face_detector_options(model_path, use_gpu=use_gpu)
        try:
            delegate_label = "GPU" if use_gpu else "CPU"
            return vision.FaceDetector.create_from_options(opts)
        except Exception as e:
            if use_gpu:
                logger.warning(
                    "GPU delegate failed ({}: {}); using CPU.",
                    type(e).__name__,
                    e,
                )
                opts = _face_detector_options(model_path, use_gpu=False)
                try:
                    delegate_label = "CPU"
                    return vision.FaceDetector.create_from_options(opts)
                except Exception as e2:
                    logger.error("Could not create face detector: {}", e2)
                    raise typer.Exit(code=1) from e2
            logger.error("Could not create face detector: {}", e)
            raise typer.Exit(code=1) from e

    cap = _open_webcam(CAMERA_INDEX)

    if show_preview:
        cv2.namedWindow(MAIN_WIN_NAME, cv2.WINDOW_AUTOSIZE)

    st_img, rot_img = _load_overlay_images(overlay_image)

    st_bands = st_img.split()
    rot_bands = rot_img.split()
    st_alpha = st_bands[3]
    rot_alpha = rot_bands[3]
    rot_angle = 0

    im_sz = st_img.size
    if overlay_image is not None:
        # Respect the PNG's alpha only; do not OR in the stock HUD ellipse (that
        # would force a full-disk blend and show white behind keyed transparency).
        mask_img = ImageChops.lighter(st_alpha, rot_alpha)
    else:
        mask_img = Image.new("L", st_img.size)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.ellipse(
            (
                im_sz[0] * TMP_OFFSET,
                im_sz[1] * TMP_OFFSET,
                im_sz[0] * (1 - TMP_OFFSET),
                im_sz[1] * (1 - TMP_OFFSET),
            ),
            fill="white",
        )
        mask_img = ImageChops.lighter(ImageChops.lighter(mask_img, st_alpha), rot_alpha)

    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_fps = float(cap.get(cv2.CAP_PROP_FPS))
    stream_fps = virtual_fps if virtual_fps > 0 else (cap_fps if cap_fps > 0 else 30.0)
    min_dim = min(input_h, input_w)
    mask_img = mask_img.resize((min_dim, min_dim))

    n_cache = 360 // ROT_RESO
    img_cache: list[list[Image.Image]] = [[] for _ in range(n_cache)]

    prefill_exc: list[BaseException] = []

    def _prefill_rotated_overlay_cache() -> None:
        """Fill ``img_cache`` for all discrete rotation steps (runs in a helper thread)."""
        try:
            if overlay_image is not None:
                shared = _build_rotated_overlay_frame(
                    rot_angle=0,
                    st_img=st_img,
                    st_alpha=st_alpha,
                    rot_img=rot_img,
                    rot_alpha=rot_alpha,
                    im_sz=im_sz,
                    tmp_offset=TMP_OFFSET,
                    min_dim=min_dim,
                    custom_static_overlay=True,
                )
                for i in range(n_cache):
                    img_cache[i].append(shared)
            else:
                for i in range(n_cache):
                    angle = i * ROT_RESO
                    img_cache[i].append(
                        _build_rotated_overlay_frame(
                            rot_angle=angle,
                            st_img=st_img,
                            st_alpha=st_alpha,
                            rot_img=rot_img,
                            rot_alpha=rot_alpha,
                            im_sz=im_sz,
                            tmp_offset=TMP_OFFSET,
                            min_dim=min_dim,
                            custom_static_overlay=False,
                        )
                    )
        except BaseException as e:
            prefill_exc.append(e)

    prefill_thread = threading.Thread(
        target=_prefill_rotated_overlay_cache,
        name="LaughingManRotOverlayPrefill",
        daemon=True,
    )
    logger.debug(
        "Pre-computing {} rotated overlay steps in the background ...",
        n_cache,
    )
    prefill_thread.start()

    roi_state = RoiState()
    roi_lambda_live = roi_lambda
    size_lambda_live = size_lambda
    t0 = time.monotonic()
    last_raw_face: tuple[int, int, int, int] | None = None

    def run_capture_loop(
        detector: vision.FaceDetector,
        vcam: pyvirtualcam.Camera | None,
    ) -> None:
        nonlocal rot_angle, last_raw_face, roi_lambda_live, size_lambda_live
        prefill_thread.join()
        if prefill_exc:
            e = prefill_exc[0]
            logger.error("Rotated overlay pre-computation failed: {}", e)
            raise typer.Exit(code=1) from e

        use_terminal_tuning = _stdin_interactive_tuning_available()
        terminal_listener_stop = threading.Event()
        terminal_user_quit = threading.Event()
        term_thread: threading.Thread | None = None

        def apply_lambda_deltas(d_roi: float, d_sz: float) -> None:
            nonlocal roi_lambda_live, size_lambda_live
            if d_roi != 0.0:
                roi_lambda_live = min(1.0, max(0.0, roi_lambda_live + d_roi))
            if d_sz != 0.0:
                size_lambda_live = min(1.0, max(0.0, size_lambda_live + d_sz))
            print(
                f"roi-lambda={roi_lambda_live:.3f}  size-lambda={size_lambda_live:.3f}",
                file=sys.stderr,
                flush=True,
            )

        if use_terminal_tuning:
            logger.info(
                "Lambda tuning from this terminal: Up/Down = roi-lambda, "
                "Left/Right = size-lambda (±{}). Quit: Ctrl+Q; stop: Ctrl+C.",
                LAMBDA_TUNE_STEP,
            )
            if show_preview:
                logger.info(
                    "Keyboard handling uses this terminal; the OpenCV preview window "
                    "does not need focus for tuning or quit."
                )
            term_thread = threading.Thread(
                target=_terminal_stdin_tune_loop,
                args=(
                    terminal_listener_stop,
                    terminal_user_quit,
                    apply_lambda_deltas,
                ),
                name="LaughingManStdinTune",
                daemon=True,
            )
            term_thread.start()
        elif show_preview:
            logger.info(
                "Tuning from the preview window: Up/Down = roi-lambda, "
                "Left/Right = size-lambda (±{}). Quit: Ctrl+Q or Ctrl+C in that window; "
                "Ctrl+C in the terminal also stops.",
                LAMBDA_TUNE_STEP,
            )
        elif virtual_cam:
            logger.info("Streaming to virtual camera; press Ctrl+C to exit.")

        try:
            while True:
                try:
                    if terminal_user_quit.is_set():
                        break

                    ok, frame = cap.read()
                    if not ok or frame is None or frame.size == 0:
                        break

                    timestamp_ms = int((time.monotonic() - t0) * 1000.0)

                    cache_idx = rot_angle // ROT_RESO
                    tmp_img = img_cache[cache_idx][0]

                    overlay_rgb = np.asarray(tmp_img.convert("RGB"))
                    mask_l = np.asarray(mask_img)

                    last_raw_face = mediapipe_detect_face(
                        detector, frame, timestamp_ms
                    )

                    smooth_and_draw(
                        frame,
                        last_raw_face,
                        roi_state,
                        overlay_rgb,
                        mask_l,
                        center_lambda=roi_lambda_live,
                        size_lambda=size_lambda_live,
                        no_face_blur_frames=no_face_blur_frames,
                        show_preview=show_preview,
                        overlay_scale=overlay_scale,
                    )

                    if vcam is not None:
                        vcam.send(frame)
                        vcam.sleep_until_next_frame()

                    rot_angle = (rot_angle + ROT_RESO) % 360

                    if show_preview:
                        delay = 1 if virtual_cam else KEY_WAIT_DELAY_MS
                        key = cv2.waitKeyEx(delay)
                        if not use_terminal_tuning and key >= 0:
                            if _should_quit_preview(key):
                                break
                            deltas = _lambda_deltas_from_arrow_key(key)
                            if deltas is not None:
                                d_roi, d_sz = deltas
                                apply_lambda_deltas(d_roi, d_sz)
                except KeyboardInterrupt:
                    logger.info("Interrupted.")
                    break
        finally:
            terminal_listener_stop.set()
            if term_thread is not None:
                term_thread.join(timeout=2.0)

    try:
        with create_face_detector() as detector:
            logger.debug(
                "TFLite delegate in use: {} (requested {}).",
                delegate_label,
                "GPU" if use_gpu else "CPU",
            )
            if virtual_cam:
                try:
                    with pyvirtualcam.Camera(
                        width=input_w,
                        height=input_h,
                        fps=stream_fps,
                        fmt=PixelFormat.BGR,
                        device=v4l2_device,
                    ) as vcam:
                        logger.debug(
                            "Virtual camera: {} ({}), {}x{} @ {:.1f} fps.",
                            vcam.device,
                            vcam.backend,
                            input_w,
                            input_h,
                            stream_fps,
                        )
                        run_capture_loop(detector, vcam)
                except RuntimeError as e:
                    logger.error(
                        "Could not open a virtual camera device. On Linux you "
                        "typically need the v4l2loopback kernel module, e.g.\n"
                        "  sudo modprobe v4l2loopback devices=1 video_nr=10 "
                        'card_label="LaughingMan"\n'
                        "Details: {}",
                        e,
                    )
                    raise typer.Exit(code=1) from e
            else:
                run_capture_loop(detector, None)
    finally:
        cap.release()
        cv2.destroyAllWindows()


@app.command()
def main(
    full_range: bool = typer.Option(
        False,
        "--full-range",
        help=(
            "Use BlazeFace full-range model (better for faces that are small "
            "or far from the camera). Default is short-range (typical desk webcam)."
        ),
    ),
    gpu: bool = typer.Option(
        False,
        "--gpu",
        help=(
            "Use MediaPipe's TensorFlow Lite GPU delegate when possible. This is "
            "not the same as CUDA in PyTorch; it uses the GPU via TFLite's "
            "graphics/compute path (vendor-dependent). Falls back to CPU if "
            "GPU init fails. AMD iGPU may work if drivers expose a supported API."
        ),
    ),
    roi_lambda: float = typer.Option(
        DEFAULT_ROI_LAMBDA,
        "--roi-lambda",
        min=0.0,
        max=1.0,
        help=(
            "Low-pass on **horizontal** overlay position vs the detector (0 = snap; "
            "higher = stickier left-right). Vertical placement follows the detector "
            "each frame. Does not control box size; see --size-lambda."
        ),
    ),
    size_lambda: float = typer.Option(
        DEFAULT_SIZE_LAMBDA,
        "--size-lambda",
        min=0.0,
        max=1.0,
        help=(
            "Low-pass on overlay **width/height** vs the detector box (higher = "
            "less size jitter when pose changes). Raise above --roi-lambda if bbox "
            "size fluctuates more than you want relative to horizontal position."
        ),
    ),
    no_face_blur_frames: int = typer.Option(
        DEFAULT_NO_FACE_BLUR_FRAMES,
        "--no-face-blur-frames",
        min=1,
        help=(
            "Consecutive frames with no face before full-frame privacy blur. "
            "Until then the overlay stays at the last smoothed position (reduces flicker)."
        ),
    ),
    virtual_cam: bool = typer.Option(
        False,
        "--virtual-cam",
        help=(
            "Expose the composited video as a virtual webcam (Linux: v4l2loopback) "
            "so apps like Discord or OBS can capture it. Requires the kernel module; "
            "see --v4l2-device."
        ),
    ),
    v4l2_device: str | None = typer.Option(
        None,
        "--v4l2-device",
        help=(
            "Virtual camera device path (e.g. /dev/video10). If omitted, the first "
            "available v4l2loopback device is used."
        ),
    ),
    virtual_fps: float = typer.Option(
        30.0,
        "--virtual-fps",
        min=0.1,
        help="Target frame rate for the virtual camera (used with --virtual-cam).",
    ),
    no_preview: bool = typer.Option(
        False,
        "--no-preview",
        help=(
            "Do not open the OpenCV preview window. Use with --virtual-cam to stream "
            "only to the virtual device (quit with Ctrl+C)."
        ),
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Verbose logging (overlay prefill, TFLite delegate, virtual camera details).",
    ),
    image: Path | None = typer.Option(
        None,
        "--image",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help=(
            "PNG/JPEG/etc. to composite on the face instead of the default "
            "Laughing Man assets (limg.png / ltext.png)."
        ),
    ),
    overlay_scale: float = typer.Option(
        1.0,
        "--scale",
        min=0.05,
        max=10.0,
        help=(
            "Scale overlay art relative to the detected face box (1.0 = default). "
            "Above 1.0 zooms in on the center of the image; below 1.0 shrinks it "
            "with clear margins. Applies to any overlay, including --image."
        ),
    ),
) -> None:
    """Run the Laughing Man webcam overlay."""
    _configure_logging(debug=debug)
    run_overlay(
        full_range=full_range,
        use_gpu=gpu,
        roi_lambda=roi_lambda,
        size_lambda=size_lambda,
        no_face_blur_frames=no_face_blur_frames,
        virtual_cam=virtual_cam,
        v4l2_device=v4l2_device,
        virtual_fps=virtual_fps,
        show_preview=not no_preview,
        overlay_image=image,
        overlay_scale=overlay_scale,
    )


if __name__ == "__main__":
    app()
