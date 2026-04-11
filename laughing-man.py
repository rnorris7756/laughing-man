#
# laughing-man.py
#
# Webcam face detection with Laughing Man overlay (Ghost in the Shell style).
# Uses MediaPipe BlazeFace for detection (better pose tolerance than Haar cascades),
# plus OpenCV display and Pillow for the rotating logo composite.
#
# Original: pyOpenCV + PIL, Jouni Paulus, 2010.
#

from __future__ import annotations

import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageChops, ImageDraw

# -----------------------------------------------------------------------------
# Why not Haar cascades?
# Haar (e.g. haarcascade_frontalface_alt) matches mostly upright, frontal faces.
# For webcams, MediaPipe Face Detector (BlazeFace) is a better default: it is
# CNN-based, runs in real time on CPU, and tolerates moderate head yaw/pitch.
# Alternatives: YuNet/RetinaFace in OpenCV DNN, or face_landmarker for tracking
# landmarks (heavier). See: https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector
# -----------------------------------------------------------------------------

# BlazeFace short-range model (typical webcam distance). Cached on first run.
BLAZE_FACE_SHORT_RANGE_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
MODEL_ENV = "LAUGHING_MAN_FACE_MODEL"

# Overlay assets: static logo + rotating text plate
STABLE_IMAGE_NAME = "limg.png"
ROT_IMAGE_NAME = "ltext.png"

# Ignore very small detections (pixels)
MIN_FACE_SIZE = (100, 100)

# BlazeFace: confidence for a box to count (lower = more recall, more false positives)
MIN_DETECTION_CONFIDENCE = 0.45
MIN_SUPPRESSION_THRESHOLD = 0.3

# Degrees; cache has 360 / ROT_RESO entries
ROT_RESO = 5

# First-order low-pass on ROI: new = roi_lambda * prev + (1 - roi_lambda) * found
ROI_LAMBDA = 0.95
# When face is lost, shrink ROI until it disappears
FADEOUT_LAMBDA = 0.99
FADEOUT_LIM = 50

# Expand detected face box slightly
ROI_SCALER = 1.3

KEY_WAIT_DELAY_MS = 25
MAIN_WIN_NAME = "Laughing Man (OpenCV)"
TMP_OFFSET = 0.1


@dataclass
class RoiState:
    """Smoothed face bounding box in float coordinates."""

    prev: tuple[float, float, float, float] | None = None


def _cache_dir() -> Path:
    """Return XDG cache dir for downloaded models."""
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".cache"
    return root / "laughing-man"


def _model_path() -> Path:
    """Path to BlazeFace Tflite; override with LAUGHING_MAN_FACE_MODEL."""
    env = os.environ.get(MODEL_ENV, "").strip()
    if env:
        return Path(env)
    return _cache_dir() / "blaze_face_short_range.tflite"


def _ensure_blaze_face_model(path: Path) -> None:
    """Download BlazeFace if missing."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    print(f"Downloading face detector model to {path} ...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(BLAZE_FACE_SHORT_RANGE_URL, tmp)
    except OSError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        print(f"ERROR: Could not download model: {e}", file=sys.stderr)
        sys.exit(1)
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


def detect_and_draw(
    frame: np.ndarray,
    detector: vision.FaceDetector,
    timestamp_ms: int,
    overlay_rgb: np.ndarray,
    mask_l: np.ndarray,
    state: RoiState,
) -> None:
    """
    Detect the largest face, smooth the ROI, and alpha-composite the overlay.

    Parameters
    ----------
    frame
        BGR image (modified in place in the face region).
    detector
        MediaPipe FaceDetector (VIDEO running mode).
    timestamp_ms
        Monotonic timestamp for ``detect_for_video`` (must not decrease).
    overlay_rgb
        RGB uint8 image, same size as the face ROI when drawn.
    mask_l
        Single-channel uint8 mask (0–255), same spatial size as ``overlay_rgb``.
    state
        Tracks the smoothed ROI between frames.
    """
    frame_h, frame_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    min_w, min_h = MIN_FACE_SIZE
    face = _pick_largest_face(list(result.detections), min_w, min_h)

    r: tuple[int, int, int, int] | None = None

    if face is None:
        if state.prev is not None:
            px, py, pw, ph = state.prev
            roi_cx = px + pw / 2.0
            roi_cy = py + ph / 2.0
            new_w = FADEOUT_LAMBDA * pw
            new_h = FADEOUT_LAMBDA * ph
            new_x = roi_cx - new_w / 2.0
            new_y = roi_cy - new_h / 2.0
            state.prev = (new_x, new_y, new_w, new_h)
            ix, iy = int(round(new_x)), int(round(new_y))
            iw, ih = int(round(new_w)), int(round(new_h))
            if iw < FADEOUT_LIM or ih < FADEOUT_LIM:
                state.prev = None
            else:
                r = (ix, iy, iw, ih)
    else:
        x, y, w, h = face
        if state.prev is None:
            state.prev = (float(x), float(y), float(w), float(h))
            r = (x, y, w, h)
        else:
            px, py, pw, ph = state.prev
            new_w = ROI_LAMBDA * pw + (1.0 - ROI_LAMBDA) * w * ROI_SCALER
            new_h = ROI_LAMBDA * ph + (1.0 - ROI_LAMBDA) * h * ROI_SCALER
            roi_cx = ROI_LAMBDA * (px + pw / 2.0) + (1.0 - ROI_LAMBDA) * (x + w / 2.0)
            roi_cy = ROI_LAMBDA * (py + ph / 2.0) + (1.0 - ROI_LAMBDA) * (y + h / 2.0)
            new_x = roi_cx - new_w / 2.0
            new_y = roi_cy - new_h / 2.0
            new_x, new_y, new_w, new_h = _clamp_roi(
                new_x, new_y, new_w, new_h, frame_w, frame_h
            )
            state.prev = (new_x, new_y, new_w, new_h)
            ix, iy = int(round(new_x)), int(round(new_y))
            iw, ih = int(round(new_w)), int(round(new_h))
            if iw < FADEOUT_LIM or ih < FADEOUT_LIM:
                state.prev = None
            else:
                r = (ix, iy, iw, ih)

    if r is None:
        cv2.imshow(MAIN_WIN_NAME, frame)
        return

    rx, ry, rw, rh = r
    face_roi = frame[ry : ry + rh, rx : rx + rw]
    if face_roi.size == 0:
        cv2.imshow(MAIN_WIN_NAME, frame)
        return

    ov = cv2.resize(overlay_rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
    mk = cv2.resize(mask_l, (rw, rh), interpolation=cv2.INTER_LINEAR)

    overlay_bgr = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR).astype(np.float32)
    mask_f = (mk.astype(np.float32) / 255.0)[..., np.newaxis]
    base = face_roi.astype(np.float32)
    blended = base * (1.0 - mask_f) + overlay_bgr * mask_f
    face_roi[:] = np.clip(blended, 0, 255).astype(np.uint8)

    cv2.imshow(MAIN_WIN_NAME, frame)


def main() -> None:
    """Run webcam capture with Laughing Man overlay."""
    model_path = _model_path()
    _ensure_blaze_face_model(model_path)

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_suppression_threshold=MIN_SUPPRESSION_THRESHOLD,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera 0", file=sys.stderr)
        sys.exit(1)

    cv2.namedWindow(MAIN_WIN_NAME, cv2.WINDOW_AUTOSIZE)

    st_img = Image.open(STABLE_IMAGE_NAME)
    rot_img = Image.open(ROT_IMAGE_NAME)
    st_img.load()
    rot_img.load()

    st_bands = st_img.split()
    rot_bands = rot_img.split()
    st_alpha = st_bands[3]
    rot_alpha = rot_bands[3]
    rot_angle = 0

    im_sz = st_img.size
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
    min_dim = min(input_h, input_w)
    mask_img = mask_img.resize((min_dim, min_dim))

    n_cache = 360 // ROT_RESO
    img_cache: list[list[Image.Image]] = [[] for _ in range(n_cache)]

    roi_state = RoiState()
    t0 = time.monotonic()

    print("Press any key in the window to quit.")

    try:
        with vision.FaceDetector.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    break

                timestamp_ms = int((time.monotonic() - t0) * 1000.0)

                cache_idx = rot_angle // ROT_RESO
                if not img_cache[cache_idx]:
                    print(f"Computing rotated overlay for angle {rot_angle}°.")
                    comb_img = Image.new("RGB", st_img.size)
                    draw_img = ImageDraw.Draw(comb_img)
                    draw_img.ellipse(
                        (
                            im_sz[0] * TMP_OFFSET,
                            im_sz[1] * TMP_OFFSET,
                            im_sz[0] * (1 - TMP_OFFSET),
                            im_sz[1] * (1 - TMP_OFFSET),
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
                    tmp_img = tmp_img.resize((min_dim, min_dim))
                    img_cache[cache_idx].append(tmp_img)
                else:
                    tmp_img = img_cache[cache_idx][0]

                overlay_rgb = np.asarray(tmp_img.convert("RGB"))
                mask_l = np.asarray(mask_img)

                detect_and_draw(
                    frame,
                    detector,
                    timestamp_ms,
                    overlay_rgb,
                    mask_l,
                    roi_state,
                )

                rot_angle = (rot_angle + ROT_RESO) % 360
                if cv2.waitKey(KEY_WAIT_DELAY_MS) >= 0:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
