#
# laughing-man.py
#
# Webcam face detection with Laughing Man overlay (Ghost in the Shell style).
# Modern stack: OpenCV (cv2), NumPy, Pillow.
#
# Original: pyOpenCV + PIL, Jouni Paulus, 2010.
#

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw

# Face detection (bundled with opencv-python)
CASCADE_NAME = "haarcascade_frontalface_alt.xml"

# Overlay assets: static logo + rotating text plate
STABLE_IMAGE_NAME = "limg.png"
ROT_IMAGE_NAME = "ltext.png"

# Ignore very small detections
MIN_FACE_SIZE = (100, 100)

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
    cascade: cv2.CascadeClassifier,
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
    cascade
        Loaded Haar cascade classifier.
    overlay_rgb
        RGB uint8 image, same size as the face ROI when drawn.
    mask_l
        Single-channel uint8 mask (0–255), same spatial size as ``overlay_rgb``.
    state
        Tracks the smoothed ROI between frames.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    frame_h, frame_w = frame.shape[:2]

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        flags=cv2.CASCADE_SCALE_IMAGE | cv2.CASCADE_FIND_BIGGEST_OBJECT,
        minSize=MIN_FACE_SIZE,
    )

    r: tuple[int, int, int, int] | None = None

    if len(faces) == 0:
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
        x, y, w, h = (int(v) for v in faces[0])
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
    cascade_path = Path(cv2.data.haarcascades) / CASCADE_NAME
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        print(f"ERROR: Could not load classifier cascade from {cascade_path}", file=sys.stderr)
        sys.exit(1)

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

    print("Press any key in the window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                break

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

            detect_and_draw(frame, cascade, overlay_rgb, mask_l, roi_state)

            rot_angle = (rot_angle + ROT_RESO) % 360
            if cv2.waitKey(KEY_WAIT_DELAY_MS) >= 0:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
