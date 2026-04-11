#
# Laughing Man webcam overlay (Ghost in the Shell style).
# MediaPipe BlazeFace + OpenCV + Pillow.
#
# Original face-detection demo: Jouni Paulus, 2010 (pyOpenCV + PIL).
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
import typer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageChops, ImageDraw

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
DEFAULT_ROI_LAMBDA = 0.35
# Stronger low-pass on w/h than on center: bbox size jitters more than real head size.
DEFAULT_SIZE_LAMBDA = 0.65
DEFAULT_FADEOUT_LAMBDA = 0.99
FADEOUT_LIM = 50
ROI_SCALER = 1.3

KEY_WAIT_DELAY_MS = 25
MAIN_WIN_NAME = "Laughing Man (OpenCV)"
TMP_OFFSET = 0.1

CAMERA_INDEX = 0

app = typer.Typer(
    help="Webcam Laughing Man face overlay (MediaPipe BlazeFace + OpenCV).",
    add_completion=False,
    no_args_is_help=True,
)


@dataclass
class RoiState:
    """Smoothed face bounding box in float coordinates."""

    prev: tuple[float, float, float, float] | None = None


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
        print(
            f"ERROR: Face model not found at {path} "
            f"({MODEL_ENV} is set; place a .tflite there).",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    print(f"Downloading face detector model to {path} ...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(url, tmp)
    except OSError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        print(f"ERROR: Could not download model: {e}", file=sys.stderr)
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


def smooth_and_draw(
    frame: np.ndarray,
    raw_face: tuple[int, int, int, int] | None,
    state: RoiState,
    overlay_rgb: np.ndarray,
    mask_l: np.ndarray,
    *,
    center_lambda: float,
    size_lambda: float,
    fadeout_lambda: float,
) -> None:
    """
    Apply temporal smoothing to the face box, then alpha-composite the overlay.

    Parameters
    ----------
    frame
        BGR image (modified in place in the face region).
    raw_face
        New detection for this logical frame, or None if none / skipped.
    state
        Tracks the smoothed ROI between frames.
    overlay_rgb
        RGB uint8 image, same size as the face ROI when drawn.
    mask_l
        Single-channel uint8 mask (0–255), same spatial size as ``overlay_rgb``.
    center_lambda
        Low-pass on the **center** of the box (higher = stickier position).
    size_lambda
        Low-pass on **width and height** (higher = less size jitter from the
        detector; use greater than ``center_lambda`` when the box size flickers).
    fadeout_lambda
        Shrink factor per frame when no face is seen.
    """
    frame_h, frame_w = frame.shape[:2]

    r: tuple[int, int, int, int] | None = None

    if raw_face is None:
        if state.prev is not None:
            px, py, pw, ph = state.prev
            roi_cx = px + pw / 2.0
            roi_cy = py + ph / 2.0
            new_w = fadeout_lambda * pw
            new_h = fadeout_lambda * ph
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
        x, y, w, h = raw_face
        if state.prev is None:
            state.prev = (float(x), float(y), float(w), float(h))
            r = (x, y, w, h)
        else:
            px, py, pw, ph = state.prev
            w_det = w * ROI_SCALER
            h_det = h * ROI_SCALER
            new_w = size_lambda * pw + (1.0 - size_lambda) * w_det
            new_h = size_lambda * ph + (1.0 - size_lambda) * h_det
            roi_cx = center_lambda * (px + pw / 2.0) + (1.0 - center_lambda) * (
                x + w / 2.0
            )
            roi_cy = center_lambda * (py + ph / 2.0) + (1.0 - center_lambda) * (
                y + h / 2.0
            )
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


def run_overlay(
    *,
    full_range: bool,
    use_gpu: bool,
    detect_interval: int,
    roi_lambda: float,
    size_lambda: float,
    fadeout_lambda: float,
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
    detect_interval
        Run BlazeFace every N frames (1 = every frame). Values greater than 1
        reduce CPU/GPU load but reuse the last detection between runs, which
        can increase perceived lag.
    roi_lambda
        Temporal smoothing for the **center** of the overlay (0 = snap to raw).
    size_lambda
        Temporal smoothing for **box width/height** (reduces detector size jitter).
    fadeout_lambda
        Shrink rate for the box when the face is lost.
    """
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
                print(
                    f"GPU delegate failed ({type(e).__name__}: {e}); using CPU.",
                    file=sys.stderr,
                )
                opts = _face_detector_options(model_path, use_gpu=False)
                try:
                    delegate_label = "CPU"
                    return vision.FaceDetector.create_from_options(opts)
                except Exception as e2:
                    print(
                        f"ERROR: Could not create face detector: {e2}",
                        file=sys.stderr,
                    )
                    raise typer.Exit(code=1) from e2
            print(f"ERROR: Could not create face detector: {e}", file=sys.stderr)
            raise typer.Exit(code=1) from e

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {CAMERA_INDEX}", file=sys.stderr)
        raise typer.Exit(code=1)

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
    frame_ix = 0
    last_raw_face: tuple[int, int, int, int] | None = None

    print("Press any key in the window to quit, or Ctrl+C to exit.", file=sys.stderr)

    try:
        with create_face_detector() as detector:
            print(
                f"TFLite delegate in use: {delegate_label} "
                f"(requested {'GPU' if use_gpu else 'CPU'}).",
                file=sys.stderr,
            )
            while True:
                try:
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

                    run_detection = detect_interval <= 1 or (frame_ix % detect_interval == 0)
                    if run_detection:
                        last_raw_face = mediapipe_detect_face(
                            detector, frame, timestamp_ms
                        )

                    smooth_and_draw(
                        frame,
                        last_raw_face,
                        roi_state,
                        overlay_rgb,
                        mask_l,
                        center_lambda=roi_lambda,
                        size_lambda=size_lambda,
                        fadeout_lambda=fadeout_lambda,
                    )

                    rot_angle = (rot_angle + ROT_RESO) % 360
                    frame_ix += 1
                    if cv2.waitKey(KEY_WAIT_DELAY_MS) >= 0:
                        break
                except KeyboardInterrupt:
                    print("Interrupted.", file=sys.stderr)
                    break
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
    detect_interval: int = typer.Option(
        1,
        "--detect-interval",
        min=1,
        help=(
            "Run BlazeFace every N frames (1 = every frame). Larger values save "
            "work but reuse the last box between detections and usually feel worse."
        ),
    ),
    roi_lambda: float = typer.Option(
        DEFAULT_ROI_LAMBDA,
        "--roi-lambda",
        min=0.0,
        max=1.0,
        help=(
            "Low-pass on the overlay **center** vs the detected face center "
            "(0 = snap; higher = stickier). Does not control box size; see "
            "--size-lambda."
        ),
    ),
    size_lambda: float = typer.Option(
        DEFAULT_SIZE_LAMBDA,
        "--size-lambda",
        min=0.0,
        max=1.0,
        help=(
            "Low-pass on overlay **width/height** vs the detector box (higher = "
            "less size jitter when pose changes). Typically higher than "
            "--roi-lambda because bbox size fluctuates more than real head size."
        ),
    ),
    fadeout_lambda: float = typer.Option(
        DEFAULT_FADEOUT_LAMBDA,
        "--fadeout-lambda",
        min=0.0,
        max=1.0,
        help="Per-frame shrink of the box when no face is detected (fade out).",
    ),
) -> None:
    """Run the Laughing Man webcam overlay."""
    run_overlay(
        full_range=full_range,
        use_gpu=gpu,
        detect_interval=detect_interval,
        roi_lambda=roi_lambda,
        size_lambda=size_lambda,
        fadeout_lambda=fadeout_lambda,
    )


if __name__ == "__main__":
    app()
