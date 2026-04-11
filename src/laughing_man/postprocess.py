"""Offline image/video overlay using the same face pipeline as the webcam."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
import typer
from loguru import logger
from mediapipe.tasks.python import vision
from PIL import Image

from laughing_man.cascade import CascadedFaceBoxSource
from laughing_man.constants import KEY_WAIT_DELAY_MS, MAIN_WIN_NAME, ROT_RESO
from laughing_man.deps import PipelineDeps
from laughing_man.detection import BlazeFaceFaceBoxSource, face_detector_options
from laughing_man.model import (
    ensure_blaze_face_model,
    ensure_yunet_model,
    resolve_model,
    resolve_yunet_model,
)
from laughing_man.overlay import (
    load_overlay_images,
    make_overlay_mask_resized,
    prefill_rotated_overlay_cache_inplace,
)
from laughing_man.privacy import GaussianBlurPrivacy
from laughing_man.protocols import FaceBoxSource
from laughing_man.roi import RoiState, smooth_and_draw
from laughing_man.yunet_face import YuNetFaceBoxSource, create_yunet_detector

_CV2 = cast(Any, cv2)

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


def infer_media_kind(path: Path) -> Literal["image", "video"]:
    """
    Classify ``path`` as a still image or a video file.

    Parameters
    ----------
    path
        Input file to inspect.

    Returns
    -------
    Literal["image", "video"]
        ``image`` when the suffix is a common raster type; otherwise the path is
        probed with OpenCV :class:`cv2.VideoCapture`.

    Raises
    ------
    typer.Exit
        If the file cannot be read as either kind.
    """
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None and frame.size > 0:
            return "video"
    logger.error("Could not open as image or video: {}", path.resolve())
    raise typer.Exit(code=1)


def default_output_path(input_path: Path, kind: Literal["image", "video"]) -> Path:
    """
    Default write path next to the input (``*_overlay``).

    Parameters
    ----------
    input_path
        Source file.
    kind
        Whether the output should be a raster still or an MP4 video.

    Returns
    -------
    Path
        Sibling path with ``_overlay`` in the stem.
    """
    stem = input_path.stem
    if kind == "video":
        return input_path.with_name(f"{stem}_overlay.mp4")
    return input_path.with_name(f"{stem}_overlay{input_path.suffix}")


def run_postprocess(
    *,
    input_path: Path,
    output_path: Path | None,
    overlay_image: Path | None,
    overlay_scale: float,
    full_range: bool,
    use_gpu: bool,
    roi_lambda: float,
    size_lambda: float,
    no_face_blur_frames: int,
    face_backend: Literal["blaze", "yunet"],
    cascade_margin: float,
    roi_motion: Literal["ema", "kalman", "kalman_flow"],
    preview: bool,
) -> None:
    """
    Apply the Laughing Man overlay to each frame of an image or video file.

    Uses the same overlay cache, face detection backends, ROI smoothing, and
    privacy pass as :func:`laughing_man.run.run_overlay`.

    Parameters
    ----------
    input_path
        Raster image or video readable by OpenCV.
    output_path
        File to write. If None, ``default_output_path`` is used.
    overlay_image
        Optional overlay art (same as ``--image`` on the webcam command).
    overlay_scale
        Relative scale of overlay vs face box.
    full_range
        BlazeFace full-range model when True.
    use_gpu
        Request TFLite GPU delegate for BlazeFace when True.
    roi_lambda, size_lambda
        Temporal smoothing on ROI center and size.
    no_face_blur_frames
        Consecutive no-face frames before full-frame privacy blur.
    face_backend
        ``blaze`` or ``yunet``.
    cascade_margin
        YuNet cascade crop expansion (ignored for BlazeFace).
    roi_motion
        ROI temporal model (EMA, Kalman, or Kalman + optical flow).
    preview
        If True, show an OpenCV window while processing.
    """
    kind = infer_media_kind(input_path)
    out = output_path if output_path is not None else default_output_path(input_path, kind)

    frame0: np.ndarray | None = None
    if kind == "image":
        frame0 = cv2.imread(str(input_path))
        if frame0 is None or frame0.size == 0:
            logger.error("Could not read image: {}", input_path.resolve())
            raise typer.Exit(code=1)
        input_h, input_w = frame0.shape[:2]
    else:
        cap_probe = cv2.VideoCapture(str(input_path))
        if not cap_probe.isOpened():
            logger.error("Could not open video: {}", input_path.resolve())
            raise typer.Exit(code=1)
        input_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if input_w <= 0 or input_h <= 0:
            ok, probe = cap_probe.read()
            if not ok or probe is None:
                logger.error("Could not read video frames: {}", input_path.resolve())
                cap_probe.release()
                raise typer.Exit(code=1)
            input_h, input_w = probe.shape[:2]
        cap_probe.release()

    st_img, rot_img = load_overlay_images(overlay_image)
    min_dim = min(input_h, input_w)
    mask_img = make_overlay_mask_resized(overlay_image, st_img, rot_img, min_dim)
    mask_l = np.asarray(mask_img)

    n_cache = 360 // ROT_RESO
    img_cache: list[list[Image.Image]] = [[] for _ in range(n_cache)]
    try:
        prefill_rotated_overlay_cache_inplace(
            img_cache,
            overlay_image=overlay_image,
            st_img=st_img,
            rot_img=rot_img,
            min_dim=min_dim,
        )
    except BaseException as e:
        logger.error("Overlay pre-computation failed: {}", e)
        raise typer.Exit(code=1) from e

    deps = PipelineDeps(privacy=GaussianBlurPrivacy())
    roi_state = RoiState()

    blaze_model_path: Path | None = None
    yunet_path: Path | None = None
    delegate_label: str = "CPU"

    if face_backend == "blaze":
        blaze_model_path, blaze_url = resolve_model(full_range)
        ensure_blaze_face_model(blaze_model_path, blaze_url)

        def create_face_detector() -> vision.FaceDetector:
            """Open FaceDetector, falling back from GPU to CPU on failure."""
            nonlocal delegate_label
            assert blaze_model_path is not None
            opts = face_detector_options(blaze_model_path, use_gpu=use_gpu)
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
                    opts = face_detector_options(blaze_model_path, use_gpu=False)
                    try:
                        delegate_label = "CPU"
                        return vision.FaceDetector.create_from_options(opts)
                    except Exception as e2:
                        logger.error("Could not create face detector: {}", e2)
                        raise typer.Exit(code=1) from e2
                logger.error("Could not create face detector: {}", e)
                raise typer.Exit(code=1) from e

    elif face_backend == "yunet":
        if full_range:
            logger.warning("--full-range applies to BlazeFace only; ignoring for YuNet.")
        yunet_path, yunet_url = resolve_yunet_model()
        ensure_yunet_model(yunet_path, yunet_url)
    else:
        logger.error("Unknown face backend: {}", face_backend)
        raise typer.Exit(code=1)

    if preview:
        cv2.namedWindow(MAIN_WIN_NAME, cv2.WINDOW_AUTOSIZE)

    def process_with_source(face_source: FaceBoxSource) -> None:
        rot_angle = 0
        t0 = time.monotonic()
        frame_idx = 0
        writer: cv2.VideoWriter | None = None
        cap: cv2.VideoCapture | None = None
        fps = 30.0

        try:
            if kind == "video":
                cap = cv2.VideoCapture(str(input_path))
                if not cap.isOpened():
                    logger.error("Could not open video: {}", input_path.resolve())
                    raise typer.Exit(code=1)
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                if fps <= 0:
                    fps = 30.0

            while True:
                if kind == "image":
                    if frame_idx > 0:
                        break
                    if frame0 is None:
                        logger.error("Internal error: image frame missing.")
                        raise typer.Exit(code=1)
                    frame = frame0.copy()
                    frame_idx += 1
                else:
                    assert cap is not None
                    ok, frame = cap.read()
                    if not ok or frame is None or frame.size == 0:
                        break
                    frame_idx += 1

                timestamp_ms = int((time.monotonic() - t0) * 1000.0)
                cache_idx = rot_angle // ROT_RESO
                tmp_img = img_cache[cache_idx][0]
                overlay_rgb = np.asarray(tmp_img.convert("RGB"))

                last_raw_face = face_source.face_box(frame, timestamp_ms)

                smooth_and_draw(
                    frame,
                    last_raw_face,
                    roi_state,
                    overlay_rgb,
                    mask_l,
                    privacy_effect=deps.privacy,
                    center_lambda=roi_lambda,
                    size_lambda=size_lambda,
                    no_face_blur_frames=no_face_blur_frames,
                    show_preview=preview,
                    overlay_scale=overlay_scale,
                    roi_motion=roi_motion,
                )

                if preview:
                    delay = KEY_WAIT_DELAY_MS if kind == "video" else 0
                    key = cv2.waitKeyEx(delay)
                    if key in (3, 17, 27):
                        break

                fh, fw = frame.shape[:2]
                if writer is None and kind == "video":
                    fourcc = _CV2.VideoWriter_fourcc(*"mp4v")
                    writer = _CV2.VideoWriter(str(out), fourcc, fps, (fw, fh))
                    if not writer.isOpened():
                        logger.error("Could not open video writer for {}", out.resolve())
                        raise typer.Exit(code=1)

                if kind == "image":
                    if not cv2.imwrite(str(out), frame):
                        logger.error("Could not write image: {}", out.resolve())
                        raise typer.Exit(code=1)
                    logger.info("Wrote {}", out.resolve())
                else:
                    assert writer is not None
                    writer.write(frame)

                rot_angle = (rot_angle + ROT_RESO) % 360

            if kind == "video":
                if writer is None:
                    logger.warning("No frames written for {}", input_path.resolve())
                else:
                    logger.info("Wrote {} ({} frames)", out.resolve(), frame_idx)
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()

    try:
        if face_backend == "blaze":
            if cascade_margin > 0:
                logger.warning("--cascade-margin applies to YuNet only; ignoring for BlazeFace.")
            with create_face_detector() as detector:
                logger.debug(
                    "TFLite delegate in use: {} (requested {}).",
                    delegate_label,
                    "GPU" if use_gpu else "CPU",
                )
                logger.info("Face backend: blaze (MediaPipe BlazeFace)")
                process_with_source(BlazeFaceFaceBoxSource(detector))
        else:
            assert yunet_path is not None
            yunet_det = create_yunet_detector(yunet_path, input_w, input_h)
            inner = YuNetFaceBoxSource(yunet_det)
            if cascade_margin > 0:
                cascaded: FaceBoxSource = CascadedFaceBoxSource(inner, cascade_margin, roi_state)
                logger.info(
                    "Face backend: yunet (OpenCV YuNet), cascade_margin={:.3f}",
                    cascade_margin,
                )
                process_with_source(cascaded)
            else:
                logger.info("Face backend: yunet (OpenCV YuNet)")
                process_with_source(inner)
    finally:
        if preview:
            cv2.destroyAllWindows()
