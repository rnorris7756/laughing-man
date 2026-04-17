"""Orchestration: webcam, detector, overlay cache, capture loop."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pyvirtualcam
import typer
from loguru import logger
from mediapipe.tasks.python import vision
from PIL import Image
from pyvirtualcam import PixelFormat

from laughing_man.camera import open_webcam
from laughing_man.cascade import CascadedFaceBoxSource
from laughing_man.constants import (
    DEFAULT_CAMERA,
    DEFAULT_ROI_MOTION,
    KEY_WAIT_DELAY_MS,
    LAMBDA_TUNE_STEP,
    MAIN_WIN_NAME,
    ROT_RESO,
)
from laughing_man.deps import PipelineDeps
from laughing_man.detection import (
    BlazeFaceFaceBoxSource,
    face_detector_options,
)
from laughing_man.model import (
    ensure_blaze_face_model,
    ensure_yunet_model,
    resolve_model,
    resolve_yunet_model,
)
from laughing_man.overlay import (
    build_overlay_rgb_cache,
    load_overlay_images,
    make_overlay_mask_resized,
    prefill_rotated_overlay_cache_inplace,
)
from laughing_man.privacy import GaussianBlurPrivacy
from laughing_man.protocols import FaceBoxSource
from laughing_man.roi import RoiState, smooth_and_draw
from laughing_man.tuning import (
    lambda_deltas_from_arrow_key,
    should_quit_preview,
    stdin_interactive_tuning_available,
    terminal_stdin_tune_loop,
)
from laughing_man.yunet_face import YuNetFaceBoxSource, create_yunet_detector


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
    camera: str = DEFAULT_CAMERA,
    overlay_image: Path | None,
    overlay_scale: float,
    face_backend: Literal["blaze", "yunet"] = "blaze",
    cascade_margin: float = 0.0,
    roi_motion: Literal["ema", "kalman", "kalman_flow"] = DEFAULT_ROI_MOTION,
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
        Temporal smoothing for **horizontal and vertical** overlay position
        (0 = snap to raw detection each frame).
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
    camera
        ``auto`` (default), integer index, or ``/dev/video*``. On Linux, ``auto``
        skips v4l2loopback output devices so the physical webcam is used.
    overlay_image
        If set, use this file as the face overlay instead of the bundled defaults
        in ``laughing_man.assets`` (``limg.png`` / ``ltext.png``).
    overlay_scale
        Scale factor for mapping overlay art onto the face ROI (see ``--scale``).
    face_backend
        ``blaze`` (MediaPipe BlazeFace) or ``yunet`` (OpenCV YuNet ONNX).
    cascade_margin
        YuNet only: crop expansion for cascade detection (see ``--cascade-margin``).
    roi_motion
        Temporal model for the face box: ``ema``, ``kalman``, or ``kalman_flow``
        (see ``--roi-motion``).
    """
    if not virtual_cam and not show_preview:
        logger.error("Need at least one of --virtual-cam or preview (omit --no-preview).")
        raise typer.Exit(code=1)

    deps = PipelineDeps(privacy=GaussianBlurPrivacy())

    cap = open_webcam(camera)

    if show_preview:
        cv2.namedWindow(MAIN_WIN_NAME, cv2.WINDOW_AUTOSIZE)

    st_img, rot_img = load_overlay_images(overlay_image)

    rot_angle = 0

    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    delegate_label: str = "CPU"
    blaze_model_path: Path | None = None
    yunet_path: Path | None = None

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

    cap_fps = float(cap.get(cv2.CAP_PROP_FPS))
    stream_fps = virtual_fps if virtual_fps > 0 else (cap_fps if cap_fps > 0 else 30.0)
    min_dim = min(input_h, input_w)
    mask_img = make_overlay_mask_resized(overlay_image, st_img, rot_img, min_dim)

    n_cache = 360 // ROT_RESO
    img_cache: list[list[Image.Image]] = [[] for _ in range(n_cache)]

    prefill_exc: list[BaseException] = []

    def _prefill_rotated_overlay_cache() -> None:
        """Fill ``img_cache`` for all discrete rotation steps (runs in a helper thread)."""
        try:
            prefill_rotated_overlay_cache_inplace(
                img_cache,
                overlay_image=overlay_image,
                st_img=st_img,
                rot_img=rot_img,
                min_dim=min_dim,
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
        face_source: FaceBoxSource,
        vcam: pyvirtualcam.Camera | None,
    ) -> None:
        nonlocal rot_angle, last_raw_face, roi_lambda_live, size_lambda_live
        if roi_motion != "ema":
            logger.info("ROI motion model: {}", roi_motion)
        prefill_thread.join()
        if prefill_exc:
            e = prefill_exc[0]
            logger.error("Rotated overlay pre-computation failed: {}", e)
            raise typer.Exit(code=1) from e

        overlay_rgb_cache = build_overlay_rgb_cache(img_cache)
        mask_l_arr = np.asarray(mask_img)

        use_terminal_tuning = stdin_interactive_tuning_available()
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
                target=terminal_stdin_tune_loop,
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
                    overlay_rgb = overlay_rgb_cache[cache_idx]
                    mask_l = mask_l_arr

                    last_raw_face = face_source.face_box(frame, timestamp_ms)

                    smooth_and_draw(
                        frame,
                        last_raw_face,
                        roi_state,
                        overlay_rgb,
                        mask_l,
                        privacy_effect=deps.privacy,
                        center_lambda=roi_lambda_live,
                        size_lambda=size_lambda_live,
                        no_face_blur_frames=no_face_blur_frames,
                        show_preview=show_preview,
                        overlay_scale=overlay_scale,
                        roi_motion=roi_motion,
                    )

                    if vcam is not None:
                        vcam.send(frame)
                        vcam.sleep_until_next_frame()

                    rot_angle = (rot_angle + ROT_RESO) % 360

                    if show_preview:
                        delay = 1 if virtual_cam else KEY_WAIT_DELAY_MS
                        key = cv2.waitKeyEx(delay)
                        if not use_terminal_tuning and key >= 0:
                            if should_quit_preview(key):
                                break
                            deltas = lambda_deltas_from_arrow_key(key)
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

    def _run_with_face_source(face_src: FaceBoxSource) -> None:
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
                    run_capture_loop(face_src, vcam)
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
            run_capture_loop(face_src, None)

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
                _run_with_face_source(BlazeFaceFaceBoxSource(detector))
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
                _run_with_face_source(cascaded)
            else:
                logger.info("Face backend: yunet (OpenCV YuNet)")
                _run_with_face_source(inner)
    finally:
        cap.release()
        cv2.destroyAllWindows()
