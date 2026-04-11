"""Orchestration: webcam, detector, overlay cache, capture loop."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyvirtualcam
import typer
from loguru import logger
from mediapipe.tasks.python import vision
from PIL import Image, ImageChops, ImageDraw
from pyvirtualcam import PixelFormat

from laughing_man.camera import open_webcam
from laughing_man.constants import (
    CAMERA_INDEX,
    KEY_WAIT_DELAY_MS,
    LAMBDA_TUNE_STEP,
    MAIN_WIN_NAME,
    ROT_RESO,
    TMP_OFFSET,
)
from laughing_man.deps import PipelineDeps
from laughing_man.detection import (
    BlazeFaceFaceBoxSource,
    face_detector_options,
)
from laughing_man.model import ensure_blaze_face_model, resolve_model
from laughing_man.overlay import build_rotated_overlay_frame, load_overlay_images
from laughing_man.privacy import GaussianBlurPrivacy
from laughing_man.roi import RoiState, smooth_and_draw
from laughing_man.tuning import (
    lambda_deltas_from_arrow_key,
    should_quit_preview,
    stdin_interactive_tuning_available,
    terminal_stdin_tune_loop,
)


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
        If set, use this file as the face overlay instead of the bundled defaults
        in ``laughing_man.assets`` (``limg.png`` / ``ltext.png``).
    overlay_scale
        Scale factor for mapping overlay art onto the face ROI (see ``--scale``).
    """
    if not virtual_cam and not show_preview:
        logger.error(
            "Need at least one of --virtual-cam or preview (omit --no-preview)."
        )
        raise typer.Exit(code=1)

    deps = PipelineDeps(privacy=GaussianBlurPrivacy())

    model_path, model_url = resolve_model(full_range)
    ensure_blaze_face_model(model_path, model_url)

    delegate_label: str

    def create_face_detector() -> vision.FaceDetector:
        """Open FaceDetector, falling back from GPU to CPU on failure."""
        nonlocal delegate_label
        opts = face_detector_options(model_path, use_gpu=use_gpu)
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
                opts = face_detector_options(model_path, use_gpu=False)
                try:
                    delegate_label = "CPU"
                    return vision.FaceDetector.create_from_options(opts)
                except Exception as e2:
                    logger.error("Could not create face detector: {}", e2)
                    raise typer.Exit(code=1) from e2
            logger.error("Could not create face detector: {}", e)
            raise typer.Exit(code=1) from e

    cap = open_webcam(CAMERA_INDEX)

    if show_preview:
        cv2.namedWindow(MAIN_WIN_NAME, cv2.WINDOW_AUTOSIZE)

    st_img, rot_img = load_overlay_images(overlay_image)

    st_bands = st_img.split()
    rot_bands = rot_img.split()
    st_alpha = st_bands[3]
    rot_alpha = rot_bands[3]
    rot_angle = 0

    im_sz = st_img.size
    if overlay_image is not None:
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
                shared = build_rotated_overlay_frame(
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
                        build_rotated_overlay_frame(
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

        face_source = BlazeFaceFaceBoxSource(detector)

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
                    tmp_img = img_cache[cache_idx][0]

                    overlay_rgb = np.asarray(tmp_img.convert("RGB"))
                    mask_l = np.asarray(mask_img)

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
