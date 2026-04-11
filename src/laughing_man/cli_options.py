"""Shared Typer :class:`typer.Option` / :class:`typer.Argument` annotations for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer

from laughing_man.constants import (
    DEFAULT_NO_FACE_BLUR_FRAMES,
    DEFAULT_ROI_LAMBDA,
    DEFAULT_ROI_MOTION,
    DEFAULT_SIZE_LAMBDA,
)

# --- Shared by ``run`` and ``postprocess`` ------------------------------------

FullRangeOpt = Annotated[
    bool,
    typer.Option(
        False,
        "--full-range",
        help=(
            "Use BlazeFace full-range model (better for faces that are small or far "
            "from the camera). Default is short-range (typical desk webcam). Ignored "
            "for --face-backend yunet."
        ),
    ),
]

GPUOpt = Annotated[
    bool,
    typer.Option(
        False,
        "--gpu",
        help=(
            "Use MediaPipe's TensorFlow Lite GPU delegate when possible. This is not "
            "the same as CUDA in PyTorch; it uses the GPU via TFLite's graphics/compute "
            "path (vendor-dependent). Falls back to CPU if GPU init fails. AMD iGPU "
            "may work if drivers expose a supported API."
        ),
    ),
]

RoiLambdaOpt = Annotated[
    float,
    typer.Option(
        DEFAULT_ROI_LAMBDA,
        "--roi-lambda",
        min=0.0,
        max=1.0,
        help=(
            "Low-pass on **horizontal and vertical** overlay position vs the detector "
            "(0 = snap; higher = less center jitter when sitting still). Does not "
            "control box size; see --size-lambda."
        ),
    ),
]

SizeLambdaOpt = Annotated[
    float,
    typer.Option(
        DEFAULT_SIZE_LAMBDA,
        "--size-lambda",
        min=0.0,
        max=1.0,
        help=(
            "Low-pass on overlay **width/height** vs the detector box (higher = less "
            "size jitter when pose changes). Raise above --roi-lambda if bbox size "
            "fluctuates more than you want relative to horizontal position."
        ),
    ),
]

NoFaceBlurFramesOpt = Annotated[
    int,
    typer.Option(
        DEFAULT_NO_FACE_BLUR_FRAMES,
        "--no-face-blur-frames",
        min=1,
        help=(
            "Consecutive frames with no face before full-frame privacy blur. Until "
            "then the overlay stays at the last smoothed position (reduces flicker)."
        ),
    ),
]

DebugOpt = Annotated[
    bool,
    typer.Option(
        False,
        "--debug",
        help=(
            "Verbose logging (overlay prefill, TFLite delegate, virtual camera when "
            "applicable, and related pipeline details)."
        ),
    ),
]

OverlayImageOpt = Annotated[
    Path | None,
    typer.Option(
        None,
        "--image",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help=(
            "PNG/JPEG/etc. to composite on the face instead of the default Laughing Man assets (limg.png / ltext.png)."
        ),
    ),
]

OverlayScaleOpt = Annotated[
    float,
    typer.Option(
        1.0,
        "--scale",
        min=0.05,
        max=10.0,
        help=(
            "Scale overlay art relative to the detected face box (1.0 = default). "
            "Above 1.0 zooms in on the center of the image; below 1.0 shrinks it with "
            "clear margins. Applies to any overlay, including --image."
        ),
    ),
]

FaceBackendOpt = Annotated[
    Literal["blaze", "yunet"],
    typer.Option(
        "blaze",
        "--face-backend",
        help=(
            "Face detector: blaze = MediaPipe BlazeFace (default); yunet = OpenCV "
            "YuNet ONNX. Run twice with different values to compare jitter. See also "
            "--full-range (blaze only) and --cascade-margin (yunet only)."
        ),
    ),
]

CascadeMarginOpt = Annotated[
    float,
    typer.Option(
        0.0,
        "--cascade-margin",
        min=0.0,
        max=2.0,
        help=(
            "YuNet only: expand the previous smoothed face box by this fraction per "
            "side and run detection on that crop first; fall back to full frame if "
            "needed. Zero disables. Ignored for blaze (BlazeFace VIDEO mode)."
        ),
    ),
]

RoiMotionOpt = Annotated[
    Literal["ema", "kalman", "kalman_flow"],
    typer.Option(
        DEFAULT_ROI_MOTION,
        "--roi-motion",
        help=(
            "How to stabilize the face box over time: ema = exponential moving average "
            "(--roi-lambda / --size-lambda); kalman = constant-velocity Kalman filter "
            "on center and size; kalman_flow = Kalman plus LK optical flow to blend the "
            "measured center toward frame-to-frame motion. For kalman modes, lambda "
            "tuning mostly affects EMA only (not used)."
        ),
    ),
]

# --- ``run`` only -------------------------------------------------------------

VirtualCamOpt = Annotated[
    bool,
    typer.Option(
        False,
        "--virtual-cam",
        help=(
            "Expose the composited video as a virtual webcam (Linux: v4l2loopback) so "
            "apps like Discord or OBS can capture it. Requires the kernel module; see "
            "--v4l2-device."
        ),
    ),
]

V4l2DeviceOpt = Annotated[
    str | None,
    typer.Option(
        None,
        "--v4l2-device",
        help=(
            "Virtual camera device path (e.g. /dev/video10). If omitted, the first "
            "available v4l2loopback device is used."
        ),
    ),
]

VirtualFpsOpt = Annotated[
    float,
    typer.Option(
        30.0,
        "--virtual-fps",
        min=0.1,
        help="Target frame rate for the virtual camera (used with --virtual-cam).",
    ),
]

NoPreviewOpt = Annotated[
    bool,
    typer.Option(
        False,
        "--no-preview",
        help=(
            "Do not open the OpenCV preview window. Use with --virtual-cam to stream "
            "only to the virtual device (quit with Ctrl+C)."
        ),
    ),
]

# --- ``postprocess`` only -----------------------------------------------------

PostprocessInputArg = Annotated[
    Path,
    typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Input image or video file (OpenCV-supported formats).",
    ),
]

PostprocessOutputOpt = Annotated[
    Path | None,
    typer.Option(
        None,
        "--output",
        "-o",
        dir_okay=False,
        resolve_path=True,
        help="Output path. Default: <input stem>_overlay.png|.mp4 next to the input.",
    ),
]

PostprocessPreviewOpt = Annotated[
    bool,
    typer.Option(
        False,
        "--preview",
        help="Show an OpenCV preview window while processing.",
    ),
]

__all__ = (
    "CascadeMarginOpt",
    "DebugOpt",
    "FaceBackendOpt",
    "FullRangeOpt",
    "GPUOpt",
    "NoFaceBlurFramesOpt",
    "NoPreviewOpt",
    "OverlayImageOpt",
    "OverlayScaleOpt",
    "PostprocessInputArg",
    "PostprocessOutputOpt",
    "PostprocessPreviewOpt",
    "RoiLambdaOpt",
    "RoiMotionOpt",
    "SizeLambdaOpt",
    "V4l2DeviceOpt",
    "VirtualCamOpt",
    "VirtualFpsOpt",
)
