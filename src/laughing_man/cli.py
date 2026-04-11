"""Typer CLI entry."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from laughing_man.constants import (
    DEFAULT_NO_FACE_BLUR_FRAMES,
    DEFAULT_ROI_LAMBDA,
    DEFAULT_ROI_MOTION,
    DEFAULT_SIZE_LAMBDA,
)
from laughing_man.logging_setup import configure_logging
from laughing_man.run import run_overlay

app = typer.Typer(
    help="Webcam Laughing Man face overlay (MediaPipe BlazeFace + OpenCV).",
    add_completion=False,
    no_args_is_help=True,
)


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
            "Low-pass on **horizontal and vertical** overlay position vs the detector "
            "(0 = snap; higher = less center jitter when sitting still). Does not "
            "control box size; see --size-lambda."
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
    face_backend: Literal["blaze", "yunet"] = typer.Option(
        "blaze",
        "--face-backend",
        help=(
            "Face detector: blaze = MediaPipe BlazeFace (default); yunet = OpenCV "
            "YuNet ONNX. Run twice with different values to compare jitter. "
            "See also --full-range (blaze only) and --cascade-margin (yunet only)."
        ),
    ),
    cascade_margin: float = typer.Option(
        0.0,
        "--cascade-margin",
        min=0.0,
        max=2.0,
        help=(
            "YuNet only: expand the previous smoothed face box by this fraction "
            "per side and run detection on that crop first; fall back to full frame "
            "if needed. Zero disables. Ignored for blaze (BlazeFace VIDEO mode)."
        ),
    ),
    roi_motion: Literal["ema", "kalman", "kalman_flow"] = typer.Option(
        DEFAULT_ROI_MOTION,
        "--roi-motion",
        help=(
            "How to stabilize the face box over time: ema = exponential moving "
            "average (--roi-lambda / --size-lambda); kalman = constant-velocity Kalman "
            "filter on center and size; kalman_flow = Kalman plus LK optical flow to "
            "blend the measured center toward frame-to-frame motion. "
            "For kalman modes, lambda tuning mostly affects EMA only (not used)."
        ),
    ),
) -> None:
    """Run the Laughing Man webcam overlay."""
    configure_logging(debug=debug)
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
        face_backend=face_backend,
        cascade_margin=cascade_margin,
        roi_motion=roi_motion,
    )


if __name__ == "__main__":
    app()
