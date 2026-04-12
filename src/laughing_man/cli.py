"""Typer CLI entry."""

from __future__ import annotations

import sys

import typer

from laughing_man.__version__ import __version__ as package_version
from laughing_man.cli_options import *
from laughing_man.logging_setup import configure_logging
from laughing_man.postprocess import run_postprocess
from laughing_man.run import run_overlay

typer_app = typer.Typer(
    help="Webcam Laughing Man face overlay (MediaPipe BlazeFace + OpenCV).",
    add_completion=False,
    no_args_is_help=True,
)


@typer_app.command("run")
def run(
    full_range: FullRangeOpt,
    gpu: GPUOpt,
    roi_lambda: RoiLambdaOpt,
    size_lambda: SizeLambdaOpt,
    no_face_blur_frames: NoFaceBlurFramesOpt,
    virtual_cam: VirtualCamOpt,
    v4l2_device: V4l2DeviceOpt,
    virtual_fps: VirtualFpsOpt,
    no_preview: NoPreviewOpt,
    debug: DebugOpt,
    image: OverlayImageOpt,
    overlay_scale: OverlayScaleOpt,
    face_backend: FaceBackendOpt,
    cascade_margin: CascadeMarginOpt,
    roi_motion: RoiMotionOpt,
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


@typer_app.command("postprocess")
def postprocess(
    input_path: PostprocessInputArg,
    output_path: PostprocessOutputOpt,
    full_range: FullRangeOpt,
    gpu: GPUOpt,
    roi_lambda: RoiLambdaOpt,
    size_lambda: SizeLambdaOpt,
    no_face_blur_frames: NoFaceBlurFramesOpt,
    debug: DebugOpt,
    image: OverlayImageOpt,
    overlay_scale: OverlayScaleOpt,
    face_backend: FaceBackendOpt,
    cascade_margin: CascadeMarginOpt,
    roi_motion: RoiMotionOpt,
    preview: PostprocessPreviewOpt,
) -> None:
    """Apply the Laughing Man face overlay to a still image or video file (offline)."""
    configure_logging(debug=debug)
    run_postprocess(
        input_path=input_path,
        output_path=output_path,
        overlay_image=image,
        overlay_scale=overlay_scale,
        full_range=full_range,
        use_gpu=gpu,
        roi_lambda=roi_lambda,
        size_lambda=size_lambda,
        no_face_blur_frames=no_face_blur_frames,
        face_backend=face_backend,
        cascade_margin=cascade_margin,
        roi_motion=roi_motion,
        preview=preview,
    )


def app() -> None:
    """CLI entry: default subcommand ``run`` when omitted (backward compatible)."""
    argv = sys.argv
    if len(argv) == 2 and argv[1] in ("-V", "--version"):
        typer.echo(f"laughing-man {package_version}")
        raise typer.Exit(code=0)
    if len(argv) == 1:
        argv.append("run")
    elif len(argv) >= 2 and argv[1] not in (
        "run",
        "postprocess",
        "-h",
        "--help",
        "-V",
        "--version",
    ):
        argv.insert(1, "run")
    typer_app()


if __name__ == "__main__":
    app()
