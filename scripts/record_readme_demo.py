#!/usr/bin/env python3
"""Command-line wrapper for the local README demo recording harness."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Annotated, Literal

import typer

from readme_demo_harness import (
    BUILD,
    FINAL_GIF,
    FINAL_MP4,
    OVERLAY_MP4,
    PALETTE_PNG,
    POSTER_JPG,
    ROOT,
    TERMINAL_MP4,
    TERMINAL_TAPE,
    app_command,
    assemble_command,
    capture_command,
    require,
    run,
    shell_join,
    stop,
)

app = typer.Typer(help="Record and assemble the README demo.", add_completion=False, no_args_is_help=True)


@app.command("check")
def check() -> None:
    """
    Check for external recording tools.

    Returns
    -------
    None
        Prints a success message or exits with a missing-tool error.
    """
    require(("uv", "vhs", "ffmpeg"))
    typer.echo("All required tools are available.")


@app.command("terminal")
def terminal(dry_run: Annotated[bool, typer.Option("--dry-run")] = False) -> None:
    """
    Render the terminal segment with VHS.

    Parameters
    ----------
    dry_run
        Print the VHS command without running it.
    """
    require(("vhs",))
    BUILD.mkdir(parents=True, exist_ok=True)
    run(["vhs", str(TERMINAL_TAPE.relative_to(ROOT))], dry_run=dry_run)
    typer.echo(f"Terminal clip: {TERMINAL_MP4.relative_to(ROOT)}")


@app.command("overlay")
def overlay(
    virtual_device: Annotated[Path, typer.Option("--virtual-device")] = Path("/dev/video10"),
    camera: Annotated[str, typer.Option("--camera")] = "auto",
    face_backend: Annotated[Literal["blaze", "yunet"], typer.Option("--face-backend")] = "blaze",
    duration: Annotated[float, typer.Option("--duration", min=1.0)] = 8.0,
    fps: Annotated[int, typer.Option("--fps", min=1)] = 30,
    video_size: Annotated[str | None, typer.Option("--video-size")] = None,
    warmup: Annotated[float, typer.Option("--warmup", min=0.0)] = 3.0,
    preview: Annotated[bool, typer.Option("--preview")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """
    Record the virtual-camera overlay segment on Linux.

    Parameters
    ----------
    virtual_device
        v4l2loopback device to record.
    camera
        Physical camera token passed to ``laughing-man``.
    face_backend
        Face detector backend for the capture.
    duration
        Recording length in seconds.
    fps
        Capture and output frames per second.
    video_size
        Optional v4l2 input size, such as ``1280x720``.
    warmup
        Seconds to let the overlay pipeline start before recording.
    preview
        Show the OpenCV preview window while recording.
    dry_run
        Print commands without running them.
    """
    require(("uv", "ffmpeg"))
    BUILD.mkdir(parents=True, exist_ok=True)
    run_app = app_command(
        virtual_device=virtual_device,
        camera=camera,
        face_backend=face_backend,
        fps=fps,
        preview=preview,
    )
    record = capture_command(
        virtual_device=virtual_device,
        output=OVERLAY_MP4,
        duration=duration,
        fps=fps,
        video_size=video_size,
    )
    typer.echo(f"$ {shell_join(run_app)}")
    typer.echo(f"$ {shell_join(record)}")
    if dry_run:
        return
    process = subprocess.Popen(run_app, cwd=ROOT, text=True)
    try:
        time.sleep(warmup)
        subprocess.run(record, cwd=ROOT, check=True)
    finally:
        stop(process)
    typer.echo(f"Overlay clip: {OVERLAY_MP4.relative_to(ROOT)}")


@app.command("assemble")
def assemble(
    terminal_clip: Annotated[Path, typer.Option("--terminal-clip", exists=True, dir_okay=False)] = TERMINAL_MP4,
    overlay_clip: Annotated[Path, typer.Option("--overlay-clip", exists=True, dir_okay=False)] = OVERLAY_MP4,
    output: Annotated[Path, typer.Option("--output", dir_okay=False)] = FINAL_MP4,
    width: Annotated[int, typer.Option("--width", min=320)] = 1280,
    height: Annotated[int, typer.Option("--height", min=240)] = 720,
    fps: Annotated[int, typer.Option("--fps", min=1)] = 30,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """
    Concatenate the terminal and overlay clips.

    Parameters
    ----------
    terminal_clip
        Terminal usage clip from VHS.
    overlay_clip
        Face overlay clip recorded locally.
    output
        Final MP4 path.
    width
        Output width in pixels.
    height
        Output height in pixels.
    fps
        Output frames per second.
    dry_run
        Print the ffmpeg command without running it.
    """
    require(("ffmpeg",))
    BUILD.mkdir(parents=True, exist_ok=True)
    run(
        assemble_command(
            terminal_clip=terminal_clip,
            overlay_clip=overlay_clip,
            output=output,
            width=width,
            height=height,
            fps=fps,
        ),
        dry_run=dry_run,
    )
    typer.echo(f"Demo MP4: {output.relative_to(ROOT)}")


@app.command("poster")
def poster(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)] = FINAL_MP4,
    output: Annotated[Path, typer.Option("--output", dir_okay=False)] = POSTER_JPG,
    timestamp: Annotated[float, typer.Option("--timestamp", min=0.0)] = 1.0,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """
    Extract a poster frame from the assembled MP4.

    Parameters
    ----------
    input_path
        Source MP4.
    output
        Poster image path.
    timestamp
        Seconds into the video.
    dry_run
        Print the ffmpeg command without running it.
    """
    require(("ffmpeg",))
    BUILD.mkdir(parents=True, exist_ok=True)
    run(
        ["ffmpeg", "-y", "-ss", f"{timestamp:g}", "-i", str(input_path), "-frames:v", "1", "-q:v", "3", str(output)],
        dry_run=dry_run,
    )
    typer.echo(f"Poster image: {output.relative_to(ROOT)}")


@app.command("gif")
def gif(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)] = FINAL_MP4,
    output: Annotated[Path, typer.Option("--output", dir_okay=False)] = FINAL_GIF,
    width: Annotated[int, typer.Option("--width", min=320)] = 960,
    fps: Annotated[int, typer.Option("--fps", min=1)] = 12,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """
    Create an optimized GIF fallback from the assembled MP4.

    Parameters
    ----------
    input_path
        Source MP4.
    output
        GIF path.
    width
        GIF width in pixels.
    fps
        GIF frames per second.
    dry_run
        Print the ffmpeg commands without running them.
    """
    require(("ffmpeg",))
    BUILD.mkdir(parents=True, exist_ok=True)
    common = f"fps={fps},scale={width}:-1:flags=lanczos"
    run(["ffmpeg", "-y", "-i", str(input_path), "-vf", f"{common},palettegen", str(PALETTE_PNG)], dry_run=dry_run)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-i",
            str(PALETTE_PNG),
            "-filter_complex",
            f"{common}[x];[x][1:v]paletteuse",
            str(output),
        ],
        dry_run=dry_run,
    )
    typer.echo(f"Demo GIF: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    app()
