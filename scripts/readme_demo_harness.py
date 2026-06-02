"""Shared helpers for the local README demo recording harness."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import typer

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "demo"
BUILD = DEMO / "build"
TERMINAL_TAPE = DEMO / "terminal.tape"
TERMINAL_MP4 = BUILD / "terminal.mp4"
OVERLAY_MP4 = BUILD / "overlay.mp4"
FINAL_MP4 = BUILD / "laughing-man-demo.mp4"
FINAL_GIF = BUILD / "laughing-man-demo.gif"
POSTER_JPG = BUILD / "laughing-man-demo-poster.jpg"
PALETTE_PNG = BUILD / "laughing-man-demo-palette.png"


def shell_join(command: list[str]) -> str:
    """
    Format a command for copyable terminal output.

    Parameters
    ----------
    command
        Subprocess argument list.

    Returns
    -------
    str
        Shell-escaped command string.
    """
    return " ".join(shlex.quote(part) for part in command)


def run(command: list[str], *, dry_run: bool = False) -> None:
    """
    Run a command from the repository root.

    Parameters
    ----------
    command
        Subprocess argument list.
    dry_run
        Print the command without executing it.
    """
    typer.echo(f"$ {shell_join(command)}")
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def require(names: tuple[str, ...]) -> None:
    """
    Fail when required executables are missing.

    Parameters
    ----------
    names
        Program names to resolve on ``PATH``.
    """
    missing = [name for name in names if shutil.which(name) is None]
    if missing:
        typer.echo(f"Missing required program(s): {', '.join(missing)}", err=True)
        raise typer.Exit(1)


def app_command(
    *,
    virtual_device: Path,
    camera: str,
    face_backend: Literal["blaze", "yunet"],
    fps: int,
    preview: bool,
) -> list[str]:
    """
    Build the ``laughing-man`` virtual-camera command.

    Parameters
    ----------
    virtual_device
        v4l2loopback device written by ``laughing-man``.
    camera
        Physical camera token passed to ``--camera``.
    face_backend
        Face detector backend.
    fps
        Virtual camera frames per second.
    preview
        Show the OpenCV preview window while recording.

    Returns
    -------
    list[str]
        Command suitable for ``subprocess.Popen``.
    """
    command = [
        "uv",
        "run",
        "laughing-man",
        "--camera",
        camera,
        "--face-backend",
        face_backend,
        "--virtual-cam",
        "--v4l2-device",
        str(virtual_device),
        "--virtual-fps",
        str(fps),
    ]
    if not preview:
        command.append("--no-preview")
    return command


def capture_command(
    *,
    virtual_device: Path,
    output: Path,
    duration: float,
    fps: int,
    video_size: str | None,
) -> list[str]:
    """
    Build the ffmpeg command that records a Linux virtual camera.

    Parameters
    ----------
    virtual_device
        v4l2loopback device to read.
    output
        MP4 path to write.
    duration
        Capture length in seconds.
    fps
        Capture and output frames per second.
    video_size
        Optional v4l2 input size, such as ``1280x720``.

    Returns
    -------
    list[str]
        Command suitable for ``subprocess.run``.
    """
    command = ["ffmpeg", "-y", "-f", "v4l2", "-framerate", str(fps)]
    if video_size:
        command += ["-video_size", video_size]
    return command + [
        "-i",
        str(virtual_device),
        "-t",
        f"{duration:g}",
        "-an",
        "-vf",
        f"fps={fps},scale=-2:720:flags=lanczos,format=yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def assemble_command(
    *,
    terminal_clip: Path,
    overlay_clip: Path,
    output: Path,
    width: int,
    height: int,
    fps: int,
) -> list[str]:
    """
    Build the ffmpeg hard-cut assembly command.

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

    Returns
    -------
    list[str]
        Command suitable for ``subprocess.run``.
    """
    norm = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p"
    )
    graph = f"[0:v]{norm}[v0];[1:v]{norm}[v1];[v0][v1]concat=n=2:v=1:a=0[v]"
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(terminal_clip),
        "-i",
        str(overlay_clip),
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]


def stop(process: subprocess.Popen[str]) -> None:
    """
    Stop a long-running process.

    Parameters
    ----------
    process
        Process returned by ``subprocess.Popen``.
    """
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
