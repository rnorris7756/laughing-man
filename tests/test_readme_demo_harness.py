"""Tests for the README demo recording harness command builders."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import readme_demo_harness as harness  # noqa: E402


def test_app_command_uses_virtual_camera_headless_by_default() -> None:
    command = harness.app_command(
        virtual_device=Path("/dev/video10"),
        camera="auto",
        face_backend="yunet",
        fps=30,
        preview=False,
    )

    assert command[:3] == ["uv", "run", "laughing-man"]
    assert command[command.index("--camera") + 1] == "auto"
    assert command[command.index("--face-backend") + 1] == "yunet"
    assert command[command.index("--v4l2-device") + 1] == "/dev/video10"
    assert "--virtual-cam" in command
    assert "--no-preview" in command


def test_app_command_preview_keeps_window_enabled() -> None:
    command = harness.app_command(
        virtual_device=Path("/dev/video10"),
        camera="/dev/video1",
        face_backend="blaze",
        fps=24,
        preview=True,
    )

    assert command[command.index("--virtual-fps") + 1] == "24"
    assert "--no-preview" not in command


def test_capture_command_includes_video_size_only_when_requested() -> None:
    without_size = harness.capture_command(
        virtual_device=Path("/dev/video10"),
        output=Path("overlay.mp4"),
        duration=8.0,
        fps=30,
        video_size=None,
    )
    with_size = harness.capture_command(
        virtual_device=Path("/dev/video10"),
        output=Path("overlay.mp4"),
        duration=8.0,
        fps=30,
        video_size="1280x720",
    )

    assert "-video_size" not in without_size
    assert with_size[with_size.index("-video_size") + 1] == "1280x720"
    assert with_size[with_size.index("-t") + 1] == "8"
    assert with_size[-1] == "overlay.mp4"


def test_assemble_command_concats_normalized_video_streams() -> None:
    command = harness.assemble_command(
        terminal_clip=Path("terminal.mp4"),
        overlay_clip=Path("overlay.mp4"),
        output=Path("demo.mp4"),
        width=1280,
        height=720,
        fps=30,
    )

    graph = command[command.index("-filter_complex") + 1]
    assert "scale=1280:720:force_original_aspect_ratio=decrease" in graph
    assert "[v0][v1]concat=n=2:v=1:a=0[v]" in graph
    assert command[command.index("-map") + 1] == "[v]"
    assert command[-1] == "demo.mp4"
