"""Tests for webcam selection and v4l2loopback avoidance."""

from __future__ import annotations

import sys
from collections.abc import Callable

import pytest
import typer

from laughing_man.camera import (
    _refuse_loopback_before_open,
    _resolve_candidates,
    linux_is_v4l2_loopback,
    linux_non_loopback_indices,
    parse_camera_token,
)


def test_parse_camera_token_index() -> None:
    assert parse_camera_token("0") == 0
    assert parse_camera_token(" 2 ") == 2


def test_parse_camera_token_path() -> None:
    assert parse_camera_token("/dev/video1") == "/dev/video1"


def test_parse_camera_token_invalid_exits() -> None:
    with pytest.raises(typer.Exit) as exc_info:
        parse_camera_token("not-a-device")
    assert exc_info.value.exit_code == 1


def test_resolve_candidates_auto_uses_index_zero_off_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When ``sys.platform`` is not Linux, ``auto`` is only ``[0]``.

    This does not open a camera or call OpenCV; it is pure candidate selection.
    We patch ``platform`` so Linux CI still exercises this branch without a webcam.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _resolve_candidates("auto") == [0]


def test_linux_is_v4l2_loopback_sysfs_only_no_camera() -> None:
    """
    ``linux_is_v4l2_loopback`` only reads sysfs symlinks; it never opens /dev/video*.

    ``video0`` may or may not exist; we only require a bool (driver-dependent).
    ``video999999`` should not exist on any test machine: missing nodes hit the
    OSError path and must be treated as not loopback, with no webcam required.
    """
    assert isinstance(linux_is_v4l2_loopback("video0"), bool)
    assert linux_is_v4l2_loopback("video999999") is False


def test_linux_non_loopback_indices_are_non_negative_ints() -> None:
    if sys.platform != "linux":
        pytest.skip("Linux V4L2 sysfs")
    indices = linux_non_loopback_indices()
    assert all(isinstance(i, int) and i >= 0 for i in indices)


def test_resolve_candidates_explicit() -> None:
    assert _resolve_candidates("1") == [1]
    assert _resolve_candidates("/dev/video2") == ["/dev/video2"]


def _patch_linux_v4l_sysfs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    basenames: list[str],
    video_is_loopback: Callable[[str], bool],
) -> None:
    """Force Linux branch and fake sysfs enumeration + driver identity."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "laughing_man.camera._linux_list_video_basenames",
        lambda: basenames,
    )
    monkeypatch.setattr(
        "laughing_man.camera.linux_is_v4l2_loopback",
        video_is_loopback,
    )


def test_resolve_candidates_auto_linux_skips_loopback_prefers_next_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``auto`` should try video1 when video0 is the only loopback node."""
    _patch_linux_v4l_sysfs(
        monkeypatch,
        basenames=["video0", "video1"],
        video_is_loopback=lambda name: name == "video0",
    )
    assert _resolve_candidates("auto") == [1]
    assert linux_non_loopback_indices() == [1]


def test_resolve_candidates_auto_linux_only_loopback_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``auto`` fails when sysfs lists devices but none are usable capture nodes."""
    _patch_linux_v4l_sysfs(
        monkeypatch,
        basenames=["video0"],
        video_is_loopback=lambda name: True,
    )
    with pytest.raises(typer.Exit) as exc_info:
        _resolve_candidates("auto")
    assert exc_info.value.exit_code == 1


def test_resolve_candidates_auto_linux_empty_sysfs_falls_back_to_index_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No sysfs nodes: fall back to trying index 0 (same as missing sysfs)."""
    _patch_linux_v4l_sysfs(
        monkeypatch,
        basenames=[],
        video_is_loopback=lambda name: False,
    )
    assert _resolve_candidates("auto") == [0]


def test_refuse_loopback_before_open_index_raises_when_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "laughing_man.camera.linux_is_v4l2_loopback",
        lambda name: name == "video0",
    )
    with pytest.raises(typer.Exit) as exc_info:
        _refuse_loopback_before_open(0)
    assert exc_info.value.exit_code == 1


def test_refuse_loopback_before_open_path_raises_when_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "laughing_man.camera.linux_is_v4l2_loopback",
        lambda name: name == "video0",
    )
    with pytest.raises(typer.Exit) as exc_info:
        _refuse_loopback_before_open("/dev/video0")
    assert exc_info.value.exit_code == 1


def test_refuse_loopback_before_open_allows_non_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "laughing_man.camera.linux_is_v4l2_loopback",
        lambda name: False,
    )
    _refuse_loopback_before_open(0)
    _refuse_loopback_before_open("/dev/video0")


def test_refuse_loopback_before_open_skipped_off_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Linux: never consult sysfs; loopback guard is a no-op."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        "laughing_man.camera.linux_is_v4l2_loopback",
        lambda name: True,
    )
    _refuse_loopback_before_open(0)
