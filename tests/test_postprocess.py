"""Tests for offline postprocess helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from laughing_man.postprocess import default_output_path, infer_media_kind


def test_default_output_path_image() -> None:
    """Default image output keeps suffix and adds _overlay."""
    p = Path("/tmp/shot.JPEG")
    out = default_output_path(p, "image")
    assert out == Path("/tmp/shot_overlay.JPEG")


def test_default_output_path_video() -> None:
    """Default video output is sibling MP4 with _overlay."""
    p = Path("/data/clips/take.mov")
    out = default_output_path(p, "video")
    assert out == Path("/data/clips/take_overlay.mp4")


def test_infer_media_kind_image_suffix(tmp_path: Path) -> None:
    """Raster suffixes are classified as image without decoding."""
    f = tmp_path / "x.png"
    f.write_bytes(b"not really png")
    assert infer_media_kind(f) == "image"


def test_infer_media_kind_unknown(tmp_path: Path) -> None:
    """Non-image suffix that OpenCV cannot open as video exits with an error."""
    f = tmp_path / "empty.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(typer.Exit):
        infer_media_kind(f)
