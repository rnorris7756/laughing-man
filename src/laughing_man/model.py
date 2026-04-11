"""BlazeFace model path resolution and download."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import typer
from loguru import logger

from laughing_man.constants import (
    BLAZE_FACE_FULL_RANGE_URL,
    BLAZE_FACE_SHORT_RANGE_URL,
    MODEL_ENV,
)


def cache_dir() -> Path:
    """Return XDG cache dir for downloaded models."""
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".cache"
    return root / "laughing-man"


def default_model_path(full_range: bool) -> Path:
    """Default cache path for the bundled BlazeFace variant."""
    name = (
        "blaze_face_full_range.tflite" if full_range else "blaze_face_short_range.tflite"
    )
    return cache_dir() / name


def resolve_model(full_range: bool) -> tuple[Path, str | None]:
    """
    Return (path, download_url or None).

    If ``LAUGHING_MAN_FACE_MODEL`` is set, that path is used and no download URL
    is applied (the file must already exist).
    """
    env = os.environ.get(MODEL_ENV, "").strip()
    if env:
        return Path(env), None
    url = BLAZE_FACE_FULL_RANGE_URL if full_range else BLAZE_FACE_SHORT_RANGE_URL
    return default_model_path(full_range), url


def ensure_blaze_face_model(path: Path, url: str | None) -> None:
    """Download BlazeFace if a URL is known and the file is missing."""
    if path.exists():
        return
    if url is None:
        logger.error(
            "Face model not found at {} ({} is set; place a .tflite there).",
            path,
            MODEL_ENV,
        )
        raise typer.Exit(code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    logger.info("Downloading face detector model to {} ...", path)
    try:
        urllib.request.urlretrieve(url, tmp)
    except OSError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        logger.error("Could not download model: {}", e)
        raise typer.Exit(code=1) from e
    tmp.replace(path)
