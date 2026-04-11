#
# Laughing Man webcam overlay (Ghost in the Shell style).
# MediaPipe BlazeFace + OpenCV + Pillow.
#
# Original face-detection demo: Jouni Paulus, 2010 (pyOpenCV + PIL).
#

from __future__ import annotations

from laughing_man.bootstrap import apply_runtime_env

apply_runtime_env()

from laughing_man.cli import app  # noqa: E402

__all__ = ["app"]
