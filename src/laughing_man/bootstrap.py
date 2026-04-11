"""Process environment defaults before OpenCV / MediaPipe import."""

from __future__ import annotations

import os
import sys


def apply_runtime_env() -> None:
    """
    Set process environment defaults to reduce Qt / TFLite console noise.

    Only uses :func:`os.environ.setdefault` so explicit user exports win.
    Intended to run once at import, before ``cv2`` or ``mediapipe``.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")

    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
        for font_root in ("/usr/share/fonts", "/usr/local/share/fonts"):
            if os.path.isdir(font_root):
                os.environ.setdefault("QT_QPA_FONTDIR", font_root)
                break
        qt_rules_extra = "*.debug=false;qt.qpa.*=false"
        prev_rules = os.environ.get("QT_LOGGING_RULES", "").strip()
        if qt_rules_extra not in prev_rules:
            os.environ["QT_LOGGING_RULES"] = (
                f"{prev_rules};{qt_rules_extra}" if prev_rules else qt_rules_extra
            )
