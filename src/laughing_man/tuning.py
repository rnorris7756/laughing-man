"""Interactive lambda tuning (terminal and OpenCV preview)."""

from __future__ import annotations

import os
import select
import sys
import threading
from typing import Callable

from loguru import logger

from laughing_man.constants import (
    LAMBDA_TUNE_STEP,
    _ARROW_KEYS_ROI_DOWN,
    _ARROW_KEYS_ROI_UP,
    _ARROW_KEYS_SIZE_LEFT,
    _ARROW_KEYS_SIZE_RIGHT,
    _QUIT_PREVIEW_KEY_CODES,
)


def lambda_deltas_from_arrow_key(key: int) -> tuple[float, float] | None:
    """
    Map an OpenCV ``waitKeyEx`` code to tuning deltas for roi-lambda and size-lambda.

    Up/down adjust roi-lambda; left/right adjust size-lambda. Returns ``None`` if
    ``key`` is not a recognized arrow code.

    Parameters
    ----------
    key
        Value from :func:`cv2.waitKeyEx` (non-negative when a key was pressed).

    Returns
    -------
    tuple[float, float] | None
        ``(delta_roi_lambda, delta_size_lambda)``, each 0 if that axis does not
        apply, or ``None`` if ``key`` is not an arrow.
    """
    if key in _ARROW_KEYS_ROI_UP:
        return (LAMBDA_TUNE_STEP, 0.0)
    if key in _ARROW_KEYS_ROI_DOWN:
        return (-LAMBDA_TUNE_STEP, 0.0)
    if key in _ARROW_KEYS_SIZE_LEFT:
        return (0.0, -LAMBDA_TUNE_STEP)
    if key in _ARROW_KEYS_SIZE_RIGHT:
        return (0.0, LAMBDA_TUNE_STEP)
    return None


def stdin_interactive_tuning_available() -> bool:
    """
    Return True if arrow / quit tuning can be read from the controlling terminal.

    Requires an interactive TTY on stdin and a supported platform API
    (POSIX ``termios`` or Windows ``msvcrt``).
    """
    if not sys.stdin.isatty():
        return False
    if sys.platform == "win32":
        return True
    try:
        import termios  # noqa: F401
    except ImportError:
        return False
    return True


def terminal_stdin_tune_loop(
    listener_stop: threading.Event,
    user_quit: threading.Event,
    apply_deltas: Callable[[float, float], None],
) -> None:
    """
    Background loop: read arrow keys and Ctrl+Q from the terminal.

    Parameters
    ----------
    listener_stop
        When set, exit the loop and return (main loop is shutting down).
    user_quit
        Set when the user presses Ctrl+Q (ASCII DC1) in the terminal reader.
    apply_deltas
        ``(delta_roi_lambda, delta_size_lambda)`` — same semantics as
        :func:`lambda_deltas_from_arrow_key` results.
    """
    try:
        if sys.platform == "win32":
            windows_console_tune_loop(listener_stop, user_quit, apply_deltas)
        else:
            posix_stdin_tune_loop(listener_stop, user_quit, apply_deltas)
    except Exception as e:
        logger.warning("Terminal key listener stopped ({}).", e)


def posix_stdin_tune_loop(
    listener_stop: threading.Event,
    user_quit: threading.Event,
    apply_deltas: Callable[[float, float], None],
) -> None:
    """POSIX ``termios`` cbreak reader for CSI/SS3 arrows and Ctrl+Q."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not listener_stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.12)
            if not r:
                continue
            b0 = os.read(fd, 1)
            if not b0:
                break
            if b0 == b"\x1b":
                r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r2:
                    continue
                b1 = os.read(fd, 1)
                if b1 == b"[":
                    r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not r3:
                        continue
                    b2 = os.read(fd, 1)
                    if b2 == b"A":
                        apply_deltas(LAMBDA_TUNE_STEP, 0.0)
                    elif b2 == b"B":
                        apply_deltas(-LAMBDA_TUNE_STEP, 0.0)
                    elif b2 == b"C":
                        apply_deltas(0.0, LAMBDA_TUNE_STEP)
                    elif b2 == b"D":
                        apply_deltas(0.0, -LAMBDA_TUNE_STEP)
                elif b1 == b"O":
                    r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not r3:
                        continue
                    b2 = os.read(fd, 1)
                    if b2 == b"A":
                        apply_deltas(LAMBDA_TUNE_STEP, 0.0)
                    elif b2 == b"B":
                        apply_deltas(-LAMBDA_TUNE_STEP, 0.0)
                    elif b2 == b"C":
                        apply_deltas(0.0, LAMBDA_TUNE_STEP)
                    elif b2 == b"D":
                        apply_deltas(0.0, -LAMBDA_TUNE_STEP)
                continue
            if b0 == b"\x11":
                user_quit.set()
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def windows_console_tune_loop(
    listener_stop: threading.Event,
    user_quit: threading.Event,
    apply_deltas: Callable[[float, float], None],
) -> None:
    """Windows console: extended-prefix arrow keys and Ctrl+Q."""
    import msvcrt

    while not listener_stop.is_set():
        if msvcrt.kbhit():
            c = msvcrt.getch()
            if c in (b"\xe0", b"\x00"):
                c2 = msvcrt.getch()
                if c2 == b"H":
                    apply_deltas(LAMBDA_TUNE_STEP, 0.0)
                elif c2 == b"P":
                    apply_deltas(-LAMBDA_TUNE_STEP, 0.0)
                elif c2 == b"K":
                    apply_deltas(0.0, -LAMBDA_TUNE_STEP)
                elif c2 == b"M":
                    apply_deltas(0.0, LAMBDA_TUNE_STEP)
            elif c == b"\x11":
                user_quit.set()
                return
        else:
            listener_stop.wait(0.05)


def should_quit_preview(key: int) -> bool:
    """
    Return True if this OpenCV ``waitKeyEx`` code should exit the preview loop.

    Handles Ctrl+C and Ctrl+Q as control-character key events when the preview
    window has focus (ASCII 3 and 17). When the terminal has focus, Ctrl+C
    still stops the process via ``KeyboardInterrupt``. Cmd+Q on macOS depends
    on the OpenCV GUI backend and may not be reported as a distinct code; use
    Ctrl+Q or terminal Ctrl+C if needed. Other keys are ignored unless handled
    elsewhere (e.g. arrow tuning).

    Parameters
    ----------
    key
        Value from :func:`cv2.waitKeyEx`, or ``-1`` if no key was pressed.

    Returns
    -------
    bool
        True if the user chose to quit from the preview window.
    """
    return key >= 0 and key in _QUIT_PREVIEW_KEY_CODES
