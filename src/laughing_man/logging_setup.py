"""Loguru configuration."""

from __future__ import annotations

import sys

from loguru import logger


def configure_logging(*, debug: bool) -> None:
    """
    Configure loguru: single stderr sink, INFO unless ``debug`` is True.

    Parameters
    ----------
    debug
        If True, emit DEBUG-level messages (e.g. overlay prefill diagnostics).
    """
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")
