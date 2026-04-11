"""Pluggable pipeline interfaces (structural typing)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FaceBoxSource(Protocol):
    """Produces a raw face bounding box from a BGR frame."""

    def face_box(
        self, frame: np.ndarray, timestamp_ms: int
    ) -> tuple[int, int, int, int] | None:
        """
        Parameters
        ----------
        frame
            BGR uint8 image.
        timestamp_ms
            Monotonic milliseconds for video-mode detectors.

        Returns
        -------
        tuple[int, int, int, int] | None
            ``(x, y, w, h)`` in pixels, or None if no face.
        """
        ...


class PrivacyEffect(Protocol):
    """Full-frame effect when privacy mode is active (e.g. blur)."""

    def apply(self, frame: np.ndarray, amount: float) -> None:
        """
        Mutate ``frame`` in place according to ``amount`` in ``[0, 1]``.

        Parameters
        ----------
        frame
            BGR uint8 image.
        amount
            Effect strength; ``0`` means no change.
        """
        ...
