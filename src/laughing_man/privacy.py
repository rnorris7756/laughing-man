"""Privacy / no-face full-frame effects."""

from __future__ import annotations

import cv2
import numpy as np

from laughing_man.constants import PRIVACY_BLUR_MAX_SIGMA


class GaussianBlurPrivacy:
    """
    Blend the frame toward a heavily Gaussian-blurred copy (full-frame privacy).
    """

    def apply(self, frame: np.ndarray, amount: float) -> None:
        """
        Parameters
        ----------
        frame
            BGR uint8 image, updated in place.
        amount
            Blend weight for the blurred layer in ``[0, 1]``.
        """
        if amount <= 1e-6:
            return
        blurred = cv2.GaussianBlur(frame, (0, 0), PRIVACY_BLUR_MAX_SIGMA)
        cv2.addWeighted(frame, 1.0 - amount, blurred, amount, 0, dst=frame)
