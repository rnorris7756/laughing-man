"""Privacy / no-face full-frame effects."""

from __future__ import annotations

import cv2
import numpy as np

from laughing_man.constants import PRIVACY_BLUR_DOWNSCALE, PRIVACY_BLUR_MAX_SIGMA


class GaussianBlurPrivacy:
    """
    Blend the frame toward a heavily Gaussian-blurred copy (full-frame privacy).

    By default blur is applied to a downscaled image (see ``PRIVACY_BLUR_DOWNSCALE``)
    with sigma scaled to match the original spatial extent, then upscaled — faster
    than blurring the full-resolution frame.
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
        blurred = _privacy_blurred_frame(frame)
        cv2.addWeighted(frame, 1.0 - amount, blurred, amount, 0, dst=frame)


def _privacy_blurred_frame(frame: np.ndarray) -> np.ndarray:
    """
    Build a full-size blurred copy of ``frame`` for privacy compositing.

    Uses downscale → Gaussian blur → upscale when ``PRIVACY_BLUR_DOWNSCALE`` < 1,
    with blur sigma ``PRIVACY_BLUR_MAX_SIGMA * scale`` on the small image so the
    effective blur radius in original pixels stays comparable.

    Parameters
    ----------
    frame
        BGR uint8 image.

    Returns
    -------
    numpy.ndarray
        BGR uint8, same shape as ``frame``.
    """
    scale = PRIVACY_BLUR_DOWNSCALE
    if scale >= 1.0:
        return cv2.GaussianBlur(frame, (0, 0), PRIVACY_BLUR_MAX_SIGMA)

    h, w = frame.shape[:2]
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    sigma_small = float(PRIVACY_BLUR_MAX_SIGMA) * scale
    blurred_small = cv2.GaussianBlur(small, (0, 0), sigma_small)
    return cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
