"""Per-session pipeline wiring (extensible)."""

from __future__ import annotations

from dataclasses import dataclass

from laughing_man.protocols import PrivacyEffect


@dataclass
class PipelineDeps:
    """
    Concrete implementations selected for one :func:`~laughing_man.run.run_overlay` run.

    Parameters
    ----------
    privacy
        Full-frame privacy treatment when no face is detected (after debounce).
    """

    privacy: PrivacyEffect
