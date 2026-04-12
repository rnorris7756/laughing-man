"""Ensure in-tree version strings stay aligned."""

from __future__ import annotations

from pathlib import Path

import tomllib

from laughing_man.__version__ import __version__


def test_version_matches_pyproject() -> None:
    """``__version__.py`` must match ``[project].version`` (both updated by ``cz bump``)."""
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == data["project"]["version"]
