"""Ensure in-tree version strings stay aligned."""

from __future__ import annotations

from pathlib import Path

from laughing_man.__version__ import __version__


def _project_version(pyproject_toml: Path) -> str:
    """Return ``[project].version`` without ``tomllib`` (stdlib is 3.11+ only)."""
    text = pyproject_toml.read_text(encoding="utf-8")
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "[project]":
            in_project = True
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = False
            continue
        if not in_project or not stripped.startswith("version"):
            continue
        key, sep, rhs = stripped.partition("=")
        if sep and key.strip() == "version":
            return rhs.strip().strip('"').strip("'")
    raise AssertionError(f"No [project] version found in {pyproject_toml}")


def test_version_matches_pyproject() -> None:
    """``__version__.py`` must match ``[project].version`` (both updated by ``cz bump``)."""
    root = Path(__file__).resolve().parents[1]
    assert __version__ == _project_version(root / "pyproject.toml")
