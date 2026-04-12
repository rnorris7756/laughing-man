"""Tests for CLI entry (``app()``)."""

from __future__ import annotations

import sys

import pytest
import typer

from laughing_man.cli import app


@pytest.mark.parametrize("flag", ("--version", "-V"))
def test_version_exits_zero(
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Top-level ``--version`` / ``-V`` print the distribution version."""
    monkeypatch.setattr(sys, "argv", ["laughing-man", flag])
    with pytest.raises(typer.Exit) as exc_info:
        app()
    assert exc_info.value.exit_code == 0
    out = capsys.readouterr().out
    assert out.strip().startswith("laughing-man ")
    assert len(out.strip().split()) >= 2
