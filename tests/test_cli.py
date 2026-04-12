"""Tests for CLI entry (``app()``)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from laughing_man.cli import app


@pytest.mark.parametrize("flag", ("--version", "-V"))
def test_version_exits_zero(
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Top-level ``--version`` / ``-V`` print the distribution version."""
    monkeypatch.setattr(sys, "argv", ["laughing-man", flag])
    with pytest.raises(SystemExit) as exc_info:
        app()
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert out.strip().startswith("laughing-man ")
    assert len(out.strip().split()) >= 2


def test_version_console_scripts_sys_exit_app_no_traceback() -> None:
    """Console scripts use ``sys.exit(app())``; ``-V`` must not leave a traceback."""
    code = (
        "import sys\n"
        "sys.argv = ['laughing-man', '-V']\n"
        "from laughing_man.cli import app\n"
        "sys.exit(app())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0
    assert "Traceback" not in result.stderr
    assert result.stdout.strip().startswith("laughing-man ")
