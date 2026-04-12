"""Regression tests for Typer/Click CLI construction."""

from __future__ import annotations

from typer.main import get_command

from laughing_man.cli import typer_app


def test_typer_app_click_group_builds() -> None:
    """
    Importing the app must allow building the underlying Click command tree.

    A Typer 0.24+ regression: ``Annotated[..., typer.Option(False, \"--opt\", ...)]``
    passes ``False`` into Click's option declarations, causing
    ``AttributeError: 'bool' object has no attribute 'isidentifier'`` when the
    CLI is first materialized. Defaults should live on the function parameter
    (``= False``) with only flag names inside ``typer.Option(...)``.

    This is intentionally stricter than ``app()`` with ``--version``, which exits
    before ``typer_app()`` runs.
    """
    get_command(typer_app)
