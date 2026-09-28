"""Shared pytest fixtures for the Python test suite.

Automated tests must never depend on a live external service. The local .env
file may have real Logfire/LangSmith credentials enabled for manual runs of the
app, but the test suite always forces observability off so results never depend
on network access or one person's personal cloud account.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_live_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")
