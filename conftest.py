"""Project-wide pytest setup for isolated temporary test artifacts."""

import os
from uuid import uuid4

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Give each run its own base before pytest creates its temp factory.

    On Windows, different OS accounts can inherit the same USERNAME and
    otherwise collide in pytest-of-<user>, whose permissions are private.
    Explicit CLI and environment temporary-directory choices take priority.
    """
    if config.option.basetemp is not None or os.environ.get("PYTEST_DEBUG_TEMPROOT"):
        return

    temp_root = config.rootpath / ".pytest_tmp"
    temp_root.mkdir(exist_ok=True)
    config.option.basetemp = str(temp_root / f"run-{uuid4().hex}")
