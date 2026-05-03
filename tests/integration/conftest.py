"""Shared setup for integration tests — sets env vars before any imports."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Must be set before any restorax module is imported so Settings picks them up.
os.environ["RESTORAX_DATABASE_URL"] = "sqlite+aiosqlite:///./test_restorax.db"
os.environ["RESTORAX_REDIS_URL"] = "redis://localhost:6379/0"
os.environ["RESTORAX_DEVICE"] = "cpu"
os.environ["RESTORAX_STORAGE_LOCAL_ROOT"] = "/tmp/restorax_test_data"
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "1"


@pytest.fixture(scope="session", autouse=True)
def _cleanup_integration_db():
    yield
    Path("test_restorax.db").unlink(missing_ok=True)
