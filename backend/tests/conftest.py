"""Test fixtures.

Every test runs against a throwaway SQLite database and data directory, with the
same code paths as production: real migrations, real Argon2 hashing, real
sessions, real CSRF middleware. Nothing here mocks authentication.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Configure the environment *before* the application package is imported, because
# settings, the engine and storage are all process-level singletons.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="bv_tests_"))
os.environ.update({
    "APP_ENV": "test",
    "DATA_DIR": str(_TEST_ROOT / "data"),
    "DATABASE_URL": f"sqlite:///{_TEST_ROOT / 'data' / 'test.db'}",
    "APP_SECRET": "test-secret-key-for-buildvisual-suite-0123456789",
    "PUBLIC_BASE_URL": "http://testserver",
    "ENGINEER_EMAIL": "Engineer@Test.Example ",
    "ENGINEER_PASSWORD": "correct-horse-battery-staple",
    "ENGINEER_NAME": "Test Engineer",
    "VIDEO_PROVIDER": "local_render",
    "LLM_PROVIDER": "none",
    "TRANSCRIPTION_ENABLED": "false",
    "RUN_WORKER_INLINE": "false",
    "MAX_SHOTS": "2",
    "SHOT_SECONDS": "4",
    # Rendering too small makes every clip genuinely soft, which the quality gate
    # correctly rejects. These are small but realistic render sizes.
    "REFERENCE_WIDTH": "480",
    "REFERENCE_HEIGHT": "270",
    "VIDEO_WIDTH": "640",
    "VIDEO_HEIGHT": "360",
    "LOCAL_RENDER_FPS": "8",
    "LOCAL_RENDER_SCALE": "0.6",
    "LOGIN_MAX_ATTEMPTS": "5",
    "LOGIN_WINDOW_MINUTES": "15",
    "LOGIN_LOCKOUT_MINUTES": "15",
})


@pytest.fixture(scope="session", autouse=True)
def _cleanup():
    yield
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def fresh_app():
    """Reset process-level singletons and rebuild the schema for each test."""
    from app import config, db, storage

    config.reset_settings_cache()
    db.reset_engine()
    storage.reset_storage()
    (Path(os.environ["DATA_DIR"])).mkdir(parents=True, exist_ok=True)
    db.migrate()

    from app.models import User
    from app.security import ensure_engineer_account

    with db.session_scope() as session:
        ensure_engineer_account(session)
        session.commit()
        assert session.query(User).count() >= 1

    yield
    db.reset_engine()
    storage.reset_storage()
    # Fresh database per test: delete the file so ids/limits never leak across tests.
    db_path = Path(os.environ["DATABASE_URL"].replace("sqlite:///", ""))
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db_path) + suffix)
        if candidate.exists():
            candidate.unlink()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(), base_url="http://testserver") as test_client:
        yield test_client


@pytest.fixture
def db_session():
    from app.db import session_scope

    with session_scope() as session:
        yield session


@pytest.fixture
def login(client):
    """Sign in and return the client with cookies + CSRF header primed."""

    def _login(email: str = "engineer@test.example", password: str = "correct-horse-battery-staple"):
        response = client.post("/api/auth/login", json={"email": email, "password": password})
        assert response.status_code == 200, response.text
        data = response.json()
        client.headers.update({"X-CSRF-Token": data["csrf_token"]})
        return data

    return _login
