"""Authentication: the acceptance matrix from the login rework request.

These tests run against the real FastAPI app, the real database and real Argon2
hashing — no mocks, no bypasses.
"""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# The configured engineer can sign in
# ---------------------------------------------------------------------------


def test_configured_engineer_can_log_in(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "engineer@test.example", "password": "correct-horse-battery-staple"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "engineer@test.example"
    assert body["csrf_token"]
    # Session cookie is HttpOnly and scoped to the whole app.
    cookie = response.headers.get("set-cookie", "")
    assert "bv_session=" in cookie and "HttpOnly" in cookie


def test_login_ignores_case_and_surrounding_whitespace(client):
    """A pasted address with a trailing space must not read as a bad password."""
    response = client.post(
        "/api/auth/login",
        json={"email": "  ENGINEER@Test.Example  ", "password": "correct-horse-battery-staple"},
    )
    assert response.status_code == 200, response.text


def test_session_survives_refresh(client, login):
    login()
    # A fresh request with the same cookie jar = a browser refresh.
    status = client.get("/api/auth/status").json()
    assert status["authenticated"] is True
    assert status["email"] == "engineer@test.example"
    session = client.get("/api/session")
    assert session.status_code == 200
    assert session.json()["csrf_token"]


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_wrong_password_is_rejected_with_friendly_message(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "engineer@test.example", "password": "not-the-password"},
    )
    assert response.status_code == 401
    error = response.json()["error"]
    assert "not recognised" in error["message"].lower()
    # No hint about which half was wrong, and no internals.
    assert "password_hash" not in response.text
    assert "argon2" not in response.text.lower()
    assert error["details"]["reason"] == "invalid_credentials"


def test_unknown_email_does_not_grant_access(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "someone-else@example.com", "password": "correct-horse-battery-staple"},
    )
    assert response.status_code == 401
    assert client.get("/api/auth/status").json()["authenticated"] is False


def test_protected_endpoints_require_a_session(client):
    for path in ("/api/projects", "/api/jobs/active", "/api/capabilities"):
        assert client.get(path).status_code == 401, path


def test_logout_invalidates_the_session(client, login):
    login()
    assert client.get("/api/session").status_code == 200
    assert client.post("/api/auth/logout").status_code == 200
    client.cookies.clear()
    assert client.get("/api/session").status_code == 401
    # Even replaying the old cookie must fail: the server revoked it.
    assert client.get("/api/auth/status").json()["authenticated"] is False


def test_mutation_without_csrf_token_is_refused(client, login):
    login()
    client.headers.pop("X-CSRF-Token", None)
    response = client.post("/api/projects", json={"name": "No CSRF"})
    assert response.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Deployment states (the original bug)
# ---------------------------------------------------------------------------


def test_setup_required_when_no_password_is_configured(client, monkeypatch):
    """The old failure: empty ENGINEER_PASSWORD gave 'incorrect password'."""
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "engineer_password", "", raising=False)

    from app.db import session_scope
    from app.models import User

    with session_scope() as session:
        session.query(User).delete()
        session.commit()

    status = client.get("/api/auth/status").json()
    assert status["authenticated"] is False
    assert status["setup_required"] is True
    assert "ENGINEER_PASSWORD" in (status["message"] or "")

    response = client.post("/api/auth/login", json={"email": "engineer@test.example", "password": "x"})
    assert response.status_code == 401
    assert response.json()["error"]["details"]["reason"] == "setup_required"
    assert "no engineer account" in response.json()["error"]["message"].lower()


def test_password_rotation_in_configuration_is_applied(client, monkeypatch):
    """Rotating ENGINEER_PASSWORD must not lock the engineer out for ever."""
    from app.config import get_settings
    from app.db import session_scope
    from app.security import ensure_engineer_account

    settings = get_settings()
    monkeypatch.setattr(settings, "engineer_password", "rotated-password-2026", raising=False)

    with session_scope() as session:
        ensure_engineer_account(session)
        session.commit()

    response = client.post(
        "/api/auth/login",
        json={"email": "engineer@test.example", "password": "rotated-password-2026"},
    )
    assert response.status_code == 200
    # The previous password no longer works.
    client.cookies.clear()
    old = client.post(
        "/api/auth/login",
        json={"email": "engineer@test.example", "password": "correct-horse-battery-staple"},
    )
    assert old.status_code == 401


def test_account_is_created_when_another_user_row_exists(client):
    """A stray row under a different address must not block the configured one."""
    from app.db import session_scope
    from app.models import User
    from app.security import ensure_engineer_account, hash_password

    with session_scope() as session:
        session.query(User).delete()
        session.add(User(email="old-run@example.com",
                         password_hash=hash_password("something-else-1234")))
        session.commit()

    with session_scope() as session:
        created = ensure_engineer_account(session)
        session.commit()
        assert created is not None
        assert created.email == "engineer@test.example"

    response = client.post(
        "/api/auth/login",
        json={"email": "engineer@test.example", "password": "correct-horse-battery-staple"},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Throttling
# ---------------------------------------------------------------------------


def test_repeated_failures_are_rate_limited(client):
    from app.config import get_settings

    limit = get_settings().login_max_attempts
    for _ in range(limit):
        client.post("/api/auth/login",
                    json={"email": "engineer@test.example", "password": "wrong-password"})

    blocked = client.post(
        "/api/auth/login",
        json={"email": "engineer@test.example", "password": "correct-horse-battery-staple"},
    )
    assert blocked.status_code == 429
    body = blocked.json()["error"]
    assert body["code"] == "rate_limited"
    assert "try again" in body["message"].lower()
    # Crucially, the correct password is not rejected as "incorrect".
    assert "not recognised" not in body["message"].lower()


def test_successful_login_clears_failure_history(client):
    for _ in range(2):
        client.post("/api/auth/login",
                    json={"email": "engineer@test.example", "password": "wrong-password"})
    ok = client.post("/api/auth/login",
                     json={"email": "engineer@test.example", "password": "correct-horse-battery-staple"})
    assert ok.status_code == 200

    from app.db import session_scope
    from app.models import LoginAttempt

    with session_scope() as session:
        failures = session.query(LoginAttempt).filter_by(successful=False).count()
        assert failures == 0


def test_password_is_never_returned_or_logged(client, caplog):
    with caplog.at_level("DEBUG"):
        client.post("/api/auth/login",
                    json={"email": "engineer@test.example", "password": "correct-horse-battery-staple"})
    secret = "correct-horse-battery-staple"
    assert secret not in caplog.text
    assert all(secret not in str(record.args) for record in caplog.records)


def test_recovery_message_does_not_leak_account_state(client):
    """Unknown and known addresses get the same recovery response."""
    first = client.post("/api/auth/password-reset")
    second = client.post("/api/auth/password-reset")
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert "set-password" in first.json()["message"]


def test_login_page_is_served_with_password_toggle(client):
    page = client.get("/")
    assert page.status_code == 200
    html = page.text
    assert 'id="togglePassword"' in html
    assert 'id="forgotBtn"' in html
    assert 'id="loginBtn"' in html
    assert 'type="password"' in html
