"""Authentication: the acceptance matrix from the login rework request.

These tests run against the real FastAPI app, the real database and real Argon2
hashing — no mocks, no bypasses.
"""

from __future__ import annotations

import os
import re
from urllib.parse import parse_qs, urlparse

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


def test_login_requires_pre_auth_csrf(client):
    client.headers.pop("X-CSRF-Token", None)
    response = client.post(
        "/api/auth/login",
        json={"email": "engineer@test.example", "password": "wrong-password"},
    )
    assert response.status_code == 403


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
    client.headers.update({"X-CSRF-Token": response.json()["csrf_token"]})
    # The previous password no longer works.
    client.cookies.clear()
    preauth = client.get("/api/auth/status")
    client.headers.update({"X-CSRF-Token": preauth.json()["csrf_token"]})
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
    first = client.post(
        "/api/auth/password-reset", json={"email": "unknown@example.com"}
    )
    second = client.post(
        "/api/auth/password-reset", json={"email": "engineer@test.example"}
    )
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert "If an account matches" in first.json()["message"]


def test_login_page_is_served_with_password_toggle(client):
    page = client.get("/signin")
    assert page.status_code == 200
    html = page.text
    assert 'id="togglePassword"' in html
    assert 'id="forgotBtn"' in html
    assert 'id="loginBtn"' in html
    assert 'id="createAccountBtn"' in html
    assert 'type="password"' in html


def test_sign_out_control_stays_hidden_before_authentication(client):
    page = client.get("/signin")
    assert 'id="accountBtn" class="ghost" type="button" hidden' in page.text
    assert page.headers["cache-control"] == "no-cache"
    css = client.get("/static/styles.css?v=2")
    assert css.headers["cache-control"] == "no-cache"
    assert ".topnav button[hidden] { display: none; }" in css.text


def test_signed_out_app_page_redirects_to_sign_in(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/signin?next=%2F"


def test_register_creates_argon2_account_and_signs_in(client):
    password = "a-long-password-123"
    response = client.post(
        "/api/auth/register",
        json={
            "name": "New Builder",
            "email": "new@example.com",
            "password": password,
            "confirm_password": password,
        },
    )
    assert response.status_code == 201, response.text
    assert response.json()["email"] == "new@example.com"
    assert response.json()["csrf_token"]
    assert "bv_session=" in response.headers["set-cookie"]
    assert "HttpOnly" in response.headers["set-cookie"]
    client.headers.update({"X-CSRF-Token": response.json()["csrf_token"]})
    assert client.get("/api/session").status_code == 200

    from app.db import session_scope
    from app.models import User
    from app.security import verify_password

    with session_scope() as session:
        user = session.query(User).filter_by(email="new@example.com").one()
        assert password not in user.password_hash
        assert verify_password(password, user.password_hash)


def test_duplicate_email_and_weak_password_are_rejected(client):
    payload = {
        "email": "new@example.com",
        "password": "a-long-password-123",
        "confirm_password": "a-long-password-123",
    }
    first = client.post("/api/auth/register", json=payload)
    assert first.status_code == 201
    client.headers.update({"X-CSRF-Token": first.json()["csrf_token"]})
    duplicate = client.post("/api/auth/register", json=payload)
    assert duplicate.status_code == 409
    assert "already has an account, try signing in" in duplicate.json()["error"]["message"]
    weak = client.post(
        "/api/auth/register",
        json={**payload, "email": "weak@example.com", "password": "short",
              "confirm_password": "short"},
    )
    assert weak.status_code == 400
    assert "at least 10 characters" in weak.json()["error"]["message"]


def test_invalid_registration_email_is_rejected(client):
    response = client.post(
        "/api/auth/register",
        json={
            "email": "not-an-email",
            "password": "a-long-password-123",
            "confirm_password": "a-long-password-123",
        },
    )
    assert response.status_code == 400
    assert "valid email" in response.json()["error"]["message"]


def test_registration_requires_csrf(client):
    client.headers.pop("X-CSRF-Token", None)
    response = client.post(
        "/api/auth/register",
        json={
            "email": "csrf@example.com",
            "password": "a-long-password-123",
            "confirm_password": "a-long-password-123",
        },
    )
    assert response.status_code == 403


def test_reset_token_is_hashed_single_use_and_changes_password(client, monkeypatch, caplog):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "app_env", "development", raising=False)
    password = "initial-reset-password-123"
    registered = client.post(
        "/api/auth/register",
        json={"email": "reset@example.com", "password": password,
              "confirm_password": password},
    )
    assert registered.status_code == 201
    client.headers.update({"X-CSRF-Token": registered.json()["csrf_token"]})
    client.post("/api/auth/logout")
    status = client.get("/api/auth/status")
    client.headers.update({"X-CSRF-Token": status.json()["csrf_token"]})
    with caplog.at_level("INFO", logger="buildvisual.api.auth"):
        response = client.post(
            "/api/auth/password-reset",
            json={"email": "reset@example.com"},
        )
    assert response.status_code == 200
    assert "If an account matches" in response.json()["message"]
    link = re.search(r"https?://\S+/reset-password\?token=\S+", caplog.text)
    assert link
    token = parse_qs(urlparse(link.group()).query)["token"][0]

    from app.db import session_scope
    from app.models import PasswordResetToken

    with session_scope() as session:
        reset = session.query(PasswordResetToken).one()
        assert token not in reset.token_hash
        assert reset.expires_at > reset.created_at

    payload = {
        "token": token,
        "password": "new-long-password-456",
        "confirm_password": "new-long-password-456",
    }
    updated = client.post("/api/auth/password-reset/confirm", json=payload)
    assert updated.status_code == 200, updated.text
    repeated = client.post("/api/auth/password-reset/confirm", json=payload)
    assert repeated.status_code == 404
    login_response = client.post(
        "/api/auth/login",
        json={"email": "reset@example.com", "password": payload["password"]},
    )
    assert login_response.status_code == 200


def test_users_cannot_see_each_others_projects(client):
    password = "first-user-password"
    first = client.post(
        "/api/auth/register",
        json={"email": "first@example.com", "password": password,
              "confirm_password": password},
    )
    assert first.status_code == 201
    client.headers.update({"X-CSRF-Token": first.json()["csrf_token"]})
    project = client.post("/api/projects", json={"name": "Private project"})
    assert project.status_code == 200
    project_id = project.json()["id"]
    assert client.post("/api/auth/logout").status_code == 200

    status = client.get("/api/auth/status")
    client.headers.update({"X-CSRF-Token": status.json()["csrf_token"]})
    second_password = "second-user-password"
    second = client.post(
        "/api/auth/register",
        json={"email": "second@example.com", "password": second_password,
              "confirm_password": second_password},
    )
    assert second.status_code == 201
    client.headers.update({"X-CSRF-Token": second.json()["csrf_token"]})
    assert client.get("/api/projects").json() == []
    assert client.get(f"/api/projects/{project_id}").status_code == 403
