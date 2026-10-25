"""Authentication, session handling and capability tokens.

Scope is deliberately one engineer, but the blueprint (p.79, p.82) is explicit
that single-user does not mean unauthenticated. Therefore:

* Argon2id password hashing (``argon2-cffi``), never plaintext, never SHA.
* Opaque random session tokens; only their SHA-256 is stored.
* HttpOnly, SameSite=Lax, Secure-when-HTTPS session cookie.
* Double-submit CSRF token for every mutating request.
* Constant-time comparison everywhere.
* Share links carry a signed, hashed, expiring token — the raw token is never
  persisted, so a database leak alone cannot mint working links.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db, utcnow
from .errors import AuthError, PermissionDenied
from .models import AuthSession, LoginAttempt, User

log = logging.getLogger("buildvisual.security")

SESSION_COOKIE = "bv_session"
CSRF_COOKIE = "bv_csrf"
CSRF_HEADER = "x-csrf-token"

_hasher = PasswordHasher(time_cost=3, memory_cost=64 * 1024, parallelism=4)


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    if len(password) < 10:
        raise ValueError("Password must be at least 10 characters")
    return _hasher.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        return _hasher.verify(stored_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except InvalidHashError:  # pragma: no cover
        return True


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def new_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def sign_share_token(raw: str) -> str:
    """HMAC the token with the app secret so a DB-only leak is insufficient."""
    settings = get_settings()
    digest = hmac.new(
        settings.resolved_secret.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return _sha256(raw + digest)


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a or "", b or "")


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def create_session(
    db: Session, user: User, *, user_agent: str | None = None, ip_hint: str | None = None
) -> tuple[str, AuthSession]:
    settings = get_settings()
    raw = new_token()
    session = AuthSession(
        token_hash=_sha256(raw),
        user_id=user.id,
        csrf_token=new_token(24),
        expires_at=utcnow() + timedelta(hours=settings.session_ttl_hours),
        user_agent=(user_agent or "")[:255] or None,
        ip_hint=(ip_hint or "")[:64] or None,
    )
    db.add(session)
    db.flush()
    return raw, session


def get_session_by_token(db: Session, raw_token: str) -> AuthSession | None:
    if not raw_token:
        return None
    row = db.execute(
        select(AuthSession).where(AuthSession.token_hash == _sha256(raw_token))
    ).scalar_one_or_none()
    if row is None or row.revoked_at is not None or row.expires_at <= utcnow():
        return None
    return row


def revoke_session(db: Session, raw_token: str) -> None:
    session = get_session_by_token(db, raw_token)
    if session:
        session.revoked_at = utcnow()


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def current_user_optional(request: Request, db: Session) -> User | None:
    token = request.cookies.get(SESSION_COOKIE)
    session = get_session_by_token(db, token or "")
    if session is None:
        return None
    user = db.get(User, session.user_id)
    if user is None or not user.is_active:
        return None
    request.state.auth_session = session
    request.state.user = user
    return user


def require_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = current_user_optional(request, db)
    if user is None:
        raise AuthError()
    return user


def require_csrf(request: Request) -> None:
    """Double-submit check for state-changing requests."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    session = getattr(request.state, "auth_session", None)
    if session is None:
        raise AuthError()
    header = request.headers.get(CSRF_HEADER, "")
    if not constant_time_equals(header, session.csrf_token):
        raise PermissionDenied("CSRF token missing or invalid.")


def normalise_email(value: str | None) -> str:
    """Address handling shared by seeding and login.

    People paste addresses with a trailing space and type mixed case; neither
    should ever produce "incorrect password". Only leading/trailing whitespace is
    removed (never internal), and case is folded for lookup only — the stored
    address keeps its original form.
    """
    return (value or "").strip().lower()


def authenticate(db: Session, email: str, password: str) -> User:
    """Verify credentials, tolerating whitespace/case in the address."""
    normalised = normalise_email(email)
    user = db.execute(select(User).where(User.email == normalised)).scalar_one_or_none()
    if user is None or not user.is_active:
        # Perform a dummy verify so response timing does not reveal whether the
        # address exists.
        verify_password(password, _hasher.hash("timing-equaliser"))
        raise AuthError("Those sign-in details were not recognised.")
    if not verify_password(password, user.password_hash):
        raise AuthError("Those sign-in details were not recognised.")
    if needs_rehash(user.password_hash):  # pragma: no cover - upgrade path
        user.password_hash = hash_password(password)
    return user


def ensure_engineer_account(db: Session) -> User | None:
    """Create or reconcile the single engineer account from configuration.

    The three bugs this replaces all produced the same misleading symptom
    ("incorrect password"):

    * it keyed off "does *any* user exist", so a row created under a different
      address in an earlier run permanently blocked the configured account;
    * it silently did nothing when ``ENGINEER_PASSWORD`` was empty, leaving the
      deployment with no account at all;
    * a password rotated in configuration never reached the stored hash.

    The configured address is now the identity, and (with
    ``ENGINEER_PASSWORD_SYNC``) the configured password is authoritative.
    """
    settings = get_settings()
    email = normalise_email(settings.engineer_email)
    password = settings.engineer_password or ""

    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()

    if user is None:
        if not password:
            log.warning(
                "no engineer account exists and ENGINEER_PASSWORD is empty; "
                "sign-in is disabled until it is configured"
            )
            return None
        user = User(
            email=email,
            password_hash=hash_password(password),
            display_name=settings.engineer_name or "Engineer",
        )
        db.add(user)
        db.flush()
        log.info("created the engineer account for %s", email)
        return user

    if not user.is_active:
        user.is_active = True

    if password and settings.engineer_password_sync and not verify_password(password, user.password_hash):
        user.password_hash = hash_password(password)
        db.flush()
        log.info(
            "ENGINEER_PASSWORD changed in configuration; the stored password for %s "
            "was updated to match",
            email,
        )
    return user


def set_session_cookies(response, raw_token: str, session: AuthSession) -> None:
    settings = get_settings()
    max_age = int((session.expires_at - utcnow()).total_seconds())
    common = {
        "max_age": max_age,
        "path": "/",
        "samesite": "lax",
        "secure": settings.cookie_secure,
    }
    response.set_cookie(SESSION_COOKIE, raw_token, httponly=True, **common)
    # Readable by the SPA so it can echo the double-submit CSRF header.
    response.set_cookie(CSRF_COOKIE, session.csrf_token, httponly=False, **common)


def set_pre_auth_csrf_cookie(response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        CSRF_COOKIE,
        token,
        max_age=3600,
        path="/",
        httponly=False,
        samesite="lax",
        secure=settings.cookie_secure,
    )


def clear_session_cookies(response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")


# ---------------------------------------------------------------------------
# Login throttling
# ---------------------------------------------------------------------------


class LoginLocked(Exception):
    """Raised when too many failed attempts were made for this email + IP."""

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("Too many sign-in attempts.")
        self.retry_after_seconds = max(1, int(retry_after_seconds))


def setup_required(db: Session) -> bool:
    """True when the deployment has no engineer account yet.

    This reports a *deployment* state, not a fact about any particular
    address, so it leaks nothing about real credentials.
    """
    return db.execute(select(User.id).limit(1)).scalar_one_or_none() is None


def _recent_failures(db: Session, email: str, ip_hint: str, since: datetime):
    return db.execute(
        select(LoginAttempt)
        .where(
            LoginAttempt.email == email,
            LoginAttempt.ip_hint == ip_hint,
            LoginAttempt.successful.is_(False),
            LoginAttempt.created_at >= since,
        )
        .order_by(LoginAttempt.created_at)
    ).scalars().all()


def check_login_allowed(db: Session, email: str, ip_hint: str) -> None:
    """Raise ``LoginLocked`` if this email + IP has exhausted its attempts."""
    settings = get_settings()
    window_start = utcnow() - timedelta(minutes=settings.login_window_minutes)
    failures = _recent_failures(db, email, ip_hint, window_start)
    if len(failures) < settings.login_max_attempts:
        return
    locked_until = failures[-1].created_at + timedelta(minutes=settings.login_lockout_minutes)
    remaining = (locked_until - utcnow()).total_seconds()
    if remaining > 0:
        raise LoginLocked(remaining)
    # Window has run out: clear it so the engineer is not stuck for ever.
    for row in failures:
        db.delete(row)
    db.flush()


def record_login_attempt(db: Session, email: str, ip_hint: str, *, successful: bool) -> None:
    db.add(LoginAttempt(email=email, ip_hint=ip_hint, successful=successful))
    if successful:
        # A good sign-in clears the failure history for this identity.
        for row in _recent_failures(
            db, email, ip_hint, utcnow() - timedelta(days=1)
        ):
            db.delete(row)
    db.flush()


def prune_login_attempts(db: Session, *, older_than_days: int = 7) -> int:
    rows = db.execute(
        select(LoginAttempt).where(
            LoginAttempt.created_at < utcnow() - timedelta(days=older_than_days)
        )
    ).scalars().all()
    for row in rows:
        db.delete(row)
    db.flush()
    return len(rows)


def session_expiry() -> datetime:
    return utcnow() + timedelta(hours=get_settings().session_ttl_hours)
