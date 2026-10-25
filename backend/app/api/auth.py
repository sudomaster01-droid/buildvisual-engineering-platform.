"""Session endpoints for the single engineer account.

Design goals (from the login rework request):
* one email field, one password field, one button — nothing else required;
* every failure mode has a distinct, human explanation:
    - ``invalid_credentials`` — address/password not recognised
    - ``rate_limited``        — too many attempts, wait
    - ``setup_required``      — the deployment has no account yet
    - ``server_unreachable``  — produced client-side, not here
* no internal detail (database errors, stack traces) ever reaches the browser;
* the address is preserved on failure so the engineer does not retype it.
"""

from __future__ import annotations

import logging
import hashlib
import re
from datetime import timedelta

from fastapi import APIRouter, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..config import get_settings
from ..db import utcnow
from ..errors import AuthError, Conflict, NotFound, RateLimited, ValidationError
from ..models import AuthSession, PasswordResetToken, User
from ..security import (
    CSRF_COOKIE,
    SESSION_COOKIE,
    LoginLocked,
    authenticate,
    check_login_allowed,
    clear_session_cookies,
    create_session,
    current_user_optional,
    ensure_engineer_account,
    hash_password,
    new_token,
    normalise_email,
    record_login_attempt,
    revoke_session,
    set_pre_auth_csrf_cookie,
    set_session_cookies,
    setup_required,
)
from .deps import DbSession
from .schemas import (
    LoginIn,
    LoginStatusOut,
    MessageOut,
    PasswordResetConfirmIn,
    PasswordResetRequestIn,
    RegisterIn,
    SessionOut,
)

router = APIRouter(tags=["auth"])
log = logging.getLogger("buildvisual.api.auth")


def _ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "")[:64]


def _raise_if_locked(db, key: str, ip_hint: str) -> None:
    try:
        check_login_allowed(db, key, ip_hint)
    except LoginLocked as locked:
        minutes = max(1, round(locked.retry_after_seconds / 60))
        raise RateLimited(
            f"Too many requests. Please try again in about {minutes} minute"
            f"{'' if minutes == 1 else 's'}.",
            details={"retry_after_seconds": locked.retry_after_seconds},
        ) from locked


def _valid_email(email: str) -> bool:
    return bool(
        len(email) <= 254
        and re.fullmatch(
            r"[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
            r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
            r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+",
            email,
            flags=re.IGNORECASE,
        )
    )


@router.get("/auth/status", response_model=LoginStatusOut)
def login_status(request: Request, response: Response, db: DbSession) -> LoginStatusOut:
    """Cheap, always-200 probe the sign-in screen uses to choose its message.

    Returning 200 here removes the misleading 401 in the browser console and lets
    the UI distinguish "not signed in" from "server has no account configured".
    """
    user = current_user_optional(request, db)
    if user is not None:
        session = getattr(request.state, "auth_session", None)
        return LoginStatusOut(
            authenticated=True,
            setup_required=False,
            display_name=user.display_name,
            email=user.email,
            csrf_token=session.csrf_token if session else None,
        )
    csrf_token = request.cookies.get(CSRF_COOKIE) or new_token(24)
    set_pre_auth_csrf_cookie(response, csrf_token)
    needs_setup = setup_required(db)
    return LoginStatusOut(
        authenticated=False,
        setup_required=needs_setup,
        email_hint=get_settings().engineer_email if needs_setup else None,
        message=(
            "No engineer account is configured yet. Set ENGINEER_EMAIL and "
            "ENGINEER_PASSWORD on the server, then reload this page."
            if needs_setup
            else None
        ),
        csrf_token=csrf_token,
    )


@router.post("/auth/login", response_model=SessionOut)
def login(payload: LoginIn, request: Request, response: Response, db: DbSession) -> SessionOut:
    settings = get_settings()
    email = normalise_email(payload.email)
    ip_hint = _ip(request)

    # Reconcile the account before the first attempt so a freshly configured
    # deployment can sign in immediately (this is what previously failed).
    ensure_engineer_account(db)
    db.commit()

    if setup_required(db):
        log.error(
            "sign-in attempted but no engineer account exists "
            "(ENGINEER_EMAIL=%r, ENGINEER_PASSWORD set=%s)",
            settings.engineer_email,
            bool(settings.engineer_password),
        )
        raise AuthError(
            "This deployment has no engineer account yet. An administrator must set "
            "ENGINEER_EMAIL and ENGINEER_PASSWORD and restart the application.",
            details={"reason": "setup_required"},
        )

    try:
        check_login_allowed(db, email, ip_hint)
    except LoginLocked as locked:
        minutes = max(1, round(locked.retry_after_seconds / 60))
        record_login_attempt(db, email, ip_hint, successful=False)
        db.commit()
        raise RateLimited(
            f"Too many sign-in attempts. Please try again in about {minutes} minute"
            f"{'' if minutes == 1 else 's'}.",
            details={"retry_after_seconds": locked.retry_after_seconds},
        ) from locked

    try:
        user = authenticate(db, email, payload.password)
    except AuthError:
        record_login_attempt(db, email, ip_hint, successful=False)
        db.commit()
        log.info("failed sign-in for %r from %s", email, ip_hint or "unknown")
        raise AuthError(
            "Those sign-in details were not recognised. Check the email address and "
            "password, then try again.",
            details={"reason": "invalid_credentials"},
        ) from None

    record_login_attempt(db, email, ip_hint, successful=True)
    raw, session = create_session(
        db,
        user,
        user_agent=request.headers.get("user-agent", ""),
        ip_hint=ip_hint,
    )
    db.commit()
    log.info("sign-in ok for %s", user.email)
    set_session_cookies(response, raw, session)
    return SessionOut(
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        csrf_token=session.csrf_token,
    )


@router.post("/auth/register", response_model=SessionOut, status_code=201)
def register(
    payload: RegisterIn, request: Request, response: Response, db: DbSession
) -> SessionOut:
    email = normalise_email(payload.email)
    ip_hint = _ip(request)
    rate_key = f"signup:{email}"
    _raise_if_locked(db, rate_key, ip_hint)

    if not _valid_email(email):
        record_login_attempt(db, rate_key, ip_hint, successful=False)
        db.commit()
        raise ValidationError("Enter a valid email address.")
    if len(payload.password) < 10:
        record_login_attempt(db, rate_key, ip_hint, successful=False)
        db.commit()
        raise ValidationError("Password must be at least 10 characters.")
    if payload.password != payload.confirm_password:
        record_login_attempt(db, rate_key, ip_hint, successful=False)
        db.commit()
        raise ValidationError("The passwords do not match.")

    existing = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if existing is not None:
        record_login_attempt(db, rate_key, ip_hint, successful=False)
        db.commit()
        raise Conflict("This email already has an account, try signing in.")

    name = (payload.name or "").strip()[:120] or email.split("@", 1)[0]
    user = User(
        email=email,
        password_hash=hash_password(payload.password),
        display_name=name,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise Conflict("This email already has an account, try signing in.") from None

    record_login_attempt(db, rate_key, ip_hint, successful=True)
    raw, session = create_session(
        db,
        user,
        user_agent=request.headers.get("user-agent", ""),
        ip_hint=ip_hint,
    )
    db.commit()
    set_session_cookies(response, raw, session)
    return SessionOut(
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        csrf_token=session.csrf_token,
    )


@router.post("/auth/logout", response_model=MessageOut)
def logout(request: Request, response: Response, db: DbSession) -> MessageOut:
    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        revoke_session(db, token)
        db.commit()
    clear_session_cookies(response)
    return MessageOut(message="Signed out.")


@router.get("/session", response_model=SessionOut)
def session_info(request: Request, db: DbSession) -> SessionOut:
    user = current_user_optional(request, db)
    if user is None:
        raise AuthError("Your session has ended. Please sign in again.")
    auth_session = request.state.auth_session
    return SessionOut(
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        csrf_token=auth_session.csrf_token,
        session_expires_at=auth_session.expires_at,
        issued_at=utcnow(),
    )


@router.post("/auth/password-reset", response_model=MessageOut)
def password_reset(
    payload: PasswordResetRequestIn, request: Request, db: DbSession
) -> MessageOut:
    """Issue a hashed, short-lived reset token without exposing account existence."""
    email = normalise_email(payload.email)
    ip_hint = _ip(request)
    rate_key = f"reset:{email}"
    _raise_if_locked(db, rate_key, ip_hint)
    record_login_attempt(db, rate_key, ip_hint, successful=False)

    user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    if user is not None:
        now = utcnow()
        for row in db.execute(
            select(PasswordResetToken).where(
                PasswordResetToken.user_id == user.id,
                PasswordResetToken.consumed_at.is_(None),
            )
        ).scalars():
            row.consumed_at = now
        raw_token = new_token()
        db.add(
            PasswordResetToken(
                token_hash=hashlib.sha256(raw_token.encode()).hexdigest(),
                user_id=user.id,
                expires_at=now + timedelta(hours=1),
            )
        )
        if get_settings().app_env == "development":
            reset_url = (
                f"{get_settings().public_base_url.rstrip('/')}"
                f"/reset-password?token={raw_token}"
            )
            log.info("Development password reset link: %s", reset_url)
    db.commit()
    return MessageOut(
        message="If an account matches that email, reset instructions are ready. "
        "No email service is configured; development reset links are written to the server console."
    )


@router.post("/auth/password-reset/confirm", response_model=MessageOut)
def confirm_password_reset(
    payload: PasswordResetConfirmIn, request: Request, db: DbSession
) -> MessageOut:
    ip_hint = _ip(request)
    rate_key = f"reset-confirm:{ip_hint}"
    _raise_if_locked(db, rate_key, ip_hint)
    if len(payload.password) < 10:
        record_login_attempt(db, rate_key, ip_hint, successful=False)
        db.commit()
        raise ValidationError("Password must be at least 10 characters.")
    if payload.password != payload.confirm_password:
        record_login_attempt(db, rate_key, ip_hint, successful=False)
        db.commit()
        raise ValidationError("The passwords do not match.")

    token_hash = hashlib.sha256(payload.token.encode()).hexdigest()
    reset = db.get(PasswordResetToken, token_hash)
    now = utcnow()
    if (
        reset is None
        or reset.consumed_at is not None
        or reset.expires_at <= now
    ):
        record_login_attempt(db, rate_key, ip_hint, successful=False)
        db.commit()
        raise NotFound("That password reset link is invalid or has expired.")
    user = db.get(User, reset.user_id)
    if user is None or not user.is_active:
        record_login_attempt(db, rate_key, ip_hint, successful=False)
        db.commit()
        raise NotFound("That password reset link is invalid or has expired.")

    user.password_hash = hash_password(payload.password)
    reset.consumed_at = now
    for session in db.execute(
        select(AuthSession).where(AuthSession.user_id == user.id)
    ).scalars():
        if session.revoked_at is None:
            session.revoked_at = now
    record_login_attempt(db, rate_key, ip_hint, successful=True)
    db.commit()
    return MessageOut(message="Password reset. Please sign in with your new password.")
