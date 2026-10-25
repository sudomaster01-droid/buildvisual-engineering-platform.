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

from fastapi import APIRouter, Request, Response

from ..config import get_settings
from ..db import utcnow
from ..errors import AuthError, RateLimited
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
    normalise_email,
    record_login_attempt,
    revoke_session,
    set_session_cookies,
    setup_required,
)
from .deps import DbSession
from .schemas import LoginIn, LoginStatusOut, MessageOut, SessionOut

router = APIRouter(tags=["auth"])
log = logging.getLogger("buildvisual.api.auth")


def _ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "")[:64]


@router.get("/auth/status", response_model=LoginStatusOut)
def login_status(request: Request, db: DbSession) -> LoginStatusOut:
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
def password_reset(_request: Request) -> MessageOut:
    """Recovery for a single-engineer deployment, without email infrastructure.

    The response is deliberately identical for every caller and reveals nothing
    about whether an account exists. Because the password is configured on the
    server (and reconciled at startup), the honest recovery path is an operator
    action, not a self-service email. The screen shows the exact command.
    """
    return MessageOut(
        message=(
            "Password recovery is handled by whoever operates this deployment. "
            "Run: python -m app.cli set-password  (it will prompt for a new password), "
            "then sign in again."
        )
    )
