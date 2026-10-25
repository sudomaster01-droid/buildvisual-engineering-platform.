"""Shared API dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Job, User
from ..security import require_csrf, require_user

DbSession = Annotated[Session, Depends(get_db)]
CurrentUser = Annotated[User, Depends(require_user)]


def csrf_guard(request: Request) -> None:
    """Enforce the double-submit CSRF token on mutating requests."""
    require_csrf(request)


CsrfProtected = Annotated[None, Depends(csrf_guard)]


def idempotency_key(
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> str | None:
    if idempotency_key is None:
        return None
    cleaned = idempotency_key.strip()
    if not cleaned:
        return None
    if len(cleaned) > 120:
        from ..errors import ValidationError

        raise ValidationError("Idempotency-Key must be 120 characters or fewer.")
    return cleaned


IdempotencyKey = Annotated[str | None, Depends(idempotency_key)]


def load_job(db: Session, job_id: str, user: User) -> Job:
    from ..errors import NotFound, PermissionDenied

    job = db.get(Job, job_id)
    if job is None:
        raise NotFound("That job could not be found.")
    if job.user_id != user.id:
        raise PermissionDenied("That job belongs to another account.")
    return job
