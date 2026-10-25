"""Job persistence helpers shared by the API, the runner and the recovery sweep.

Keeping these in one place means the state machine is enforced in a single
function (``transition``) rather than being re-implemented in three call sites.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import utcnow
from ..errors import Conflict
from ..models import IdempotencyKey, Job, JobEvent, Project, UsageLedger
from ..pipeline import state_machine as sm

log = logging.getLogger("buildvisual.jobs")


def record_event(
    db: Session, job: Job, *, stage: str, message: str, level: str = "info",
    data: dict[str, Any] | None = None,
) -> JobEvent:
    event = JobEvent(
        job_id=job.id, stage=stage, status=job.status, message=message[:500],
        level=level, data=data or {},
    )
    db.add(event)
    db.flush()
    return event


def transition(
    db: Session,
    job: Job,
    target: str,
    *,
    message: str | None = None,
    level: str = "info",
    data: dict[str, Any] | None = None,
    force: bool = False,
) -> Job:
    """Move a job to ``target``, validating the transition and logging it."""
    current = job.stage or sm.CREATED
    if current == target:
        return job
    if not force and not sm.can_transition(current, target):
        # Strict by default: an illegal jump is a bug, not something to log past.
        # Recovery paths (fail/cancel) pass force=True deliberately.
        raise Conflict(
            f"Job {job.id} cannot move from {current} to {target}.",
            details={"job_id": job.id, "from": current, "to": target},
        )
    job.stage = target
    job.status = sm.user_status(target)
    job.updated_at = utcnow()
    if target == sm.READY_FOR_REVIEW and job.finished_at is None:
        job.finished_at = utcnow()
    if target in sm.TERMINAL and job.finished_at is None:
        job.finished_at = utcnow()
    record_event(
        db, job, stage=target,
        message=message or sm.stage_label(target),
        level=level, data=data or {},
    )
    return job


def fail_job(
    db: Session, job: Job, *,
    error_code: str,
    user_message: str,
    retryable: bool,
    details: dict[str, Any] | None = None,
) -> Job:
    job.error_code = error_code
    job.error_message = user_message[:1000]
    job.user_message = user_message
    job.retryable = retryable
    transition(
        db, job, sm.FAILED, message=user_message, level="error",
        data={"error_code": error_code, "retryable": retryable, **(details or {})},
        force=True,
    )
    project = db.get(Project, job.project_id)
    if project is not None:
        project.status = "failed"
    return job


def touch_job(db: Session, job: Job) -> None:
    job.updated_at = utcnow()


def request_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def get_idempotent_job(db: Session, key: str, user_id: str, scope: str = "create_job") -> Job | None:
    if not key:
        return None
    row = db.get(IdempotencyKey, key)
    if row is None or row.user_id != user_id or row.scope != scope:
        return None
    if not row.job_id:
        return None
    return db.get(Job, row.job_id)


def store_idempotency(
    db: Session, key: str, user_id: str, scope: str, job: Job, payload_hash: str
) -> None:
    if not key:
        return
    existing = db.get(IdempotencyKey, key)
    if existing is not None:
        return
    db.add(IdempotencyKey(
        key=key, user_id=user_id, scope=scope, request_hash=payload_hash, job_id=job.id,
    ))
    db.flush()


def job_spend_cents(db: Session, job_id: str) -> int:
    rows = db.execute(
        select(UsageLedger.cost_cents).where(UsageLedger.job_id == job_id)
    ).scalars().all()
    return int(sum(rows or []))


def spend_since(db: Session, since) -> int:
    rows = db.execute(
        select(UsageLedger.cost_cents).where(UsageLedger.created_at >= since)
    ).scalars().all()
    return int(sum(rows or []))


def charge(
    db: Session,
    job: Job,
    *,
    provider: str,
    operation: str,
    cost_cents: int,
    units: float = 0.0,
    unit_name: str = "request",
    basis: str = "configured_rate",
    estimated: bool = True,
    detail: dict[str, Any] | None = None,
) -> UsageLedger:
    row = UsageLedger(
        job_id=job.id, project_id=job.project_id, provider=provider, operation=operation,
        units=units, unit_name=unit_name, cost_cents=int(cost_cents),
        cost_basis=basis, estimated=estimated, detail=detail or {},
    )
    db.add(row)
    db.flush()
    job.spent_cents = job_spend_cents(db, job.id)
    return row


def active_jobs_for_user(db: Session, user_id: str) -> list[Job]:
    rows = db.execute(
        select(Job).where(Job.user_id == user_id).order_by(Job.created_at.desc())
    ).scalars().all()
    return [job for job in rows if job.stage not in sm.TERMINAL]


def latest_active_job(db: Session, user_id: str) -> Job | None:
    jobs = active_jobs_for_user(db, user_id)
    return jobs[0] if jobs else None


def job_deadline_seconds() -> int:
    from ..config import get_settings

    return get_settings().job_max_runtime_seconds


def set_deadline(job: Job) -> None:
    job.deadline_at = utcnow() + timedelta(seconds=job_deadline_seconds())
