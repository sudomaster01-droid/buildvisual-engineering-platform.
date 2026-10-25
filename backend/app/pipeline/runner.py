"""Worker queue + step runner.

Blueprint p.70/p.71/p.73:
* the queue is a database table, not an in-memory list, so a restart loses nothing;
* a step is claimed with a lease + heartbeat, so two workers cannot run the same
  step and a crashed worker's step is reclaimed after the lease expires;
* every step is idempotent: re-running it writes a *new* version row or reuses the
  one already recorded on the job, and never duplicates billable work silently;
* failures are classified into retryable (transient provider/network) vs
  permanent (validation, capability, budget) with bounded attempts.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import session_scope, utcnow
from ..errors import (
    BudgetExceeded,
    BuildVisualError,
    CapabilityUnavailable,
    Conflict,
    QualityGateFailed,
    ValidationError,
)
from ..models import Job, PipelineStep, Project
from ..pipeline import state_machine as sm
from ..services.jobs import fail_job, record_event, set_deadline, transition
from ..storage import get_storage
from .context import StepContext
from .steps import finish, generate, intake, model as model_steps, reference

log = logging.getLogger("buildvisual.runner")


@dataclass
class StepSpec:
    name: str
    stage: str
    run: Callable[[StepContext], dict[str, Any]]
    max_attempts: int = 3
    #: permanent errors abort the job immediately instead of retrying
    retry_on: tuple[type[BaseException], ...] = ()


#: The pipeline, in order. Generation and rectification share the tail; the head
#: differs only in what ``interpret_house``/``build_locked_model`` decide.
STEPS: list[StepSpec] = [
    StepSpec("validate_input", sm.VALIDATING, intake.validate_input, max_attempts=1),
    StepSpec("interpret_house", sm.INTERPRETING, intake.interpret_house, max_attempts=2),
    StepSpec("build_locked_model", sm.MODEL_BUILD, model_steps.build_locked_model, max_attempts=2),
    StepSpec("complete_design", sm.DESIGN_COMPLETION, model_steps.complete_design, max_attempts=2),
    StepSpec("plan_shots", sm.SHOT_PLANNING, reference.plan_shots, max_attempts=2),
    StepSpec("render_references", sm.REFERENCE_RENDER, reference.render_references, max_attempts=2),
    StepSpec("compose_prompts", sm.PROMPT_COMPOSITION, reference.compose_prompts_step, max_attempts=2),
    StepSpec("generate_clips", sm.PROVIDER_SUBMITTED, generate.generate_clips,
             max_attempts=2, retry_on=(Exception,)),
    StepSpec("assemble_film", sm.ASSEMBLING, finish.assemble_film, max_attempts=2),
    StepSpec("quality_gate", sm.QUALITY_CHECK, finish.quality_gate, max_attempts=2),
    StepSpec("publish_version", sm.READY_FOR_REVIEW, finish.publish_version, max_attempts=2),
]

STEP_BY_NAME = {step.name: step for step in STEPS}


def step_names_for(job: Job) -> list[str]:
    """All jobs run the same sequence; behaviour differs inside the steps."""
    return [step.name for step in STEPS]


def seed_steps(db: Session, job: Job) -> list[PipelineStep]:
    existing = {row.name: row for row in db.execute(
        select(PipelineStep).where(PipelineStep.job_id == job.id)
    ).scalars().all()}
    rows: list[PipelineStep] = []
    for index, spec in enumerate(STEPS):
        row = existing.get(spec.name)
        if row is None:
            row = PipelineStep(
                job_id=job.id, name=spec.name, sequence=index,
                state="pending", max_attempts=spec.max_attempts,
            )
            db.add(row)
        rows.append(row)
    db.flush()
    return rows


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------


class Worker:
    def __init__(self, *, worker_id: str | None = None, poll_seconds: float | None = None):
        settings = get_settings()
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self.poll_seconds = poll_seconds if poll_seconds is not None else settings.worker_poll_interval_seconds
        self.lease_seconds = settings.worker_lease_seconds
        self._stop = threading.Event()
        self.storage = get_storage()

    # -- control ----------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:  # pragma: no cover - loop under supervision
        log.info("worker %s starting", self.worker_id)
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except Exception:  # pragma: no cover - defensive
                log.exception("worker loop error")
                worked = False
            if not worked:
                self._stop.wait(self.poll_seconds)
        log.info("worker %s stopped", self.worker_id)

    # -- one unit of work -------------------------------------------------
    def run_once(self) -> bool:
        with session_scope() as db:
            step = claim_next_step(db, self.worker_id, lease_seconds=self.lease_seconds)
            if step is None:
                sweep_stalled(db)
                return False
            job = db.get(Job, step.job_id)
            if job is None:  # pragma: no cover - cascade deletes the step
                return False
            self._execute(db, job, step)
            return True

    def _execute(self, db: Session, job: Job, step: PipelineStep) -> None:
        spec = STEP_BY_NAME.get(step.name)
        if spec is None:  # pragma: no cover - guards against a renamed step
            step.state = "failed"
            step.error_message = "unknown step"
            return
        log_extra = {"job_id": job.id, "step": step.name, "worker": self.worker_id}
        logger = logging.LoggerAdapter(log, log_extra)
        ctx = StepContext(
            job=job, db=db, settings=get_settings(), storage=self.storage,
            worker_id=self.worker_id,
            heartbeat=lambda: self._heartbeat(db, step),
            log=logger, step_name=step.name, cache={},
        )

        step.state = "running"
        step.attempts = (step.attempts or 0) + 1
        step.started_at = step.started_at or utcnow()
        step.worker_id = self.worker_id
        step.lease_expires_at = utcnow() + timedelta(seconds=self.lease_seconds)
        job.worker_id = self.worker_id
        job.current_step = step.name
        job.attempts = (job.attempts or 0) + 1 if step.attempts == 1 else job.attempts
        if job.started_at is None:
            job.started_at = utcnow()
            set_deadline(job)
        job.updated_at = utcnow()
        _advance_stage(db, job, spec.stage)
        db.flush()
        db.commit()  # publish the stage change before the step starts working

        if job.deadline_at and utcnow() > job.deadline_at:
            fail_job(
                db, job, error_code="job_timeout",
                user_message=(
                    "This job exceeded its time limit and was stopped. Its cost so far is "
                    "recorded; you can try again."
                ),
                retryable=True, details={"step": step.name},
            )
            step.state = "failed"
            step.error_message = "job deadline exceeded"
            return

        try:
            output = spec.run(ctx)
            step.state = "succeeded"
            step.output = output or {}
            step.finished_at = utcnow()
            step.lease_expires_at = None
            step.error_code = None
            db.flush()
            logger.info("step %s succeeded", step.name)
        except BuildVisualError as exc:
            self._handle_failure(db, job, step, spec, exc, retryable=exc.retryable)
        except Exception as exc:  # noqa: BLE001 - classified below
            self._handle_failure(db, job, step, spec, exc, retryable=isinstance(exc, OSError))
        finally:
            job.current_step = None
            db.flush()

    def _handle_failure(
        self, db: Session, job: Job, step: PipelineStep, spec: StepSpec,
        exc: BaseException, *, retryable: bool,
    ) -> None:
        code = getattr(exc, "code", None) or exc.__class__.__name__.lower()
        message = getattr(exc, "user_message", None) or str(exc) or "The step failed."
        attempts_left = (step.max_attempts or 1) - (step.attempts or 1)
        fatal_types = (ValidationError, CapabilityUnavailable, BudgetExceeded, Conflict)
        permanent = isinstance(exc, fatal_types) or not retryable
        log.warning(
            "step %s failed (%s): %s (attempts_left=%s permanent=%s)",
            step.name, code, message, attempts_left, permanent,
        )
        step.error_code = str(code)[:64]
        step.error_message = message[:1000]
        step.lease_expires_at = None
        if not permanent and attempts_left > 0:
            step.state = "pending"
            step.worker_id = None
            job.status = "queued"
            record_event(
                db, job, stage=job.stage, level="warning",
                message="A step did not finish and will be retried automatically.",
                data={"step": step.name, "attempts_left": attempts_left, "code": str(code)},
            )
            return
        step.state = "failed"
        step.finished_at = utcnow()
        if not isinstance(exc, BuildVisualError):
            code = "internal_error"
            message = (
                "Something went wrong inside the pipeline. The technical detail has been "
                "logged; you can try again."
            )
        # An unexpected internal error is retryable at job level: it is almost
        # always transient (a restart, a full disk), and the step budget still
        # bounds how many times it can happen.
        retry_user = (
            True if not isinstance(exc, BuildVisualError)
            else bool(getattr(exc, "retryable", False))
        ) and not isinstance(exc, fatal_types)
        fail_job(
            db, job, error_code=str(code), user_message=message,
            retryable=retry_user,
            details={"step": step.name,
                     "details": getattr(exc, "details", {}) or {}},
        )

    def _heartbeat(self, db: Session, step: PipelineStep) -> None:
        step.heartbeat_at = utcnow()
        step.lease_expires_at = utcnow() + timedelta(seconds=self.lease_seconds)
        step.state = "running"
        db.flush()


def _advance_stage(db: Session, job: Job, stage: str) -> None:
    if job.stage == stage:
        return
    if not sm.can_transition(job.stage, stage):
        # e.g. provider_processing → retrieving already handled inside the step
        log.debug("stage %s -> %s not on the main path", job.stage, stage)
        return
    transition(db, job, stage)


# ---------------------------------------------------------------------------
# Claiming
# ---------------------------------------------------------------------------


def claim_next_step(db: Session, worker_id: str, *, lease_seconds: int = 300) -> PipelineStep | None:
    """Claim the earliest runnable step with a compare-and-set update."""
    now = utcnow()
    stale = now - timedelta(seconds=lease_seconds)

    statement = (
        select(PipelineStep, Job)
        .join(Job, Job.id == PipelineStep.job_id)
        .where(
            Job.stage.notin_(list(sm.TERMINAL)),
            PipelineStep.state.in_(("pending", "queued", "running")),
            or_(
                PipelineStep.state != "running",
                PipelineStep.lease_expires_at.is_(None),
                PipelineStep.lease_expires_at < now,
            ),
        )
        .order_by(PipelineStep.sequence)
        .limit(25)
    )
    candidates = db.execute(statement).all()
    if not candidates:
        return None

    for step, job in candidates:
        # Only the earliest unfinished step of a job is runnable.
        earlier = db.execute(
            select(func.count()).select_from(PipelineStep).where(
                PipelineStep.job_id == step.job_id,
                PipelineStep.sequence < step.sequence,
                PipelineStep.state.notin_(("succeeded", "skipped")),
            )
        ).scalar_one()
        if earlier:
            continue
        updated = db.execute(
            PipelineStep.__table__.update()
            .where(
                PipelineStep.id == step.id,
                or_(
                    PipelineStep.lease_expires_at.is_(None),
                    PipelineStep.lease_expires_at < now,
                    and_(PipelineStep.state == "pending", PipelineStep.worker_id.is_(None)),
                ),
            )
            .values(
                state="running", worker_id=worker_id,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                heartbeat_at=now,
            )
        )
        if updated.rowcount != 1:
            continue  # lost the race to another worker
        db.flush()
        log.debug("worker %s claimed step %s of job %s", worker_id, step.name, job.id)
        return db.get(PipelineStep, step.id)
    return None


def sweep_stalled(db: Session) -> dict[str, int]:
    """Recovery scanner: re-queue expired leases and fail dead jobs.

    Blueprint p.73 — runs before each idle poll and periodically in the API
    process, so a restarted deployment resumes work without operator action.
    """
    settings = get_settings()
    now = utcnow()
    requeued = failed = 0

    stalled = db.execute(
        select(PipelineStep).where(
            PipelineStep.state == "running",
            PipelineStep.lease_expires_at.isnot(None),
            PipelineStep.lease_expires_at < now - timedelta(seconds=5),
        )
    ).scalars().all()
    for step in stalled:
        job = db.get(Job, step.job_id)
        if job is None:
            continue
        if step.attempts >= (step.max_attempts or 1):
            step.state = "failed"
            step.error_message = "worker lease expired repeatedly"
            fail_job(
                db, job, error_code="worker_lost",
                user_message=(
                    "Processing was interrupted and could not be resumed automatically. "
                    "You can run it again."
                ),
                retryable=True, details={"step": step.name},
            )
            failed += 1
        else:
            step.state = "pending"
            step.worker_id = None
            step.lease_expires_at = None
            record_event(
                db, job, stage=job.stage, level="warning",
                message="Processing was interrupted and has been resumed automatically.",
                data={"step": step.name},
            )
            requeued += 1

    # Deadlines.
    overdue = db.execute(
        select(Job).where(
            Job.stage.notin_(list(sm.TERMINAL)),
            Job.deadline_at.isnot(None),
            Job.deadline_at < now,
        )
    ).scalars().all()
    for job in overdue:
        fail_job(
            db, job, error_code="job_timeout",
            user_message=(
                "This job exceeded its time limit and was stopped. Recorded cost is "
                "preserved; you can try again."
            ),
            retryable=True, details={"swept": True},
        )
        failed += 1

    if requeued or failed:
        log.info("recovery sweep: requeued=%s failed=%s", requeued, failed)
    return {"requeued": requeued, "failed": failed}


# ---------------------------------------------------------------------------
# Cancellation + ad-hoc execution (tests, CLI)
# ---------------------------------------------------------------------------


def cancel_job(db: Session, job: Job, *, reason: str = "Canceled by the engineer.") -> Job:
    if job.stage in sm.TERMINAL:
        raise Conflict(f"Job {job.id} is already {job.stage}.")
    for step in db.execute(
        select(PipelineStep).where(
            PipelineStep.job_id == job.id,
            PipelineStep.state.in_(("pending", "queued", "running")),
        )
    ).scalars().all():
        step.state = "skipped"
        step.finished_at = utcnow()
        step.lease_expires_at = None
    job.user_message = reason
    job.error_code = "canceled"
    transition(db, job, sm.CANCELED, message=reason, level="warning", force=True)
    project = db.get(Project, job.project_id)
    if project is not None and project.status == "generating":
        project.status = "draft"
    db.flush()
    return job


def run_job_to_completion(job_id: str, *, worker_id: str = "inline", max_steps: int = 40) -> Job:
    """Synchronous driver used by tests and ``python -m app.cli run``."""
    worker = Worker(worker_id=worker_id)
    for _ in range(max_steps):
        with session_scope() as db:
            job = db.get(Job, job_id)
            if job is None:
                raise LookupError(job_id)
            if job.stage in sm.TERMINAL:
                return job
        step = None
        with session_scope() as db:
            job = db.get(Job, job_id)
            pending = db.execute(
                select(PipelineStep).where(
                    PipelineStep.job_id == job_id,
                    PipelineStep.state.in_(("pending", "running")),
                ).order_by(PipelineStep.sequence)
            ).scalars().first()
            if pending is None:
                return job
            step = pending
        # Claim and execute exactly this step so unrelated jobs are untouched.
        with session_scope() as db:
            job = db.get(Job, job_id)
            row = db.get(PipelineStep, step.id)
            if row is None or row.state not in ("pending", "running"):
                continue
            row.worker_id = worker.worker_id
            row.lease_expires_at = utcnow() + timedelta(seconds=worker.lease_seconds)
            db.flush()
            worker._execute(db, job, row)
    raise RuntimeError("job did not finish within the step budget")


def install_signal_handlers(worker: Worker) -> None:  # pragma: no cover - process plumbing
    def _handler(signum, _frame):  # noqa: ANN001
        log.info("signal %s received, stopping worker", signum)
        worker.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def main() -> None:  # pragma: no cover - entry point
    from ..logging_setup import configure_logging

    configure_logging()
    worker = Worker()
    install_signal_handlers(worker)
    worker.run_forever()


if __name__ == "__main__":  # pragma: no cover
    main()
