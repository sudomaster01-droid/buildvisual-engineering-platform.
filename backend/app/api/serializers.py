"""Objects → API payloads. One place for what the browser is allowed to see."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import utcnow
from ..models import Job, JobEvent, Project, VideoVersion
from ..pipeline import state_machine as sm
from .schemas import JobOut, ProjectOut, StageOut, VideoOut


def progress_for(job: Job) -> list[StageOut]:
    """Derive the rail from the persisted stage — never from a timer.

    A stage is ``done`` only once the pipeline has moved past it, so the rail can
    never show a percentage the system cannot justify.
    """
    order = sm.PROGRESS_ORDER
    stage = job.stage
    index = order.index(stage) if stage in order else None
    completed = stage in (sm.READY_FOR_REVIEW, sm.COMPLETED)
    failed = stage == sm.FAILED

    stages: list[StageOut] = []
    for position, key in enumerate(order):
        if completed:
            state = "done"
        elif index is None:
            state = "pending"
        elif position < index:
            state = "done"
        elif position == index:
            state = "failed" if failed else "current"
        else:
            state = "pending"
        stages.append(StageOut(key=key, label=sm.stage_label(key), state=state))
    if failed and index is None:
        # The job failed somewhere outside the linear path: show it at the front.
        stages.insert(0, StageOut(key=stage, label=sm.stage_label(stage), state="failed"))
    if stage == sm.WAITING_FOR_CLARIFICATION:
        stages.insert(0, StageOut(key=stage, label=sm.stage_label(stage), state="current"))
    return stages


def current_message(db: Session, job: Job) -> tuple[str, str]:
    event = db.execute(
        select(JobEvent).where(JobEvent.job_id == job.id).order_by(JobEvent.created_at.desc()).limit(1)
    ).scalar_one_or_none()
    if event is not None:
        return event.message, event.level
    return sm.stage_label(job.stage), "info"


def events_for(db: Session, job: Job, limit: int = 40) -> list[dict[str, Any]]:
    rows = db.execute(
        select(JobEvent).where(JobEvent.job_id == job.id)
        .order_by(JobEvent.created_at.desc()).limit(limit)
    ).scalars().all()
    return [
        {
            "message": row.message,
            "level": row.level,
            "stage": row.stage,
            "stage_label": sm.stage_label(row.stage),
            "at": row.created_at.isoformat() + "Z",
        }
        for row in reversed(rows)
    ]


def video_out(db: Session, version: VideoVersion | None) -> VideoOut | None:
    if version is None:
        return None
    return VideoOut(
        id=version.id,
        version=version.version,
        title=version.title,
        state=version.state,
        generation_mode=version.generation_mode,
        provider=version.provider,
        duration_s=round(version.duration_s, 2),
        width=version.width,
        height=version.height,
        size_bytes=version.size_bytes,
        qa_passed=version.qa_passed,
        qa_summary=version.qa_summary or {},
        created_at=version.created_at,
        has_poster=bool(version.poster_key),
        parent_id=version.parent_id,
    )


def job_out(db: Session, job: Job, *, include_events: bool = True) -> JobOut:
    project = db.get(Project, job.project_id)
    message, level = current_message(db, job)
    video = video_out(db, db.get(VideoVersion, job.video_version_id)) if job.video_version_id else None
    diagnostics = job.diagnostics or {}
    generation = diagnostics.get("generation") or {}
    quality = diagnostics.get("quality") or {}
    rectification = diagnostics.get("rectification") or {}
    interpretation = diagnostics.get("interpretation") or {}
    spent = job.spent_cents or 0

    started = job.started_at or job.created_at
    elapsed = ((job.finished_at or utcnow()) - started).total_seconds()

    error = None
    if job.status == "failed":
        error = {
            "code": job.error_code or "failed",
            "message": job.error_message or job.user_message or "The job failed.",
            "retryable": bool(job.retryable),
        }
    return JobOut(
        id=job.id,
        project_id=job.project_id,
        project_name=project.name if project else "",
        kind=job.kind,
        stage=job.stage,
        status=job.status,
        input_mode=job.input_mode,
        progress=progress_for(job),
        current_message=message,
        message_level=level,
        events=events_for(db, job) if include_events else [],
        error=error,
        retryable=bool(job.retryable),
        clarification=job.clarification or None,
        video=video,
        created_at=job.created_at,
        updated_at=job.updated_at,
        elapsed_seconds=round(max(0.0, elapsed), 1),
        cost={
            "spent_cents": spent,
            "budget_cents": job.budget_cents or 0,
            "generation_mode": (
                "external_ai" if generation.get("external") else "model_render"
            ) if generation else None,
            "provider": generation.get("provider"),
        },
        rectification=(
            {
                "category": rectification.get("category"),
                "description": rectification.get("description"),
                "requires_new_model": bool(rectification.get("requires_new_model")),
                "affected_shots": rectification.get("affected_shots") or [],
                "text": job.rectification_text,
                "applied": bool(job.video_version_id),
            }
            if rectification else None
        ),
        interpretation=(
            {
                "source": (interpretation.get("report") or {}).get("source")
                or ("drawing" if interpretation.get("source") == "drawing" else "requirements"),
                "note": (interpretation.get("report") or {}).get("note"),
                "warnings": (interpretation.get("report") or {}).get("warnings") or [],
                "confidence": interpretation.get("confidence") or {},
            }
            if interpretation else None
        ),
        quality=(
            {
                "passed": quality.get("passed"),
                "severity": quality.get("severity"),
                "defects": [
                    {"code": d.get("code"), "severity": d.get("severity"),
                     "message": d.get("message", "")}
                    for d in quality.get("defects", [])
                ],
            }
            if quality else None
        ),
    )


def project_out(db: Session, project: Project) -> ProjectOut:
    count = db.execute(
        select(func.count()).select_from(VideoVersion).where(VideoVersion.project_id == project.id)
    ).scalar_one()
    latest_version = db.execute(
        select(func.max(VideoVersion.version)).where(VideoVersion.project_id == project.id)
    ).scalar()
    return ProjectOut(
        id=project.id,
        name=project.name,
        status=project.status,
        created_at=project.created_at,
        updated_at=project.updated_at,
        latest_video_version_id=project.latest_video_version_id,
        video_count=int(count or 0),
        latest_version=int(latest_version) if latest_version else None,
    )
