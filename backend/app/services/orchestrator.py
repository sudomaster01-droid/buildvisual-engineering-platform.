"""Orchestration: turning an engineer's request into a queued, recoverable job.

This module owns *admission*: validation of the request envelope, rate limits,
idempotency, project bookkeeping, source artifact persistence and enqueueing.
Execution belongs to ``app.pipeline.runner``.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import utcnow
from ..errors import (
    CapabilityUnavailable,
    ClarificationRequired,
    Conflict,
    NotFound,
    PermissionDenied,
    RateLimited,
    ValidationError,
)
from ..models import (
    HouseSpec,
    Job,
    Project,
    RectificationRequest,
    SourceArtifact,
    Transcript,
    VideoVersion,
)
from ..pipeline import state_machine as sm
from ..pipeline.runner import seed_steps
from .intake.validators import ValidatedFile, validate_audio, validate_drawing, validate_text
from .jobs import (
    active_jobs_for_user,
    charge,
    fail_job,
    get_idempotent_job,
    record_event,
    request_hash,
    set_deadline,
    store_idempotency,
)
from .rectify.apply import build_plan
from .rectify.classify import classify_change, classify_with_llm
from .interpret.llm import build_llm_client, llm_available
from ..spec_support import spec_from_row
from ..storage import get_storage, key_for_source

log = logging.getLogger("buildvisual.orchestrator")


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def create_project(db: Session, user_id: str, name: str | None = None) -> Project:
    project = Project(
        user_id=user_id, name=(name or "Untitled house").strip()[:160] or "Untitled house",
        status="draft",
    )
    db.add(project)
    db.flush()
    return project


def get_project(db: Session, project_id: str, user_id: str) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFound("That project could not be found.")
    if project.user_id != user_id:
        raise PermissionDenied("That project belongs to another account.")
    return project


# ---------------------------------------------------------------------------
# Rate limiting (per user, persisted so a restart cannot bypass it)
# ---------------------------------------------------------------------------


def enforce_limits(db: Session, user_id: str) -> None:
    settings = get_settings()
    active = active_jobs_for_user(db, user_id)
    if len(active) >= settings.max_active_jobs_per_user:
        raise RateLimited(
            f"You already have {len(active)} job(s) running. Wait for one to finish "
            "or cancel it before starting another."
        )
    recent = db.execute(
        select(Job).where(Job.user_id == user_id, Job.created_at >= utcnow() - timedelta(hours=1))
    ).scalars().all()
    if len(recent) >= settings.max_jobs_per_hour:
        raise RateLimited(
            "That is a lot of jobs in one hour. Try again in a little while."
        )


# ---------------------------------------------------------------------------
# Source artifacts
# ---------------------------------------------------------------------------


def store_artifact(
    db: Session, project: Project, validated: ValidatedFile, *,
    filename: str, text_content: str | None = None,
) -> SourceArtifact:
    storage = get_storage()
    artifact = SourceArtifact(
        project_id=project.id,
        kind=validated.kind,
        storage_key="pending",
        original_filename=(filename or "upload")[:255],
        mime=validated.mime,
        size_bytes=validated.size_bytes,
        checksum=validated.checksum,
        width=validated.width,
        height=validated.height,
        page_count=validated.page_count,
        duration_seconds=validated.duration_seconds,
        validation_status="valid",
        validation_report=validated.report,
        text_content=text_content,
    )
    db.add(artifact)
    db.flush()
    extension = validated.extension or ".bin"
    key = key_for_source(project.id, artifact.id, extension)
    stored = storage.put_bytes(key, validated.data)
    artifact.storage_key = stored.key
    artifact.checksum = stored.checksum
    db.flush()
    return artifact


def store_text_artifact(db: Session, project: Project, text: str) -> SourceArtifact:
    payload = text.encode()
    artifact = SourceArtifact(
        project_id=project.id,
        kind="text",
        storage_key="pending",
        original_filename="requirements.txt",
        mime="text/plain",
        size_bytes=len(payload),
        checksum=hashlib.sha256(payload).hexdigest(),
        validation_status="valid",
        validation_report={"validated": True, "characters": len(text)},
        text_content=text,
    )
    db.add(artifact)
    db.flush()
    key = key_for_source(project.id, artifact.id, ".txt")
    stored = get_storage().put_bytes(key, payload)
    artifact.storage_key = stored.key
    db.flush()
    return artifact


# ---------------------------------------------------------------------------
# Generation jobs
# ---------------------------------------------------------------------------


def create_generation_job(
    db: Session,
    *,
    user_id: str,
    project_id: str | None,
    project_name: str | None,
    text: str | None,
    drawing: tuple[str, bytes] | None,
    audio: tuple[str, bytes] | None,
    locale: str = "en",
    idempotency_key: str | None = None,
) -> Job:
    settings = get_settings()

    payload = {
        "project_id": project_id,
        "text": (text or "").strip(),
        "drawing": _fingerprint(drawing),
        "audio": _fingerprint(audio),
        "locale": locale,
    }
    digest = request_hash(payload)
    if idempotency_key:
        previous = get_idempotent_job(db, idempotency_key, user_id)
        if previous is not None:
            log.info("idempotent replay of key %s → job %s", idempotency_key, previous.id)
            return previous

    enforce_limits(db, user_id)

    project = (
        get_project(db, project_id, user_id) if project_id
        else create_project(db, user_id, project_name)
    )
    if project.status == "generating":
        raise Conflict("This project already has a job running.")

    artifact_ids: list[str] = []
    if drawing is not None:
        filename, data = drawing
        validated = validate_drawing(filename, data)
        artifact = store_artifact(db, project, validated, filename=filename)
        artifact_ids.append(artifact.id)
    if audio is not None:
        filename, data = audio
        validated_audio = validate_audio(filename, data)
        artifact = store_artifact(db, project, validated_audio, filename=filename)
        artifact_ids.append(artifact.id)
    cleaned_text = validate_text(text) if (text or "").strip() else None
    if cleaned_text:
        artifact = store_text_artifact(db, project, cleaned_text)
        artifact_ids.append(artifact.id)
    if not artifact_ids:
        raise ValidationError(
            "Add a drawing, a description, or a voice note before generating the video."
        )
    if drawing is None and cleaned_text is None:
        # Voice alone is fine only when transcription is configured.
        if not settings.transcription_available:
            raise ValidationError(
                "Voice transcription is not configured here, so a recording alone "
                "cannot be interpreted. Type the requirements as well."
            )

    job = Job(
        project_id=project.id,
        user_id=user_id,
        kind="generate",
        status=sm.user_status(sm.CREATED),
        stage=sm.CREATED,
        input_mode=(
            "drawing" if drawing and not (cleaned_text or audio)
            else "voice" if audio else "text"
        ),
        source_artifact_ids=artifact_ids,
        text_requirements=cleaned_text,
        locale=locale or "en",
        budget_cents=settings.max_job_spend_cents,
        diagnostics={"request_hash": digest},
    )
    db.add(job)
    db.flush()
    seed_steps(db, job)
    set_deadline(job)
    project.status = "generating"
    record_event(
        db, job, stage=sm.CREATED, message="Queued.",
        data={"input_mode": job.input_mode, "artifacts": len(artifact_ids)},
    )
    store_idempotency(db, idempotency_key or "", user_id, "create_job", job, digest)
    db.flush()
    log.info("created job %s for project %s", job.id, project.id)
    return job


def _fingerprint(payload: tuple[str, bytes] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    filename, data = payload
    return {
        "filename": filename,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Rectification jobs
# ---------------------------------------------------------------------------


def create_rectification_job(
    db: Session, *,
    user_id: str,
    video_version_id: str,
    text: str,
    idempotency_key: str | None = None,
) -> tuple[Job, dict[str, Any]]:
    """Classify a correction and queue the job that produces the next version.

    Returns the job plus a summary of how the request was understood so the UI can
    tell the engineer what will change (and, for structural edits, that a new
    model version is built).
    """
    video = db.get(VideoVersion, video_version_id)
    if video is None:
        raise NotFound("That video version could not be found.")
    project = get_project(db, video.project_id, user_id)
    if video.state != "ready" or not video.qa_passed:
        raise Conflict("Only a video that passed review can be corrected.")

    cleaned = (text or "").strip()
    if len(cleaned) < 3:
        raise ValidationError("Describe the change you want, e.g. \"make the lighting warmer\".")

    digest = request_hash({"video_version_id": video_version_id, "text": cleaned})
    if idempotency_key:
        previous = get_idempotent_job(db, idempotency_key, user_id)
        if previous is not None:
            intent_row = db.execute(
                select(RectificationRequest).where(RectificationRequest.job_id == previous.id)
            ).scalar_one_or_none()
            return previous, (intent_row.applied if intent_row else {})

    enforce_limits(db, user_id)

    spec_row = db.get(HouseSpec, _spec_id_for_video(db, video))
    if spec_row is None:
        raise Conflict("The house model behind this video is no longer available.")
    spec = spec_from_row(spec_row)

    classifier = "rules"
    intent = classify_change(cleaned, spec)
    if llm_available() and (intent.requires_confirmation or intent.category == "clarification"):
        try:
            client = build_llm_client()
        except CapabilityUnavailable:  # pragma: no cover - guarded by llm_available()
            client = None
        if client is not None:
            refined = classify_with_llm(cleaned, spec, client)
            if refined is not None:
                intent = refined
                classifier = "llm"

    summary: dict[str, Any] = {
        "category": intent.category,
        "confidence": round(intent.confidence, 3),
        "classifier": classifier,
        "targets": intent.targets,
        "affected_shots": intent.affected_shots,
    }

    try:
        plan = build_plan(intent, spec)
    except ClarificationRequired as exc:
        details = exc.details or {}
        summary["question"] = details.get("question") or exc.user_message
        summary["options"] = details.get("options") or []
        summary["reason"] = details.get("reason")
        summary["supported"] = details.get("supported") or []
        raise ClarificationRequired(
            exc.user_message, details=summary,
        ) from exc

    # Tell the client up front whether the locked model is kept or rebuilt, so the
    # UI can say so before any work starts.
    summary["requires_new_model"] = plan.requires_new_model
    summary["description"] = plan.description
    summary["notes"] = plan.notes or []

    job = Job(
        project_id=project.id,
        user_id=user_id,
        kind="rectify",
        status=sm.user_status(sm.CREATED),
        stage=sm.CREATED,
        input_mode="text",
        text_requirements=None,
        parent_video_version_id=video.id,
        rectification_text=cleaned,
        rectification_category=plan.category,
        house_spec_id=spec_row.id,
        model_version_id=video.model_version_id,
        budget_cents=get_settings().max_job_spend_cents,
        diagnostics={
            "request_hash": digest,
            "rectification": {
                "category": plan.category,
                "description": plan.description,
                "requires_new_model": plan.requires_new_model,
                "structural_spec": (
                    plan.structural_spec.model_dump(mode="json")
                    if plan.structural_spec else None
                ),
                "design_overrides": plan.design_overrides,
                "staging": plan.design_overrides.get("staging", {}),
                "lighting_overrides": plan.design_overrides.get("lighting_overrides", {}),
                "shot_adjustments": plan.shot_adjustments,
                "affected_shots": plan.affected_shots,
                "parent_house_spec_id": spec_row.id,
                "parent_model_version_id": video.model_version_id,
                "intent": intent.as_dict() if hasattr(intent, "as_dict") else None,
                "notes": plan.notes,
            },
        },
    )
    db.add(job)
    db.flush()
    seed_steps(db, job)
    set_deadline(job)
    project.status = "generating"

    request_row = RectificationRequest(
        project_id=project.id,
        job_id=job.id,
        parent_video_version_id=video.id,
        text=cleaned,
        category=plan.category,
        classifier=classifier,
        confidence=intent.confidence,
        targets=intent.targets,
        affected_shots=plan.affected_shots,
        applied={**summary, "requires_new_model": plan.requires_new_model,
                 "description": plan.description},
    )
    db.add(request_row)
    record_event(
        db, job, stage=sm.CREATED,
        message=f"Correction queued ({plan.category}).",
        data={**summary, "requires_new_model": plan.requires_new_model},
    )
    store_idempotency(db, idempotency_key or "", user_id, "create_job", job, digest)
    db.flush()
    return job, summary


def _spec_id_for_video(db: Session, video: VideoVersion) -> str:
    from ..models import ModelVersion

    model_row = db.get(ModelVersion, video.model_version_id)
    if model_row is not None:
        return model_row.house_spec_id
    return ""


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


def retry_job(db: Session, job: Job, user_id: str) -> Job:
    """Re-queue a failed job from its first unfinished step."""
    if job.user_id != user_id:
        raise PermissionDenied()
    if job.stage not in (sm.FAILED, sm.CANCELED):
        raise Conflict("Only a failed or canceled job can be retried.")
    if not job.retryable:
        raise Conflict(
            "This job cannot be retried as-is. Check the message and start a new job."
        )
    from ..models import PipelineStep

    steps = db.execute(
        select(PipelineStep).where(PipelineStep.job_id == job.id)
    ).scalars().all()
    for step in steps:
        if step.state != "succeeded":
            step.state = "pending"
            step.error_code = None
            step.error_message = None
            step.worker_id = None
            step.lease_expires_at = None
    job.error_code = None
    job.error_message = None
    job.user_message = None
    job.finished_at = None
    job.retryable = False
    job.stage = sm.CREATED if job.stage == sm.CANCELED else job.stage
    job.status = sm.user_status(sm.CREATED)
    job.stage = sm.CREATED
    set_deadline(job)
    project = db.get(Project, job.project_id)
    if project is not None:
        project.status = "generating"
    record_event(db, job, stage=sm.CREATED, message="Queued again.",
                 data={"manual_retry": True})
    db.flush()
    return job


def cancel(db: Session, job: Job, user_id: str) -> Job:
    from ..pipeline.runner import cancel_job

    if job.user_id != user_id:
        raise PermissionDenied()
    return cancel_job(db, job)


def retry_status(db: Session, job: Job) -> dict[str, Any]:
    from ..models import PipelineStep

    steps = db.execute(
        select(PipelineStep).where(PipelineStep.job_id == job.id).order_by(PipelineStep.sequence)
    ).scalars().all()
    return {
        step.name: {"state": step.state, "attempts": step.attempts, "error": step.error_code}
        for step in steps
    }
