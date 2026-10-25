"""SQLAlchemy models — the persistent record of every pipeline decision.

Design rules taken from the blueprint (p.77):
* non-guessable UUID primary keys,
* foreign keys and unique idempotency constraints,
* schema/model version stamped on every dependent record,
* media lives in object storage; only metadata lives here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base, utcnow


def new_id() -> str:
    return uuid.uuid4().hex


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), default="Engineer")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class LoginAttempt(Base):
    """Persisted login throttling.

    Stored in the database rather than in memory so a restart or a second
    process cannot reset the counter, and so a lockout survives deployment.
    """

    __tablename__ = "login_attempts"
    __table_args__ = (Index("ix_login_attempts_lookup", "email", "ip_hint", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    ip_hint: Mapped[str] = mapped_column(String(64), default="")
    successful: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    #: sha256 of the session token; the raw token only ever exists in the cookie.
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    csrf_token: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ip_hint: Mapped[str | None] = mapped_column(String(64), nullable=True)


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# Project + inputs
# ---------------------------------------------------------------------------


class Project(Base, TimestampMixin):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(160), default="Untitled house")
    #: 'draft' | 'generating' | 'ready' | 'failed'
    status: Mapped[str] = mapped_column(String(24), default="draft", index=True)
    latest_video_version_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    jobs: Mapped[list["Job"]] = relationship(back_populates="project")


class SourceArtifact(Base):
    """An immutable, checksummed copy of exactly what the engineer supplied."""

    __tablename__ = "source_artifacts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    #: 'drawing' | 'audio' | 'text'
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mime: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(nullable=True)
    #: 'valid' | 'rejected'
    validation_status: Mapped[str] = mapped_column(String(16), default="valid")
    validation_report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    text_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(32), index=True)
    source_artifact_id: Mapped[str] = mapped_column(String(32), index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    provider: Mapped[str] = mapped_column(String(32), default="unknown")
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------
# Interpretation → model → design
# ---------------------------------------------------------------------------


class HouseSpec(Base):
    __tablename__ = "house_specs"
    __table_args__ = (UniqueConstraint("project_id", "version", name="uq_house_spec_version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    parent_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    schema_version: Mapped[str] = mapped_column(String(16), default="1.0.0")
    spec: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    confidence: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    unresolved: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ModelVersion(Base):
    """The locked structural reference. Immutable once written."""

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("project_id", "version", name="uq_model_version"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    house_spec_id: Mapped[str] = mapped_column(String(32), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    builder_version: Mapped[str] = mapped_column(String(16), default="1.0.0")
    geometry_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_key: Mapped[str] = mapped_column(String(512), nullable=False)  # GLB
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    warnings: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class SceneManifest(Base):
    """Styling layer. References geometry object ids; never mutates geometry."""

    __tablename__ = "scene_manifests"
    __table_args__ = (UniqueConstraint("project_id", "version", name="uq_scene_version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(32), index=True)
    model_version_id: Mapped[str] = mapped_column(String(32), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    seed: Mapped[int] = mapped_column(Integer, default=0)
    palette_id: Mapped[str] = mapped_column(String(64), default="warm_natural")
    manifest: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ShotPlan(Base):
    __tablename__ = "shot_plans"
    __table_args__ = (UniqueConstraint("project_id", "version", name="uq_shot_plan_version"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(32), index=True)
    model_version_id: Mapped[str] = mapped_column(String(32), index=True)
    scene_manifest_id: Mapped[str] = mapped_column(String(32), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    planner_version: Mapped[str] = mapped_column(String(16), default="1.0.0")
    plan: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ReferenceArtifact(Base):
    __tablename__ = "reference_artifacts"
    __table_args__ = (Index("ix_refs_shot_kind", "shot_plan_id", "shot_id", "kind"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    shot_plan_id: Mapped[str] = mapped_column(String(32), index=True)
    shot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    #: 'color' | 'depth' | 'edge' | 'segmentation'
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    frame_role: Mapped[str] = mapped_column(String(16), default="first")
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    checksum: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(String(32), index=True)
    shot_plan_id: Mapped[str] = mapped_column(String(32), index=True)
    template_version: Mapped[str] = mapped_column(String(16), default="1.0.0")
    provider: Mapped[str] = mapped_column(String(24), default="generic")
    prompts: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    redacted_prompt_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


class Job(Base, TimestampMixin):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    #: 'generate' | 'rectify'
    kind: Mapped[str] = mapped_column(String(16), default="generate")
    #: see pipeline.state_machine.STATES
    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    stage: Mapped[str] = mapped_column(String(32), default="created")
    input_mode: Mapped[str] = mapped_column(String(16), default="text")
    source_artifact_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    transcript_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    text_requirements: Mapped[str | None] = mapped_column(Text, nullable=True)
    locale: Mapped[str] = mapped_column(String(16), default="en")

    parent_job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parent_video_version_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rectification_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    rectification_category: Mapped[str | None] = mapped_column(String(32), nullable=True)

    house_spec_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_version_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    scene_manifest_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    shot_plan_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    video_version_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    clarification: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retryable: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    budget_cents: Mapped[int] = mapped_column(Integer, default=0)
    spent_cents: Mapped[int] = mapped_column(Integer, default=0)
    current_step: Mapped[str | None] = mapped_column(String(64), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    diagnostics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    project: Mapped[Project] = relationship(back_populates="jobs")


class PipelineStep(Base):
    """Durable queue row + step bookkeeping (lease based, restart safe)."""

    __tablename__ = "pipeline_steps"
    __table_args__ = (
        UniqueConstraint("job_id", "name", name="uq_step_per_job"),
        Index("ix_steps_job_seq", "job_id", "sequence"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    #: 'pending' | 'queued' | 'running' | 'succeeded' | 'failed' | 'skipped'
    state: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    output: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class JobEvent(Base):
    __tablename__ = "job_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    stage: Mapped[str] = mapped_column(String(32), default="")
    status: Mapped[str] = mapped_column(String(32), default="")
    message: Mapped[str] = mapped_column(String(500), default="")
    level: Mapped[str] = mapped_column(String(16), default="info")
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    scope: Mapped[str] = mapped_column(String(32), default="create_job")
    request_hash: Mapped[str] = mapped_column(String(64), default="")
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------
# Provider work
# ---------------------------------------------------------------------------


class ProviderTask(Base):
    __tablename__ = "provider_tasks"
    __table_args__ = (Index("ix_ptask_job_shot", "job_id", "shot_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(String(32), index=True)
    shot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(24), nullable=False)
    model: Mapped[str] = mapped_column(String(64), default="")
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    external_id: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    #: canonical: 'submitted' | 'processing' | 'succeeded' | 'failed' | 'expired'
    status: Mapped[str] = mapped_column(String(24), default="submitted", index=True)
    request_hash: Mapped[str] = mapped_column(String(64), default="")
    request_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    idempotency_key: Mapped[str] = mapped_column(String(128), default="", index=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class GeneratedClip(Base):
    __tablename__ = "generated_clips"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(String(32), index=True)
    shot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_task_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    provider: Mapped[str] = mapped_column(String(24), default="")
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    #: 'accepted' | 'rejected'
    status: Mapped[str] = mapped_column(String(16), default="accepted")
    duration_s: Mapped[float] = mapped_column(default=0.0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float] = mapped_column(default=0.0)
    checksum: Mapped[str] = mapped_column(String(64), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------
# Quality + output
# ---------------------------------------------------------------------------


class QualityResult(Base):
    __tablename__ = "quality_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str] = mapped_column(String(32), index=True)
    #: 'clip' | 'final'
    target: Mapped[str] = mapped_column(String(16), default="final")
    target_id: Mapped[str] = mapped_column(String(64), default="")
    evaluator_version: Mapped[str] = mapped_column(String(16), default="1.0.0")
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    severity: Mapped[str] = mapped_column(String(16), default="none")
    signals: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    defects: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    reason_code: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class VideoVersion(Base):
    """Immutable delivered film. Corrections create children, never edits."""

    __tablename__ = "video_versions"
    __table_args__ = (
        UniqueConstraint("project_id", "version", name="uq_video_version"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(32), index=True)
    job_id: Mapped[str] = mapped_column(String(32), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    model_version_id: Mapped[str] = mapped_column(String(32), index=True)
    scene_manifest_id: Mapped[str] = mapped_column(String(32), default="")
    shot_plan_id: Mapped[str] = mapped_column(String(32), default="")
    provider: Mapped[str] = mapped_column(String(24), default="local_render")
    #: 'external_ai' | 'model_render'
    generation_mode: Mapped[str] = mapped_column(String(16), default="model_render")
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    poster_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    duration_s: Mapped[float] = mapped_column(default=0.0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float] = mapped_column(default=0.0)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    checksum: Mapped[str] = mapped_column(String(64), default="")
    #: 'ready' | 'failed'
    state: Mapped[str] = mapped_column(String(16), default="ready", index=True)
    qa_passed: Mapped[bool] = mapped_column(Boolean, default=False)
    qa_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    cost_cents: Mapped[int] = mapped_column(Integer, default=0)
    cost_basis: Mapped[str] = mapped_column(String(16), default="unmetered")
    title: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class RectificationRequest(Base):
    __tablename__ = "rectification_requests"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(String(32), index=True)
    job_id: Mapped[str] = mapped_column(String(32), index=True)
    parent_video_version_id: Mapped[str] = mapped_column(String(32), index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    #: 'presentation' | 'scene_composition' | 'camera' | 'structural' | 'clarification'
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    classifier: Mapped[str] = mapped_column(String(32), default="rules")
    confidence: Mapped[float] = mapped_column(default=0.0)
    targets: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    affected_shots: Mapped[list[str]] = mapped_column(JSON, default=list)
    applied: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ShareLink(Base):
    __tablename__ = "share_links"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    video_version_id: Mapped[str] = mapped_column(String(32), index=True)
    project_id: Mapped[str] = mapped_column(String(32), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_by: Mapped[str] = mapped_column(String(32), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class UsageLedger(Base):
    __tablename__ = "usage_ledger"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    job_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    project_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    operation: Mapped[str] = mapped_column(String(48), nullable=False)
    units: Mapped[float] = mapped_column(default=0.0)
    unit_name: Mapped[str] = mapped_column(String(24), default="request")
    cost_cents: Mapped[int] = mapped_column(Integer, default=0)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    #: 'provider_reported' | 'configured_rate' | 'unmetered'
    cost_basis: Mapped[str] = mapped_column(String(24), default="unmetered")
    estimated: Mapped[bool] = mapped_column(Boolean, default=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
