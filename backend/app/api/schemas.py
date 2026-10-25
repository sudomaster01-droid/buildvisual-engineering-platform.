"""Pydantic request/response contracts for the HTTP API.

These are the only shapes the browser sees. Internal records (prompts, provider
payloads, storage keys) are never serialised directly, which is what keeps the
private prompt layer private and prevents leaking local paths or provider URLs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SessionOut(BaseModel):
    user_id: str
    email: str
    display_name: str
    csrf_token: str
    session_expires_at: datetime | None = None
    issued_at: datetime | None = None


class LoginStatusOut(BaseModel):
    """Always-200 sign-in state probe (see ``/api/auth/status``)."""

    authenticated: bool
    setup_required: bool = False
    display_name: str | None = None
    email: str | None = None
    email_hint: str | None = None
    csrf_token: str | None = None
    message: str | None = None


class LoginIn(BaseModel):
    #: Length is validated here; format is not over-constrained, because a long
    #: but valid address must never be rejected as a "wrong password".
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)


class CapabilitiesOut(BaseModel):
    drawing_interpretation: bool
    transcription: bool
    video_provider: str
    video_provider_external: bool
    video_provider_display: str
    video_note: str
    accepted_drawing_formats: list[str]
    accepted_audio_formats: list[str]
    max_upload_mb: int
    max_audio_seconds: int
    max_text_chars: int
    max_shots: int
    shot_seconds: int
    llm_provider: str
    limitations: list[str]


class ProjectIn(BaseModel):
    name: str = Field(default="Untitled house", max_length=160)


class ProjectOut(BaseModel):
    id: str
    name: str
    status: str
    created_at: datetime
    updated_at: datetime
    latest_video_version_id: str | None = None
    video_count: int = 0
    latest_version: int | None = None


class StageOut(BaseModel):
    key: str
    label: str
    state: Literal["done", "current", "pending", "failed", "skipped"]


class DefectOut(BaseModel):
    code: str
    severity: str
    message: str


class VideoOut(BaseModel):
    id: str
    version: int
    title: str
    state: str
    generation_mode: str
    provider: str
    duration_s: float
    width: int
    height: int
    size_bytes: int
    qa_passed: bool
    qa_summary: dict[str, Any] = {}
    created_at: datetime
    has_poster: bool = False
    parent_id: str | None = None


class VideoDownloadOut(BaseModel):
    url: str
    expires_in: int
    filename: str


class JobOut(BaseModel):
    id: str
    project_id: str
    project_name: str
    kind: str
    stage: str
    status: str
    input_mode: str
    progress: list[StageOut]
    current_message: str
    message_level: str = "info"
    events: list[dict[str, Any]] = []
    error: dict[str, Any] | None = None
    retryable: bool = False
    clarification: dict[str, Any] | None = None
    video: VideoOut | None = None
    created_at: datetime
    updated_at: datetime
    elapsed_seconds: float
    cost: dict[str, Any] = {}
    rectification: dict[str, Any] | None = None
    interpretation: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None


class GenerateOut(BaseModel):
    job: JobOut
    replayed: bool = False


class RectifyIn(BaseModel):
    text: str = Field(min_length=3, max_length=2000)


class RectifyOut(BaseModel):
    job: JobOut
    understanding: dict[str, Any]


class ShareIn(BaseModel):
    #: Upper bound is enforced by SHARE_MAX_EXPIRY_DAYS in configuration, so a
    #: generous value here is clamped rather than rejected with a validation error.
    expires_days: int = Field(default=7, ge=1, le=3650)


class ShareOut(BaseModel):
    id: str
    url: str
    expires_at: datetime
    view_count: int = 0


class ShareListOut(BaseModel):
    shares: list[ShareOut]


class PublicShareOut(BaseModel):
    title: str
    version: int
    duration_s: float
    width: int
    height: int
    created_at: datetime
    video_url: str
    poster_url: str | None = None
    generation_mode: str


class MessageOut(BaseModel):
    message: str
