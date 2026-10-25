"""Job creation, progress and control.

The single-page UI needs three things from this router:
* create one job from drawing / voice / text with a single button,
* recover the active job after a refresh,
* show truthful, event-derived progress until a video exists.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from ..db import session_scope, utcnow
from ..errors import ValidationError
from ..models import Job, VideoVersion
from ..pipeline import state_machine as sm
from ..services import orchestrator
from ..services.jobs import latest_active_job, request_hash, get_idempotent_job
from .deps import CurrentUser, DbSession, IdempotencyKey, load_job
from .schemas import GenerateOut, JobOut, MessageOut
from .serializers import job_out, video_out

router = APIRouter(tags=["jobs"])
log = logging.getLogger("buildvisual.api.jobs")

MAX_READ = 64 * 1024 * 1024  # hard ceiling before validation runs


def _human_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.0f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.0f} KB"
    return f"{num_bytes} bytes"


async def _read_upload(upload: UploadFile | None, limit: int) -> tuple[str, bytes] | None:
    if upload is None or not upload.filename:
        return None
    data = await upload.read(limit + 1)
    await upload.close()
    if len(data) > limit:
        raise ValidationError(
            f"That file is {_human_size(len(data))}; this deployment accepts up to "
            f"{_human_size(limit)} per upload. Reduce it and try again.",
            details={"size_bytes": len(data), "limit_bytes": limit},
        )
    return (upload.filename, data)


@router.post("/jobs", response_model=GenerateOut, status_code=201)
async def create_job(
    request: Request,
    db: DbSession,
    user: CurrentUser,
    idempotency: IdempotencyKey,
    drawing: Annotated[UploadFile | None, File()] = None,
    audio: Annotated[UploadFile | None, File()] = None,
    text: Annotated[str | None, Form()] = None,
    project_id: Annotated[str | None, Form()] = None,
    project_name: Annotated[str | None, Form()] = None,
    locale: Annotated[str, Form()] = "en",
) -> GenerateOut:
    from ..config import get_settings

    settings = get_settings()
    drawing_payload = await _read_upload(drawing, settings.max_upload_bytes)
    audio_payload = await _read_upload(audio, settings.max_audio_bytes)

    replay = get_idempotent_job(db, idempotency or "", user.id) if idempotency else None
    if replay is not None:
        return GenerateOut(job=job_out(db, replay), replayed=True)

    job = orchestrator.create_generation_job(
        db,
        user_id=user.id,
        project_id=project_id or None,
        project_name=project_name,
        text=text,
        drawing=drawing_payload,
        audio=audio_payload,
        locale=locale or "en",
        idempotency_key=idempotency,
    )
    db.commit()
    return GenerateOut(job=job_out(db, job), replayed=False)


@router.get("/jobs/active", response_model=JobOut | None)
def active_job(db: DbSession, user: CurrentUser) -> JobOut | None:
    """Refresh recovery: the newest job that has not reached a terminal state."""
    job = latest_active_job(db, user.id)
    if job is None:
        return None
    return job_out(db, job)


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: DbSession, user: CurrentUser, events: bool = True) -> JobOut:
    job = load_job(db, job_id, user)
    return job_out(db, job, include_events=events)


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str, db: DbSession, user: CurrentUser) -> dict:
    job = load_job(db, job_id, user)
    payload = job_out(db, job)
    return {
        "id": job.id,
        "stage": job.stage,
        "status": job.status,
        "message": payload.current_message,
        "events": payload.events,
    }


@router.get("/jobs/{job_id}/stream")
def stream_job(job_id: str, db: DbSession, user: CurrentUser) -> StreamingResponse:
    """Server-sent events: stage changes as they happen (no invented percentages)."""
    load_job(db, job_id, user)

    def _events():
        import time

        last = None
        deadline = time.time() + 60 * 30
        while time.time() < deadline:
            with session_scope() as session:
                job = session.get(Job, job_id)
                if job is None:
                    yield "event: gone\ndata: {}\n\n"
                    return
                payload = {
                    "stage": job.stage,
                    "status": job.status,
                    "message": job_out(session, job, include_events=False).current_message,
                    "video_version_id": job.video_version_id,
                    "error": (job.error_message if job.status == "failed" else None),
                }
                body = json.dumps(payload)
                if body != last:
                    last = body
                    yield f"event: update\ndata: {body}\n\n"
                if job.stage in sm.TERMINAL:
                    yield "event: end\ndata: {}\n\n"
                    return
            time.sleep(1.5)
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/jobs/{job_id}/retry", response_model=JobOut)
def retry_job(job_id: str, db: DbSession, user: CurrentUser) -> JobOut:
    job = load_job(db, job_id, user)
    orchestrator.retry_job(db, job, user.id)
    db.commit()
    return job_out(db, job)


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, db: DbSession, user: CurrentUser) -> JobOut:
    job = load_job(db, job_id, user)
    orchestrator.cancel(db, job, user.id)
    db.commit()
    return job_out(db, job)


@router.get("/jobs/{job_id}/steps")
def job_steps(job_id: str, db: DbSession, user: CurrentUser) -> dict:
    """Step-level diagnostics (no prompts, no provider payloads)."""
    job = load_job(db, job_id, user)
    return {
        "job_id": job.id,
        "stage": job.stage,
        "steps": orchestrator.retry_status(db, job),
        "attempts": job.attempts,
        "spent_cents": job.spent_cents,
        "deadline_at": job.deadline_at.isoformat() + "Z" if job.deadline_at else None,
    }


@router.get("/videos/{video_id}", response_model=dict)
def video_detail(video_id: str, db: DbSession, user: CurrentUser) -> dict:
    from ..errors import NotFound, PermissionDenied

    version = db.get(VideoVersion, video_id)
    if version is None:
        raise NotFound("That video could not be found.")
    project = orchestrator.get_project(db, version.project_id, user.id)
    payload = video_out(db, version)
    return {
        "video": payload.model_dump(mode="json") if payload else None,
        "project": {"id": project.id, "name": project.name},
        "history": [
            {
                "id": row.id,
                "version": row.version,
                "created_at": row.created_at.isoformat() + "Z",
                "qa_passed": row.qa_passed,
                "duration_s": round(row.duration_s, 2),
                "parent_id": row.parent_id,
            }
            for row in db.execute(
                select(VideoVersion).where(VideoVersion.project_id == project.id)
                .order_by(VideoVersion.version.desc())
            ).scalars().all()
        ],
    }


@router.get("/videos/{video_id}/stream.mp4")
def video_stream(video_id: str, db: DbSession, user: CurrentUser):
    """Same-origin playback endpoint (CSP keeps media on 'self').

    The browser gets a short-lived signed URL rather than a storage path, so the
    player never learns where the file lives and links cannot be shared blindly.
    """
    from fastapi.responses import Response

    from ..config import get_settings
    from ..errors import NotFound
    from ..storage import sign_media_token

    version = db.get(VideoVersion, video_id)
    if version is None:
        raise NotFound("That video could not be found.")
    orchestrator.get_project(db, version.project_id, user.id)
    if version.state != "ready":
        raise NotFound("This video is not available yet.")
    token = sign_media_token(
        version.storage_key,
        expires_in=get_settings().download_url_ttl_seconds,
        scope="video",
    )
    return Response(
        status_code=307,
        headers={
            "Location": f"/api/media/{token}?fn=buildvisual-v{version.version}.mp4",
            "Cache-Control": "no-store",
        },
    )


@router.post("/videos/{video_id}/save", response_model=MessageOut)
def save_to_project(video_id: str, db: DbSession, user: CurrentUser) -> MessageOut:
    """Explicit "Save to Project".

    A published version is always attached to its project; this records that the
    engineer deliberately kept this version, which is what the project list shows.
    """
    from ..errors import NotFound
    from ..models import Project  # noqa: F401  (keeps the FK relationship visible)

    version = db.get(VideoVersion, video_id)
    if version is None:
        raise NotFound("That video could not be found.")
    project = orchestrator.get_project(db, version.project_id, user.id)
    project.status = "saved"
    project.updated_at = utcnow()
    project.latest_video_version_id = version.id
    db.commit()
    return MessageOut(message=f"Version {version.version} saved to {project.name}.")
