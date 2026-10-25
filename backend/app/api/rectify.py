"""Review & rectify endpoint.

The four correction classes are handled by ``services.rectify``; this router only
performs authorisation, calls orchestration and reports how the request was
understood so the engineer can see what will change before work starts.
"""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from ..errors import NotFound
from ..models import Job, RectificationRequest, VideoVersion
from ..services import orchestrator
from .deps import CurrentUser, DbSession
from .schemas import RectifyIn, RectifyOut
from .serializers import job_out

router = APIRouter(tags=["rectify"])


@router.post("/videos/{video_id}/rectify", response_model=RectifyOut, status_code=201)
def rectify(
    video_id: str, payload: RectifyIn, db: DbSession, user: CurrentUser
) -> RectifyOut:
    version = db.get(VideoVersion, video_id)
    if version is None:
        raise NotFound("That video version could not be found.")
    orchestrator.get_project(db, version.project_id, user.id)

    job, summary = orchestrator.create_rectification_job(
        db, user_id=user.id, video_version_id=video_id, text=payload.text
    )
    db.commit()
    return RectifyOut(job=job_out(db, job), understanding=summary)


@router.get("/videos/{video_id}/rectifications")
def rectification_history(video_id: str, db: DbSession, user: CurrentUser) -> list[dict]:
    version = db.get(VideoVersion, video_id)
    if version is None:
        raise NotFound("That video version could not be found.")
    orchestrator.get_project(db, version.project_id, user.id)
    rows = db.execute(
        select(RectificationRequest, Job)
        .join(Job, Job.id == RectificationRequest.job_id)
        .where(RectificationRequest.parent_video_version_id == video_id)
        .order_by(RectificationRequest.created_at.desc())
    ).all()
    return [
        {
            "id": request.id,
            "text": request.text,
            "category": request.category,
            "classifier": request.classifier,
            "confidence": round(request.confidence, 3),
            "job_id": request.job_id,
            "job_stage": job.stage,
            "job_status": job.status,
            "requires_new_model": bool(request.applied.get("requires_new_model")),
            "created_at": request.created_at.isoformat() + "Z",
        }
        for request, job in rows
    ]
