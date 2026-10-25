"""Project + version history endpoints.

CSRF is enforced centrally by middleware (see ``app.main``), so routers stay thin.
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import select

from ..models import Job, Project, VideoVersion
from ..pipeline import state_machine as sm
from ..services import orchestrator
from ..services.jobs import active_jobs_for_user
from .deps import CurrentUser, DbSession
from .schemas import MessageOut, ProjectIn, ProjectOut, VideoOut
from .serializers import project_out, video_out

router = APIRouter(tags=["projects"])


@router.get("/projects", response_model=list[ProjectOut])
def list_projects(
    db: DbSession, user: CurrentUser, limit: int = Query(default=50, le=200)
) -> list[ProjectOut]:
    rows = db.execute(
        select(Project).where(Project.user_id == user.id)
        .order_by(Project.updated_at.desc()).limit(limit)
    ).scalars().all()
    return [project_out(db, row) for row in rows]


@router.post("/projects", response_model=ProjectOut)
def create_project(payload: ProjectIn, db: DbSession, user: CurrentUser) -> ProjectOut:
    project = orchestrator.create_project(db, user.id, payload.name)
    db.commit()
    return project_out(db, project)


@router.get("/projects/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, db: DbSession, user: CurrentUser) -> ProjectOut:
    return project_out(db, orchestrator.get_project(db, project_id, user.id))


@router.get("/projects/{project_id}/videos", response_model=list[VideoOut])
def project_videos(project_id: str, db: DbSession, user: CurrentUser) -> list[VideoOut]:
    """Full immutable version history, newest first."""
    orchestrator.get_project(db, project_id, user.id)
    rows = db.execute(
        select(VideoVersion).where(VideoVersion.project_id == project_id)
        .order_by(VideoVersion.version.desc())
    ).scalars().all()
    return [v for v in (video_out(db, row) for row in rows) if v is not None]


@router.get("/projects/{project_id}/jobs")
def project_jobs(project_id: str, db: DbSession, user: CurrentUser) -> list[dict]:
    """Jobs belonging to a project, so a refresh can find an in-flight run."""
    orchestrator.get_project(db, project_id, user.id)
    rows = db.execute(
        select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc()).limit(50)
    ).scalars().all()
    return [
        {
            "id": row.id,
            "kind": row.kind,
            "stage": row.stage,
            "status": row.status,
            "active": row.stage not in sm.TERMINAL,
            "created_at": row.created_at.isoformat() + "Z",
        }
        for row in rows
    ]


@router.delete("/projects/{project_id}", response_model=MessageOut)
def delete_project(project_id: str, db: DbSession, user: CurrentUser) -> MessageOut:
    project = orchestrator.get_project(db, project_id, user.id)
    if any(job.project_id == project_id for job in active_jobs_for_user(db, user.id)):
        from ..errors import Conflict

        raise Conflict("Cancel the running job before deleting this project.")
    db.delete(project)
    db.commit()
    return MessageOut(message="Project deleted.")
