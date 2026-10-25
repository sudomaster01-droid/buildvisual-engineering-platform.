"""Signed, authenticated media delivery.

Two access paths, both narrow:
* the owner: ``/videos/{id}/download`` mints a short-lived signed token, then
  ``/media/{token}`` streams the file (no storage key ever reaches the browser);
* a share link: ``/share/{token}`` resolves a hashed, expiring, revocable token.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from datetime import timedelta

from ..config import get_settings
from ..db import utcnow
from ..errors import NotFound, PermissionDenied
from ..models import ShareLink, VideoVersion
from ..services import orchestrator
from ..storage import get_storage, parse_media_token, sign_media_token
from .deps import CurrentUser, DbSession
from .schemas import MessageOut, ShareIn, ShareListOut, ShareOut, VideoDownloadOut
from .serializers import video_out

router = APIRouter(tags=["media"])
log = logging.getLogger("buildvisual.api.media")


def _signed_url(key: str, *, scope: str, filename: str) -> tuple[str, int]:
    settings = get_settings()
    expires_in = settings.download_url_ttl_seconds
    token = sign_media_token(key, expires_in=expires_in, scope=scope)
    base = settings.public_base_url.rstrip("/")
    return f"{base}/api/media/{token}?fn={filename}", expires_in


@router.get("/videos/{video_id}/download", response_model=VideoDownloadOut)
def video_download(video_id: str, db: DbSession, user: CurrentUser) -> VideoDownloadOut:
    version = db.get(VideoVersion, video_id)
    if version is None:
        raise NotFound("That video could not be found.")
    orchestrator.get_project(db, version.project_id, user.id)
    if version.state != "ready" or not version.qa_passed:
        raise NotFound("This video version is not available for download.")
    filename = f"buildvisual-v{version.version}.mp4"
    url, expires = _signed_url(version.storage_key, scope="video", filename=filename)
    return VideoDownloadOut(url=url, expires_in=expires, filename=filename)


@router.get("/videos/{video_id}/poster")
def video_poster(video_id: str, db: DbSession, user: CurrentUser) -> Response:
    version = db.get(VideoVersion, video_id)
    if version is None or not version.poster_key:
        raise NotFound("No poster frame is available for this video.")
    orchestrator.get_project(db, version.project_id, user.id)
    url, _ = _signed_url(version.poster_key, scope="poster", filename="poster.jpg")
    return Response(status_code=307, headers={"Location": url})


@router.get("/media/{token}")
def serve_signed(token: str, request: Request, fn: str | None = None) -> Response:
    """Resolve a signed token to an object. Only the owner's mints are accepted."""
    storage = get_storage()
    key = _resolve_key(token)
    if key is None:
        raise PermissionDenied("This download link is no longer valid.")
    data = storage.open_reader(key)
    size = storage.size(key)
    media_type = "video/mp4" if key.endswith(".mp4") else "image/jpeg"
    return StreamingResponse(
        _iter_file(data, size),
        media_type=media_type,
        headers={
            "Content-Length": str(size),
            "Content-Disposition": f'inline; filename="{_safe_filename(fn or Path(key).name)}"',
            "Cache-Control": "private, max-age=300",
        },
    )


def _iter_file(handle, size: int, chunk: int = 512 * 1024):
    try:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            yield block
    finally:
        handle.close()


def _safe_filename(name: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120]
    return cleaned or "video.mp4"


def _resolve_key(token: str) -> str | None:
    """Verify a media token and return the object key it authorises."""
    parsed = parse_media_token(token)
    return parsed[0] if parsed else None


# ---------------------------------------------------------------------------
# Share links
# ---------------------------------------------------------------------------


@router.post("/videos/{video_id}/share", response_model=ShareOut, status_code=201)
def create_share(video_id: str, payload: ShareIn, db: DbSession, user: CurrentUser) -> ShareOut:
    import hashlib
    import secrets

    settings = get_settings()
    version = db.get(VideoVersion, video_id)
    if version is None:
        raise NotFound("That video could not be found.")
    orchestrator.get_project(db, version.project_id, user.id)
    if version.state != "ready" or not version.qa_passed:
        raise PermissionDenied("Only a video that passed quality checks can be shared.")
    days = min(payload.expires_days, settings.share_max_expiry_days)
    raw = secrets.token_urlsafe(32)
    link = ShareLink(
        video_version_id=version.id,
        project_id=version.project_id,
        token_hash=hashlib.sha256(raw.encode()).hexdigest(),
        created_by=user.id,
        expires_at=utcnow() + timedelta(days=days),
    )
    db.add(link)
    db.commit()
    base = settings.public_base_url.rstrip("/")
    return ShareOut(
        id=link.id,
        url=f"{base}/share/{raw}",
        expires_at=link.expires_at,
        view_count=link.view_count,
    )


@router.get("/videos/{video_id}/shares", response_model=ShareListOut)
def list_shares(video_id: str, db: DbSession, user: CurrentUser) -> ShareListOut:
    from sqlalchemy import select

    version = db.get(VideoVersion, video_id)
    if version is None:
        raise NotFound("That video could not be found.")
    orchestrator.get_project(db, version.project_id, user.id)
    rows = db.execute(
        select(ShareLink).where(ShareLink.video_version_id == video_id)
        .order_by(ShareLink.created_at.desc())
    ).scalars().all()
    base = get_settings().public_base_url.rstrip("/")
    return ShareListOut(shares=[
        ShareOut(
            id=row.id,
            # The raw token is never stored, so an existing link cannot be re-shown;
            # revoke and create a new one instead.
            url=f"{base}/share/…" if row.revoked_at else f"{base}/share/<redacted>",
            expires_at=row.expires_at,
            view_count=row.view_count,
        )
        for row in rows
    ])


@router.delete("/shares/{share_id}", response_model=MessageOut)
def revoke_share(share_id: str, db: DbSession, user: CurrentUser) -> MessageOut:
    link = db.get(ShareLink, share_id)
    if link is None:
        raise NotFound("That share link could not be found.")
    orchestrator.get_project(db, link.project_id, user.id)
    if link.revoked_at is None:
        link.revoked_at = utcnow()
        db.commit()
    return MessageOut(message="Share link revoked.")
