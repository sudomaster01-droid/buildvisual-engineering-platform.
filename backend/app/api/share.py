"""Public (unauthenticated) share links.

A share is a capability URL: hashed at rest, time-limited, revocable, and scoped
to exactly one video version. It exposes the film and its basic facts and nothing
else — no project list, no prompts, no model data.
"""

from __future__ import annotations

import hashlib
import html

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db, utcnow
from ..errors import NotFound
from ..models import ShareLink, VideoVersion
from ..storage import sign_media_token
from .schemas import PublicShareOut

router = APIRouter(tags=["share"])

DbSession = Depends(get_db)


def _resolve(db: Session, token: str) -> ShareLink:
    if not token or len(token) > 200:
        raise NotFound("This link is not available.")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    link = db.execute(
        select(ShareLink).where(ShareLink.token_hash == token_hash)
    ).scalar_one_or_none()
    if link is None or link.revoked_at is not None:
        raise NotFound("This link is not available.")
    if link.expires_at <= utcnow():
        raise NotFound("This link has expired.")
    return link


def _video_for(db: Session, link: ShareLink) -> VideoVersion:
    version = db.get(VideoVersion, link.video_version_id)
    if version is None or version.state != "ready":
        raise NotFound("This link is not available.")
    return version


def _media_urls(version: VideoVersion) -> tuple[str, str | None]:
    settings = get_settings()
    ttl = settings.download_url_ttl_seconds
    video = sign_media_token(version.storage_key, expires_in=ttl, scope="video")
    poster = None
    if version.poster_key:
        token = sign_media_token(version.poster_key, expires_in=ttl, scope="poster")
        poster = f"/api/media/{token}?fn=poster.jpg"
    return f"/api/media/{video}?fn=buildvisual-v{version.version}.mp4", poster


@router.get("/share/{token}", response_class=HTMLResponse)
def share_page(token: str, db: Session = DbSession) -> HTMLResponse:
    link = _resolve(db, token)
    version = _video_for(db, link)
    link.view_count = (link.view_count or 0) + 1
    db.commit()

    video_url, poster_url = _media_urls(version)
    whole_minutes = int(version.duration_s // 60)
    seconds = int(version.duration_s % 60)
    title = html.escape(version.title or "Architectural walkthrough")
    mode_note = (
        "AI-generated scenes guided by a locked 3D model."
        if version.generation_mode == "external_ai"
        else "Rendered directly from the locked 3D model (model render, not AI footage)."
    )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; min-height:100vh; background:#070d1c; color:#e6eefc;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif; }}
  header {{ display:flex; align-items:center; gap:10px; padding:18px 22px;
            border-bottom:1px solid rgba(120,170,255,.16); }}
  .mark {{ width:26px;height:26px;border-radius:8px;
           background:linear-gradient(135deg,#22d3ee,#6366f1); }}
  .brand {{ font-weight:600; letter-spacing:.4px; }}
  main {{ max-width: 940px; margin: 0 auto; padding: 26px 20px 60px; }}
  h1 {{ font-size: clamp(20px,3vw,28px); font-weight:600; margin:0 0 6px; }}
  .meta {{ color:#93a7c9; font-size:14px; margin-bottom:18px; }}
  video {{ width:100%; border-radius:16px; background:#000; display:block;
           border:1px solid rgba(120,170,255,.22); }}
  .note {{ margin-top:14px; font-size:13px; color:#8ea3c6; }}
  .pill {{ display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
           background:rgba(34,211,238,.12); color:#7ee7f7;
           border:1px solid rgba(34,211,238,.28); margin-right:8px; }}
</style></head>
<body>
<header><div class="mark"></div><div class="brand">BuildVisual</div></header>
<main>
  <h1>{title}</h1>
  <div class="meta">
    <span class="pill">Version {version.version}</span>
    <span class="pill">{whole_minutes:02d}:{seconds:02d}</span>
    <span class="pill">{version.width}×{version.height}</span>
  </div>
  <video controls preload="metadata" playsinline{'' if not poster_url else f' poster="{poster_url}"'}
         src="{video_url}"></video>
  <p class="note">{html.escape(mode_note)} Shared by the architect. This link expires
  on {link.expires_at.strftime('%d %b %Y')}.</p>
</main>
</body></html>"""
    return HTMLResponse(
        body,
        headers={"X-Robots-Tag": "noindex, nofollow", "Cache-Control": "no-store"},
    )


@router.get("/share/{token}/info", response_model=PublicShareOut)
def share_info(token: str, db: Session = DbSession) -> PublicShareOut:
    link = _resolve(db, token)
    version = _video_for(db, link)
    video_url, poster_url = _media_urls(version)
    return PublicShareOut(
        title=version.title or "Architectural walkthrough",
        version=version.version,
        duration_s=version.duration_s,
        width=version.width,
        height=version.height,
        created_at=version.created_at,
        video_url=video_url,
        poster_url=poster_url,
        generation_mode=version.generation_mode,
    )


@router.get("/share/{token}/video")
def share_video(token: str, db: Session = DbSession) -> Response:
    link = _resolve(db, token)
    version = _video_for(db, link)
    video_url, _ = _media_urls(version)
    return Response(
        status_code=307, headers={"Location": video_url, "Cache-Control": "no-store"}
    )
