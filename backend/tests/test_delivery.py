"""Delivery security: signed downloads, share links, and cross-account access."""

from __future__ import annotations

import pytest

HOUSE_TEXT = "Two bedroom bungalow with a flat roof and a small veranda"


@pytest.fixture
def film(client, login):
    from app.db import session_scope
    from app.models import VideoVersion
    from app.pipeline.runner import run_job_to_completion
    from app.services import orchestrator

    login()
    with session_scope() as session:
        user_id = session.query(__import__("app.models", fromlist=["User"]).User).first().id
        job = orchestrator.create_generation_job(
            session, user_id=user_id, project_id=None, project_name="Delivery test",
            text=HOUSE_TEXT, drawing=None, audio=None,
        )
        job_id = job.id
    finished = run_job_to_completion(job_id, worker_id="pytest-delivery")
    assert finished.stage == "completed", (finished.stage, finished.error_message)
    with session_scope() as session:
        version = session.get(VideoVersion, finished.video_version_id)
        return {"id": version.id, "user_id": user_id, "project_id": version.project_id,
                "storage_key": version.storage_key}


# ---------------------------------------------------------------------------
# Signed downloads
# ---------------------------------------------------------------------------


def test_download_link_is_signed_and_expiring(client, login, film):
    login()
    payload = client.get(f"/api/videos/{film['id']}/download").json()
    assert payload["expires_in"] <= 900
    assert payload["filename"].endswith(".mp4")
    # The storage key must not be guessable or readable from the URL.
    assert "videos/" not in payload["url"].split("?")[0]

    served = client.get(payload["url"].replace("http://testserver", ""))
    assert served.status_code == 200
    assert served.headers["content-type"] == "video/mp4"


def test_tampered_media_token_is_refused(client, login, film):
    login()
    url = client.get(f"/api/videos/{film['id']}/download").json()["url"]
    token = url.split("/api/media/")[1].split("?")[0]      # signature only
    tampered = token[:-6] + ("aaaaaa" if not token.endswith("aaaaaa") else "bbbbbb")
    assert client.get(f"/api/media/{tampered}").status_code == 403


def test_media_token_with_a_rewritten_key_is_refused(client, login, film):
    """Changing the encoded object key invalidates the HMAC."""
    import base64

    login()
    url = client.get(f"/api/videos/{film['id']}/download").json()["url"]
    token = url.split("/api/media/")[1].split("?")[0]
    encoded, rest = token.split(".", 1)
    key = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    forged_key = key.replace(film["id"], "0" * len(film["id"]))
    forged = base64.urlsafe_b64encode(forged_key.encode()).decode().rstrip("=")
    assert client.get(f"/api/media/{forged}.{rest}").status_code == 403


def test_expired_media_token_is_refused(client, login, film, monkeypatch):
    """A link stops working once its expiry passes."""
    import time

    from app.storage import sign_media_token

    login()
    token = sign_media_token(film["storage_key"], expires_in=60, scope="video")
    assert client.get(f"/api/media/{token}").status_code == 200

    future = time.time() + 3600
    monkeypatch.setattr("app.storage.time.time", lambda: future)
    assert client.get(f"/api/media/{token}").status_code == 403


def test_media_token_cannot_be_replayed_for_another_object(client, login, film):
    from app.storage import sign_media_token, verify_download

    token = sign_media_token(film["storage_key"], expires_in=60, scope="video")
    other_key = "videos/someone/else.mp4"
    # The HMAC binds the key: the same signature is invalid for a different object.
    assert verify_download(other_key, token.split(".", 1)[0] + ".extra.parts") is False


# ---------------------------------------------------------------------------
# Share links
# ---------------------------------------------------------------------------


def test_share_link_works_and_can_be_revoked(client, login, film):
    login()
    share = client.post(f"/api/videos/{film['id']}/share", json={"expires_days": 3})
    assert share.status_code == 201
    url = share.json()["url"]
    token = url.rsplit("/", 1)[-1]

    anonymous = client.__class__(client.app, base_url="http://testserver")
    page = anonymous.get(f"/share/{token}")
    assert page.status_code == 200
    assert "<video" in page.text
    assert "noindex" in page.headers.get("x-robots-tag", "")

    info = anonymous.get(f"/share/{token}/info").json()
    assert info["video_url"].startswith("/api/media/")

    # Watching through the share link actually delivers bytes.
    media = anonymous.get(info["video_url"])
    assert media.status_code == 200 and len(media.content) > 1000

    assert client.delete(f"/api/shares/{share.json()['id']}").status_code == 200
    assert anonymous.get(f"/share/{token}").status_code == 404


def test_unknown_share_token_is_not_found(client):
    assert client.get("/share/not-a-real-token").status_code == 404
    assert client.get("/share/../etc/passwd").status_code in (404, 400)


def test_share_expiry_is_capped(client, login, film):
    from app.config import get_settings

    login()
    response = client.post(f"/api/videos/{film['id']}/share", json={"expires_days": 3650})
    assert response.status_code == 201
    from datetime import timedelta

    from app.db import utcnow

    expires = response.json()["expires_at"].replace("Z", "")
    from datetime import datetime

    delta = datetime.fromisoformat(expires) - utcnow()
    assert delta <= timedelta(days=get_settings().share_max_expiry_days + 1)


# ---------------------------------------------------------------------------
# Cross-account access
# ---------------------------------------------------------------------------


def test_another_account_cannot_reach_the_film(client, login, film):
    from app.db import session_scope
    from app.models import User
    from app.security import hash_password

    with session_scope() as session:
        if not session.query(User).filter_by(email="intruder@example.com").first():
            session.add(User(email="intruder@example.com",
                             password_hash=hash_password("intruder-password-1")))
            session.commit()

    client.cookies.clear()
    intruder = client.post(
        "/api/auth/login",
        json={"email": "intruder@example.com", "password": "intruder-password-1"},
    )
    assert intruder.status_code == 200
    client.headers.update({"X-CSRF-Token": intruder.json()["csrf_token"]})

    assert client.get(f"/api/videos/{film['id']}").status_code in (403, 404)
    assert client.get(f"/api/videos/{film['id']}/download").status_code in (403, 404)
    assert client.get(f"/api/projects/{film['project_id']}/videos").status_code in (403, 404)
    assert client.post(f"/api/videos/{film['id']}/share", json={"expires_days": 1}).status_code in (403, 404)
    assert client.post(f"/api/videos/{film['id']}/rectify", json={"text": "warmer"}).status_code in (403, 404)


def test_storage_keys_never_appear_in_api_responses(client, login, film):
    login()
    for path in (f"/api/videos/{film['id']}", f"/api/jobs/{film['project_id']}"):
        body = client.get(path).text
        assert "objects/" not in body
        assert film["storage_key"] not in body
