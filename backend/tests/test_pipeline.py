"""End-to-end pipeline: text → model → design → references → film → QA → delivery.

Uses the built-in raster renderer, so no external API key is needed and the run is
deterministic. This exercises the same state machine, persistence, assembly and
quality gate that an external provider travels through.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

HOUSE_TEXT = (
    "Three bedroom bungalow with a flat roof, open plan living and dining, "
    "a garage and a covered veranda at the front"
)


def _run(client, headers=None):
    from app.pipeline.runner import run_job_to_completion

    response = client.post(
        "/api/jobs",
        data={"text": HOUSE_TEXT, "project_name": "Pipeline test house"},
        headers=headers or {},
    )
    assert response.status_code == 201, response.text
    job_id = response.json()["job"]["id"]
    job = run_job_to_completion(job_id, worker_id="pytest")
    return job_id, job


def test_full_generation_produces_a_playable_film(client, login):
    login()
    job_id, job = _run(client, {"Idempotency-Key": "pipe-1"})

    assert job.stage == "completed", (job.stage, job.error_code, job.error_message)
    assert job.video_version_id

    detail = client.get(f"/api/jobs/{job_id}").json()
    video = detail["video"]
    assert video["version"] == 1
    assert video["state"] == "ready"
    assert video["qa_passed"] is True
    from app.config import get_settings

    settings = get_settings()
    assert video["width"] == settings.video_width
    assert video["height"] == settings.video_height
    assert video["duration_s"] > 0
    assert video["generation_mode"] == "model_render"

    # The artefact exists on disk and is a real decodable MP4.
    from app.db import session_scope
    from app.models import VideoVersion
    from app.services.media import ffmpeg as ff
    from app.storage import get_storage

    with session_scope() as session:
        row = session.get(VideoVersion, video["id"])
        local = get_storage().local_path(row.storage_key)
        assert local is not None and local.exists()
        info = ff.probe(local)
        assert info.duration_s > 0
        assert info.codec in {"h264", "hevc", "vp9"}
        assert local.read_bytes()[:12][4:8] == b"ftyp"

    # Download is a signed, expiring URL that actually serves the bytes.
    download = client.get(f"/api/videos/{video['id']}/download")
    assert download.status_code == 200
    url = download.json()["url"]
    assert "/api/media/" in url
    media = client.get(url.replace("http://testserver", ""))
    assert media.status_code == 200
    assert media.headers["content-type"] == "video/mp4"
    assert len(media.content) > 10_000


def test_pipeline_records_versions_quality_and_cost(client, login):
    login()
    job_id, job = _run(client, {"Idempotency-Key": "pipe-2"})

    from app.db import session_scope
    from app.models import (
        GeneratedClip, HouseSpec, ModelVersion, QualityResult, SceneManifest,
        ShotPlan, UsageLedger, VideoVersion,
    )

    with session_scope() as session:
        assert session.query(HouseSpec).count() == 1
        assert session.query(ModelVersion).count() == 1
        assert session.query(SceneManifest).count() == 1
        plans = session.query(ShotPlan).all()
        assert plans and len(plans[0].plan["shots"]) >= 1
        clips = session.query(GeneratedClip).filter_by(status="accepted").all()
        assert clips, "no accepted clips were recorded"
        for clip in clips:
            assert clip.checksum and clip.duration_s > 0
        quality = session.query(QualityResult).filter_by(target="final").all()
        assert quality and quality[-1].passed is True
        versions = session.query(VideoVersion).all()
        assert len(versions) == 1 and versions[0].qa_passed is True
        # Cost is recorded honestly: local rendering is unmetered, not invented spend.
        ledger = session.query(UsageLedger).all()
        assert ledger
        assert all(row.cost_basis == "unmetered" for row in ledger)
        assert all(row.cost_cents == 0 for row in ledger)


def test_idempotency_key_replays_the_same_job(client, login):
    login()
    headers = {"Idempotency-Key": "replay-me"}
    first = client.post("/api/jobs", data={"text": HOUSE_TEXT}, headers=headers)
    second = client.post("/api/jobs", data={"text": HOUSE_TEXT}, headers=headers)
    assert first.status_code == second.status_code == 201
    assert first.json()["job"]["id"] == second.json()["job"]["id"]
    assert second.json()["replayed"] is True


def test_refresh_recovery_finds_the_active_job(client, login):
    login()
    created = client.post("/api/jobs", data={"text": HOUSE_TEXT}, headers={"Idempotency-Key": "active-1"})
    job_id = created.json()["job"]["id"]
    active = client.get("/api/jobs/active").json()
    assert active is not None and active["id"] == job_id
    # Progress is derived from the persisted stage, never a timer.
    assert all(stage["state"] in {"done", "current", "pending", "failed", "skipped"}
               for stage in active["progress"])


def test_failed_progress_rail_uses_persisted_step_states(client, login):
    login()
    created = client.post("/api/jobs", data={"text": HOUSE_TEXT})
    assert created.status_code == 201
    job_id = created.json()["job"]["id"]

    from app.db import session_scope
    from app.models import Job, PipelineStep

    with session_scope() as session:
        job = session.get(Job, job_id)
        rows = session.query(PipelineStep).filter_by(job_id=job_id).order_by(
            PipelineStep.sequence
        ).all()
        assert len(rows) == 11
        for row in rows[:7]:
            row.state = "succeeded"
        rows[7].state = "failed"
        rows[7].error_code = "provider_rejected"
        job.stage = "failed"
        job.status = "failed"
        job.error_code = "provider_error"
        job.error_message = "The provider rejected all scene requests."

    payload = client.get(f"/api/jobs/{job_id}").json()
    assert len(payload["progress"]) == 11
    assert [stage["state"] for stage in payload["progress"]] == [
        "done", "done", "done", "done", "done", "done",
        "done", "failed", "pending", "pending", "pending",
    ]


def test_cancellation_stops_the_job(client, login):
    login()
    created = client.post("/api/jobs", data={"text": HOUSE_TEXT}, headers={"Idempotency-Key": "cancel-1"})
    job_id = created.json()["job"]["id"]
    canceled = client.post(f"/api/jobs/{job_id}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "canceled"
    from app.db import session_scope
    from app.models import Job

    with session_scope() as session:
        assert session.get(Job, job_id).stage == "canceled"
    assert client.get("/api/jobs/active").json() is None


def test_quality_gate_blocks_publication_of_a_broken_film(client, login, tmp_path):
    """A black/corrupt film must never be published as ready."""
    from app.errors import QualityGateFailed
    from app.services.qa.evaluator import evaluate_final
    from app.services.media import ffmpeg as ff

    black = tmp_path / "black.mp4"
    ff.run([
        "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:size=320x180:rate=24:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(black),
    ])
    report = evaluate_final(black, video_id="test", expected_duration_s=3.0)
    assert report.passed is False
    assert report.severity in {"major", "critical"}
    codes = {d.code for d in report.defects}
    # The film-level evaluator reports segment/staticness codes (the clip evaluator
    # uses black_frames/frozen_motion); either way a black film must be blocked.
    assert codes & {"black_segments", "film_static", "film_soft"}
    assert QualityGateFailed  # imported to document the exception the step raises
