"""Review & Rectify: classification, routing and immutable versioning."""

from __future__ import annotations

import pytest

HOUSE_TEXT = (
    "Three bedroom bungalow with a flat roof, open plan living and dining "
    "and a covered veranda"
)


# ---------------------------------------------------------------------------
# Classification (pure, fast)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("make the lighting warmer", "presentation"),
        ("use lighter materials and a calmer palette", "presentation"),
        ("change the living room furniture", "scene_composition"),
        ("show the entrance more clearly", "camera"),
        ("add another bedroom", "structural"),
        ("change the roof to a gable roof", "structural"),
        ("hmm can you improve it", "clarification"),
    ],
)
def test_change_requests_are_classified(text, expected):
    from app.services.interpret.rules import build_spec_from_requirements
    from app.services.rectify.classify import classify_change

    spec, _ = build_spec_from_requirements(HOUSE_TEXT, seed=1)
    intent = classify_change(text, spec)
    assert intent.category == expected, (text, intent.category, intent.matched_terms)
    assert 0.0 <= intent.confidence <= 1.0


def test_only_structural_changes_require_a_new_model():
    from app.services.interpret.rules import build_spec_from_requirements
    from app.services.rectify.apply import build_plan
    from app.services.rectify.classify import classify_change

    spec, _ = build_spec_from_requirements(HOUSE_TEXT, seed=1)
    for text in ("make the lighting warmer", "change the living room furniture",
                 "show the entrance more clearly"):
        plan = build_plan(classify_change(text, spec), spec)
        assert plan.requires_new_model is False, text
        assert plan.structural_spec is None

    structural = build_plan(classify_change("add another bedroom", spec), spec)
    assert structural.requires_new_model is True
    assert structural.structural_spec is not None
    assert len(structural.structural_spec.rooms) > len(spec.rooms)


def test_ambiguous_request_asks_one_question():
    from app.errors import ClarificationRequired
    from app.services.interpret.rules import build_spec_from_requirements
    from app.services.rectify.apply import build_plan
    from app.services.rectify.classify import classify_change

    spec, _ = build_spec_from_requirements(HOUSE_TEXT, seed=1)
    intent = classify_change("hmm can you improve it", spec)
    with pytest.raises(ClarificationRequired):
        build_plan(intent, spec)


# ---------------------------------------------------------------------------
# Versioning against a real generated film
# ---------------------------------------------------------------------------


def test_corrections_create_immutable_versions(client, login):
    """Aesthetic then structural correction on one real film.

    Aesthetic  → new video version, same locked model.
    Structural → new house spec + new model version + new video version.
    In both cases every earlier version stays deliverable.
    """
    from app.db import session_scope
    from app.models import HouseSpec, ModelVersion, VideoVersion
    from app.pipeline.runner import run_job_to_completion
    from app.services import orchestrator

    login()

    # --- the original film ------------------------------------------------
    with session_scope() as session:
        user_id = session.query(__import__("app.models", fromlist=["User"]).User).first().id
        job = orchestrator.create_generation_job(
            session, user_id=user_id, project_id=None, project_name="Rectify test",
            text=HOUSE_TEXT, drawing=None, audio=None,
        )
        job_id = job.id
    first = run_job_to_completion(job_id, worker_id="pytest-rectify")
    assert first.stage == "completed", (first.stage, first.error_message)

    with session_scope() as session:
        origin = session.query(VideoVersion).order_by(VideoVersion.version).first()
        models_at_start = session.query(ModelVersion).count()
        specs_at_start = session.query(HouseSpec).count()
        origin_id = origin.id

    # --- aesthetic correction --------------------------------------------
    with session_scope() as session:
        job, summary = orchestrator.create_rectification_job(
            session, user_id=user_id, video_version_id=origin_id,
            text="make the lighting warmer",
        )
        aesthetic_job = job.id
        assert summary["category"] == "presentation"
        assert summary["requires_new_model"] is False
    aesthetic = run_job_to_completion(aesthetic_job, worker_id="pytest-rectify")
    assert aesthetic.stage == "completed", (aesthetic.stage, aesthetic.error_message)

    with session_scope() as session:
        versions = session.query(VideoVersion).order_by(VideoVersion.version).all()
        assert len(versions) == 2
        old, new = versions
        assert old.state == "ready" and old.qa_passed      # previous kept intact
        assert new.version == 2 and new.state == "ready" and new.qa_passed
        assert new.parent_id == old.id
        assert new.model_version_id == old.model_version_id   # geometry untouched
        assert new.scene_manifest_id != old.scene_manifest_id  # presentation rebuilt
        assert session.query(ModelVersion).count() == models_at_start
        latest_id = new.id

    # --- structural correction -------------------------------------------
    with session_scope() as session:
        job, summary = orchestrator.create_rectification_job(
            session, user_id=user_id, video_version_id=latest_id,
            text="add another bedroom",
        )
        structural_job = job.id
        assert summary["category"] == "structural"
        assert summary["requires_new_model"] is True
    structural = run_job_to_completion(structural_job, worker_id="pytest-rectify")
    assert structural.stage == "completed", (structural.stage, structural.error_message)

    with session_scope() as session:
        assert session.query(ModelVersion).count() == models_at_start + 1
        assert session.query(HouseSpec).count() == specs_at_start + 1
        versions = session.query(VideoVersion).order_by(VideoVersion.version).all()
        assert len(versions) == 3
        assert versions[-1].model_version_id != versions[-2].model_version_id
        assert versions[-1].parent_id == versions[-2].id
        assert all(version.state == "ready" for version in versions)

        # The structural spec really did gain rooms.
        specs = session.query(HouseSpec).order_by(HouseSpec.version).all()
        assert len(specs[-1].spec["rooms"]) > len(specs[0].spec["rooms"])

    # --- and the history API shows every version --------------------------
    history = client.get(f"/api/videos/{origin_id}").json()
    assert len(history["history"]) == 3


def test_correction_on_an_unknown_video_is_not_found(client, login):
    login()
    response = client.post("/api/videos/does-not-exist/rectify", json={"text": "make it warmer"})
    assert response.status_code == 404


def test_rectify_endpoint_requires_authentication(client):
    response = client.post("/api/videos/anything/rectify", json={"text": "make it warmer"})
    assert response.status_code == 401
