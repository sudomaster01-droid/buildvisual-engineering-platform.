"""Job state machine: legal moves, terminal states and recovery semantics."""

from __future__ import annotations

import pytest

from app.errors import Conflict
from app.pipeline import state_machine as sm


def test_happy_path_is_walkable():
    path = [
        sm.CREATED, sm.VALIDATING, sm.INTERPRETING, sm.MODEL_BUILD,
        sm.DESIGN_COMPLETION, sm.SHOT_PLANNING, sm.REFERENCE_RENDER,
        sm.PROMPT_COMPOSITION, sm.PROVIDER_SUBMITTED, sm.PROVIDER_PROCESSING,
        sm.RETRIEVING, sm.ASSEMBLING, sm.QUALITY_CHECK, sm.READY_FOR_REVIEW,
        sm.COMPLETED,
    ]
    for current, target in zip(path, path[1:]):
        assert sm.can_transition(current, target), f"{current} → {target}"


def test_backwards_and_skipped_moves_are_illegal():
    assert not sm.can_transition(sm.ASSEMBLING, sm.MODEL_BUILD)
    assert not sm.can_transition(sm.CREATED, sm.QUALITY_CHECK)
    assert not sm.can_transition(sm.READY_FOR_REVIEW, sm.ASSEMBLING)


def test_terminal_states_are_final():
    for terminal in (sm.COMPLETED, sm.FAILED, sm.CANCELED):
        assert sm.TRANSITIONS[terminal] == set()
        assert sm.is_terminal(terminal)


def test_qa_can_send_a_film_back_for_regeneration():
    """A failed quality check is a legal loop, not a dead end."""
    assert sm.can_transition(sm.QUALITY_CHECK, sm.PROVIDER_SUBMITTED)


def test_every_stage_has_user_facing_copy():
    for stage in sm.TRANSITIONS:
        assert sm.stage_label(stage)
        assert sm.user_status(stage)


def test_assert_transition_raises_conflict():
    with pytest.raises(Conflict):
        sm.assert_transition(sm.COMPLETED, sm.VALIDATING)


def test_job_transition_persists_stage_and_event(client, login, db_session):
    from app.models import Job, Project, User
    from app.services.jobs import transition

    user = db_session.query(User).first()
    project = Project(user_id=user.id, name="State test")
    db_session.add(project)
    db_session.flush()
    job = Job(project_id=project.id, user_id=user.id, kind="generate")
    db_session.add(job)
    db_session.flush()

    transition(db_session, job, sm.VALIDATING)
    assert job.stage == sm.VALIDATING
    assert job.status == "processing"

    with pytest.raises(Conflict):
        transition(db_session, job, sm.COMPLETED)

    events = db_session.query(__import__("app.models", fromlist=["JobEvent"]).JobEvent) \
        .filter_by(job_id=job.id).count()
    assert events == 1


def test_cancel_is_terminal_and_releases_steps(client, login, db_session):
    from app.models import HouseSpec, Job, Project, User
    from app.pipeline.runner import seed_steps
    from app.services.orchestrator import cancel

    user = db_session.query(User).first()
    project = Project(user_id=user.id, name="Cancel test")
    db_session.add(project)
    db_session.flush()
    job = Job(project_id=project.id, user_id=user.id, kind="generate")
    db_session.add(job)
    db_session.flush()
    seed_steps(db_session, job)

    cancel(db_session, job, user.id)
    assert job.stage == sm.CANCELED
    steps = db_session.query(__import__("app.models", fromlist=["PipelineStep"]).PipelineStep) \
        .filter_by(job_id=job.id).all()
    assert all(step.state == "skipped" for step in steps)
    del HouseSpec
