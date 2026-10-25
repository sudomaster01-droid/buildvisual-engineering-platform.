"""Explicit job state machine.

Blueprint p.72: every transition is written transactionally with a timestamp,
invalid jumps are rejected, and terminal states are terminal — a correction
creates a *new* job/version rather than pushing a completed one backwards.

The map below is the single source of truth used by the runner, the API and the
UI (the UI receives stage names, not internal state).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import Conflict

CREATED = "created"
VALIDATING = "validating"
INTERPRETING = "interpreting"
MODEL_BUILD = "model_build"
DESIGN_COMPLETION = "design_completion"
SHOT_PLANNING = "shot_planning"
REFERENCE_RENDER = "reference_render"
PROMPT_COMPOSITION = "prompt_composition"
PROVIDER_SUBMITTED = "provider_submitted"
PROVIDER_PROCESSING = "provider_processing"
RETRIEVING = "retrieving"
ASSEMBLING = "assembling"
QUALITY_CHECK = "quality_check"
READY_FOR_REVIEW = "ready_for_review"
COMPLETED = "completed"
WAITING_FOR_CLARIFICATION = "waiting_for_clarification"
FAILED = "failed"
CANCELED = "canceled"

TERMINAL = {COMPLETED, FAILED, CANCELED}
ACTIVE = {
    CREATED, VALIDATING, INTERPRETING, MODEL_BUILD, DESIGN_COMPLETION,
    SHOT_PLANNING, REFERENCE_RENDER, PROMPT_COMPOSITION, PROVIDER_SUBMITTED,
    PROVIDER_PROCESSING, RETRIEVING, ASSEMBLING, QUALITY_CHECK, READY_FOR_REVIEW,
    WAITING_FOR_CLARIFICATION,
}

#: Allowed forward/terminal moves.
TRANSITIONS: dict[str, set[str]] = {
    CREATED: {VALIDATING, FAILED, CANCELED},
    VALIDATING: {INTERPRETING, WAITING_FOR_CLARIFICATION, FAILED, CANCELED},
    WAITING_FOR_CLARIFICATION: {INTERPRETING, FAILED, CANCELED},
    INTERPRETING: {MODEL_BUILD, WAITING_FOR_CLARIFICATION, FAILED, CANCELED},
    MODEL_BUILD: {DESIGN_COMPLETION, FAILED, CANCELED},
    DESIGN_COMPLETION: {SHOT_PLANNING, FAILED, CANCELED},
    SHOT_PLANNING: {REFERENCE_RENDER, FAILED, CANCELED},
    REFERENCE_RENDER: {PROMPT_COMPOSITION, FAILED, CANCELED},
    PROMPT_COMPOSITION: {PROVIDER_SUBMITTED, FAILED, CANCELED},
    PROVIDER_SUBMITTED: {PROVIDER_PROCESSING, RETRIEVING, FAILED, CANCELED},
    PROVIDER_PROCESSING: {RETRIEVING, PROVIDER_SUBMITTED, FAILED, CANCELED},
    RETRIEVING: {ASSEMBLING, PROVIDER_SUBMITTED, FAILED, CANCELED},
    ASSEMBLING: {QUALITY_CHECK, FAILED, CANCELED},
    QUALITY_CHECK: {READY_FOR_REVIEW, PROVIDER_SUBMITTED, ASSEMBLING, FAILED, CANCELED},
    READY_FOR_REVIEW: {COMPLETED, FAILED, CANCELED},
    # Terminal states have no exits.
    COMPLETED: set(),
    FAILED: set(),
    CANCELED: set(),
}

#: Human-facing progress copy (blueprint p.12 wording, kept plain).
STAGE_LABELS: dict[str, str] = {
    CREATED: "Preparing",
    VALIDATING: "Receiving house information",
    WAITING_FOR_CLARIFICATION: "Waiting for one detail",
    INTERPRETING: "Understanding the design",
    MODEL_BUILD: "Building the 3D model",
    DESIGN_COMPLETION: "Choosing materials and furnishing",
    SHOT_PLANNING: "Planning the walkthrough",
    REFERENCE_RENDER: "Rendering structural references",
    PROMPT_COMPOSITION: "Preparing the scenes",
    PROVIDER_SUBMITTED: "Generating video scenes",
    PROVIDER_PROCESSING: "Generating video scenes",
    RETRIEVING: "Collecting generated scenes",
    ASSEMBLING: "Assembling the video",
    QUALITY_CHECK: "Checking the video",
    READY_FOR_REVIEW: "Ready to review",
    COMPLETED: "Ready to review",
    FAILED: "Could not be completed",
    CANCELED: "Canceled",
}

#: Ordered list used to render the progress rail.
PROGRESS_ORDER = [
    VALIDATING, INTERPRETING, MODEL_BUILD, DESIGN_COMPLETION, SHOT_PLANNING,
    REFERENCE_RENDER, PROMPT_COMPOSITION, PROVIDER_SUBMITTED, ASSEMBLING,
    QUALITY_CHECK, READY_FOR_REVIEW,
]

#: Job status shown to the user, derived from the internal stage.
USER_STATUS: dict[str, str] = {
    CREATED: "queued",
    VALIDATING: "processing",
    INTERPRETING: "processing",
    WAITING_FOR_CLARIFICATION: "needs_input",
    MODEL_BUILD: "processing",
    DESIGN_COMPLETION: "processing",
    SHOT_PLANNING: "processing",
    REFERENCE_RENDER: "processing",
    PROMPT_COMPOSITION: "processing",
    PROVIDER_SUBMITTED: "processing",
    PROVIDER_PROCESSING: "processing",
    RETRIEVING: "processing",
    ASSEMBLING: "processing",
    QUALITY_CHECK: "processing",
    READY_FOR_REVIEW: "ready",
    COMPLETED: "ready",
    FAILED: "failed",
    CANCELED: "canceled",
}


@dataclass
class Transition:
    from_state: str
    to_state: str
    stage: str
    message: str = ""


def can_transition(current: str, target: str) -> bool:
    return target in TRANSITIONS.get(current, set())


def assert_transition(current: str, target: str) -> None:
    if not can_transition(current, target):
        raise Conflict(
            f"Illegal job transition {current} → {target}.",
            details={"from": current, "to": target},
        )


def user_status(stage: str) -> str:
    return USER_STATUS.get(stage, "processing")


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage)


def is_terminal(stage: str) -> bool:
    return stage in TERMINAL
