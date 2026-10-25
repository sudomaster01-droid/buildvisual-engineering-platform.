"""Step 8: external (or local) video generation with targeted regeneration.

Blueprint p.58/p.66/p.67:
* persist the provider task *before* polling,
* map status honestly, never invent progress percentages,
* download the output immediately (Veo deletes it after 2 days),
* evaluate each clip and regenerate only the shot that failed, within an
  attempt budget and a spend ceiling,
* never mark anything complete without an artifact on disk.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from ...errors import (
    BudgetExceeded,
    CapabilityUnavailable,
    ProviderError,
    ProviderOutputExpired,
    ProviderRateLimited,
)
from ...models import (
    GeneratedClip,
    Job,
    PromptVersion,
    ProviderTask,
    ReferenceArtifact,
    ShotPlan as ShotPlanRow,
)
from ...services.film.shotplan import plan_film
from ...services.jobs import charge, spend_since
from ...services.media import ffmpeg as ff
from ...services.qa.evaluator import evaluate_clip, summarise
from ...services.video.base import VideoRequest
from ...services.video.factory import build_provider, is_external
from ...storage import key_for_clip
from ..context import StepContext
from .reference import load_model_and_scene

log = logging.getLogger("buildvisual.steps.generate")


@dataclass
class ShotOutcome:
    shot_id: str
    shot_type: str
    clip_id: str | None
    storage_key: str | None
    provider: str
    attempts: int
    qa_passed: bool
    qa_report: dict
    duration_s: float


def generate_clips(ctx: StepContext) -> dict[str, Any]:
    job = ctx.job
    model, scene, _ = load_model_and_scene(ctx)
    plan_row = ctx.db.get(ShotPlanRow, job.shot_plan_id)
    if plan_row is None:
        raise RuntimeError("The shot plan is missing.")
    plan = plan_film(
        model, scene, max_shots=ctx.settings.max_shots,
        clip_seconds=ctx.settings.shot_seconds,
        adjustments=ctx.rectification.get("shot_adjustments") or {},
    )
    prompt_row = ctx.db.execute(
        select(PromptVersion).where(PromptVersion.job_id == job.id)
        .order_by(PromptVersion.created_at.desc()).limit(1)
    ).scalar_one_or_none()
    if prompt_row is None:
        raise RuntimeError("Prompts were not composed for this job.")
    prompts = {item["shot_id"]: item for item in prompt_row.prompts.get("items", [])}

    provider_name = ctx.settings.resolved_video_provider
    if is_external(provider_name) and not _provider_configured(ctx, provider_name):
        raise CapabilityUnavailable(
            f"The configured video provider ({provider_name}) has no credentials."
        )
    ctx.note(
        f"Generating {len(plan.shots)} scenes with "
        + ("the external video service." if is_external(provider_name)
           else "the local model renderer (no AI video key configured)."),
        data={
            "provider": provider_name,
            "external": is_external(provider_name),
            "shots": len(plan.shots),
        },
    )

    provider = build_provider(
        provider_name,
        mesh=styled_mesh_cache(ctx, scene),
        shot_plan=plan,
        sun_direction=tuple(scene.manifest["lighting"]["sun"]["direction"]),
        lighting=scene.manifest["lighting"],
    )
    caps = provider.capabilities()

    outcomes: dict[str, ShotOutcome] = {}
    total_cost = 0
    failed_shots: list[str] = []

    for position, shot in enumerate(plan.shots, start=1):
        ctx.heartbeat()
        ctx.note(
            f"Generating scene {position} of {len(plan.shots)}: {shot.title}.",
            data={"shot_id": shot.id, "position": position, "of": len(plan.shots)},
        )
        prompt = prompts.get(shot.id, {})
        request = _build_request(ctx, shot, prompt, plan_row, caps)
        outcome = _generate_one_shot(ctx, shot, request, provider, provider_name, caps)
        outcomes[shot.id] = outcome
        if not outcome.qa_passed:
            failed_shots.append(shot.id)
        ctx.note(
            f"Scene ready: {shot.title}.",
            data={
                "shot_id": shot.id,
                "shot_type": shot.type,
                "qa_passed": outcome.qa_passed,
                "attempts": outcome.attempts,
            },
            level="info" if outcome.qa_passed else "warning",
        )

    total_cost = _job_spend_cents(ctx)

    if len(failed_shots) == len(outcomes):
        raise ProviderError(
            "None of the scenes passed the quality checks, so no video could be produced.",
            details={"failed_shots": failed_shots},
            retryable=True,
        )

    job.diagnostics = {
        **(job.diagnostics or {}),
        "generation": {
            "provider": provider_name,
            "external": is_external(provider_name),
            "shots": {
                shot_id: {
                    "shot_type": o.shot_type,
                    "clip_id": o.clip_id,
                    "storage_key": o.storage_key,
                    "attempts": o.attempts,
                    "qa_passed": o.qa_passed,
                    "qa": o.qa_report,
                    "duration_s": o.duration_s,
                }
                for shot_id, o in outcomes.items()
            },
            "failed_shots": failed_shots,
        },
    }
    ctx.db.flush()
    return {
        "shots": len(outcomes),
        "failed": failed_shots,
        "cost_cents": total_cost,
    }


def _provider_configured(ctx: StepContext, name: str) -> bool:
    if name == "veo":
        return bool(ctx.settings.gemini_api_key)
    if name == "runway":
        return bool(ctx.settings.runway_api_key)
    return True


def _build_request(ctx: StepContext, shot, prompt: dict, plan_row, caps) -> VideoRequest:
    """Build a request the provider has declared it can accept.

    Reference frames are attached only when the adapter advertises support, so an
    unsupported capability degrades to prompt-only guidance instead of failing the
    shot (and never silently claims structural control it does not have).
    """
    provider = ctx.settings.resolved_video_provider
    first = _reference_bytes(ctx, plan_row.id, shot.id, "first", "color") \
        if caps.supports_first_frame else None
    last = _reference_bytes(ctx, plan_row.id, shot.id, "last", "color") \
        if caps.supports_last_frame else None

    prompt_text = prompt.get("prompt", "")
    if provider == "runway":
        prompt_text = prompt.get("compact") or prompt_text[:900]

    duration = int(round(shot.duration_s))
    allowed = sorted(caps.durations_s)
    if allowed and duration not in allowed:
        duration = min(allowed, key=lambda value: abs(value - duration))

    aspect = (
        "1280:720" if provider == "runway" and "1280:720" in caps.aspect_ratios
        else ctx.settings.video_aspect_ratio
    )
    if caps.aspect_ratios and aspect not in caps.aspect_ratios:
        aspect = caps.aspect_ratios[0]
    resolution = "720p"
    if caps.resolutions and resolution not in caps.resolutions:
        resolution = caps.resolutions[0]

    return VideoRequest(
        shot_id=shot.id,
        prompt=prompt_text[: caps.prompt_max_chars],
        negative_prompt=prompt.get("negative", ""),
        duration_s=duration,
        aspect_ratio=aspect,
        resolution=resolution,
        first_frame=first,
        last_frame=last,
        seed=abs(hash((plan_row.id, shot.id))) % (2**31),
    )


def _reference_bytes(ctx: StepContext, shot_plan_id: str, shot_id: str,
                     role: str, kind: str) -> bytes | None:
    row = ctx.db.execute(
        select(ReferenceArtifact).where(
            ReferenceArtifact.shot_plan_id == shot_plan_id,
            ReferenceArtifact.shot_id == shot_id,
            ReferenceArtifact.frame_role == role,
            ReferenceArtifact.kind == kind,
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    return ctx.storage.open_reader(row.storage_key).read()


def _job_spend_cents(ctx: StepContext) -> int:
    """Spend recorded in the ledger for this job (compute and provider kept apart)."""
    from ...models import UsageLedger

    rows = ctx.db.execute(
        select(UsageLedger.cost_cents).where(UsageLedger.job_id == ctx.job.id)
    ).scalars().all()
    return int(sum(rows or []))


def _generate_one_shot(ctx, shot, request, provider, provider_name, caps) -> ShotOutcome:
    job = ctx.job
    settings = ctx.settings
    max_attempts = max(1, settings.max_provider_attempts_per_shot)
    problems = provider.validate(request)
    if problems:
        raise ProviderError(
            "The shot request is incompatible with the provider: " + "; ".join(problems),
            details={"shot_id": shot.id, "provider": provider_name},
            retryable=False,
        )

    estimate_cents, basis, estimated = provider.estimate_cost_cents(request)
    _enforce_budget(ctx, estimate_cents)

    last_report: dict = {}
    clip_row: GeneratedClip | None = None
    attempts = 0

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        ctx.heartbeat()
        idempotency_key = hashlib.sha256(
            f"{job.id}:{shot.id}:{attempt}:{request.prompt[:80]}".encode()
        ).hexdigest()
        task = ProviderTask(
            job_id=job.id, shot_id=shot.id, provider=provider_name,
            model=caps.model, attempt=attempt,
            request_hash=hashlib.sha256(
                (request.prompt + str(request.duration_s)).encode()
            ).hexdigest(),
            request_snapshot={
                "duration_s": request.duration_s,
                "aspect_ratio": request.aspect_ratio,
                "has_first_frame": request.first_frame is not None,
                "has_last_frame": request.last_frame is not None,
                "prompt_chars": len(request.prompt),
            },
            idempotency_key=idempotency_key,
            status="submitted",
            submitted_at=_utcnow(),
        )
        ctx.db.add(task)
        ctx.db.flush()

        started = time.perf_counter()
        try:
            submitted = provider.submit(request, idempotency_key=idempotency_key)
            task.external_id = submitted.task_id
            task.status = "processing"
            ctx.db.flush()
            _set_stage(ctx, "provider_processing" if is_external(provider_name) else "provider_submitted")

            status = _await_provider(ctx, provider, submitted.task_id, shot)
            if status.state != "succeeded":
                raise ProviderError(
                    status.message or "The provider reported a failure.",
                    details={"provider_state": status.provider_state, "shot_id": shot.id},
                    retryable=status.retryable,
                )
            _set_stage(ctx, "retrieving")
            output = provider.fetch(submitted.task_id, status)
            task.status = "succeeded"
            task.completed_at = _utcnow()
            task.latency_ms = int((time.perf_counter() - started) * 1000)
        except ProviderRateLimited as exc:
            task.status = "failed"
            task.error_code = exc.code
            task.error_message = exc.user_message
            ctx.db.flush()
            _charge_attempt(ctx, shot, provider_name, caps.model, estimate_cents, basis, estimated, attempt, failed=True)
            if attempt == max_attempts:
                raise
            _backoff(attempt)
            continue
        except ProviderError as exc:
            task.status = "failed"
            task.error_code = exc.code
            task.error_message = exc.user_message[:500]
            task.completed_at = _utcnow()
            ctx.db.flush()
            _charge_attempt(ctx, shot, provider_name, caps.model, estimate_cents, basis, estimated, attempt, failed=True)
            last_report = {"error": exc.user_message}
            if not exc.retryable or attempt == max_attempts:
                log.warning("shot %s failed after %s attempt(s): %s", shot.id, attempt, exc.user_message)
                return ShotOutcome(
                    shot_id=shot.id, shot_type=shot.type, clip_id=None,
                    storage_key=None, provider=provider_name, attempts=attempt,
                    qa_passed=False, qa_report=last_report, duration_s=0.0,
                )
            _backoff(attempt)
            continue

        _charge_attempt(ctx, shot, provider_name, caps.model, estimate_cents, basis, estimated, attempt, failed=False)

        clip_bytes = output.data
        key = key_for_clip(job.project_id, job.id, shot.id, attempt)
        stored = ctx.storage.put_bytes(key, clip_bytes)
        local = ctx.storage.local_path(stored.key)
        if local is None:  # S3 backend: QA needs a local file
            from tempfile import NamedTemporaryFile

            handle = NamedTemporaryFile(suffix=".mp4", delete=False)
            handle.write(clip_bytes)
            handle.close()
            local = Path(handle.name)

        try:
            info = ff.probe(local)
        except ff.MediaError as exc:
            task.status = "failed"
            task.error_message = "provider output was not a readable video"
            ctx.db.flush()
            last_report = {"error": str(exc)[:200]}
            if attempt == max_attempts:
                return ShotOutcome(
                    shot_id=shot.id, shot_type=shot.type, clip_id=None, storage_key=None,
                    provider=provider_name, attempts=attempt, qa_passed=False,
                    qa_report=last_report, duration_s=0.0,
                )
            continue

        ref_first = _reference_path(ctx, plan_row_id(ctx), shot.id, "first")
        report = evaluate_clip(
            local, clip_id=f"{shot.id}#a{attempt}",
            expected_duration_s=shot.duration_s,
            reference_first=ref_first,
        )
        last_report = report.as_dict()
        clip_row = GeneratedClip(
            job_id=job.id, shot_id=shot.id, provider_task_id=task.id,
            provider=provider_name, storage_key=stored.key,
            status="accepted" if report.passed else "rejected",
            duration_s=info.duration_s, width=info.width, height=info.height,
            fps=info.fps, checksum=stored.checksum, size_bytes=stored.size,
        )
        ctx.db.add(clip_row)
        ctx.db.flush()

        if report.passed:
            return ShotOutcome(
                shot_id=shot.id, shot_type=shot.type, clip_id=clip_row.id,
                storage_key=stored.key, provider=provider_name, attempts=attempt,
                qa_passed=True, qa_report=last_report, duration_s=info.duration_s,
            )
        log.warning("clip for %s failed QA (attempt %s): %s", shot.id, attempt, report.reason_code)
        if attempt < max_attempts:
            _backoff(attempt)

    return ShotOutcome(
        shot_id=shot.id, shot_type=shot.type,
        clip_id=clip_row.id if clip_row else None,
        storage_key=clip_row.storage_key if clip_row else None,
        provider=provider_name, attempts=attempts, qa_passed=False,
        qa_report=last_report, duration_s=clip_row.duration_s if clip_row else 0.0,
    )


def plan_row_id(ctx: StepContext) -> str:
    return ctx.job.shot_plan_id or ""


def _reference_path(ctx: StepContext, shot_plan_id: str, shot_id: str, role: str) -> Path | None:
    row = ctx.db.execute(
        select(ReferenceArtifact).where(
            ReferenceArtifact.shot_plan_id == shot_plan_id,
            ReferenceArtifact.shot_id == shot_id,
            ReferenceArtifact.frame_role == role,
            ReferenceArtifact.kind == "color",
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    return ctx.storage.local_path(row.storage_key)


def styled_mesh_cache(ctx: StepContext, scene):
    from ...services.design.compose import styled_scene_mesh

    model, _scene, _row = load_model_and_scene(ctx)
    return styled_scene_mesh(model, scene)


def _await_provider(ctx: StepContext, provider, task_id: str, shot):
    settings = ctx.settings
    deadline = time.time() + min(
        settings.step_timeout_seconds, settings.job_max_runtime_seconds
    )
    interval = max(2, settings.provider_poll_interval_seconds)
    while time.time() < deadline:
        ctx.heartbeat()
        status = provider.poll(task_id)
        if status.state in ("succeeded", "failed", "expired"):
            return status
        time.sleep(interval if is_external(provider.name) else 0.0)
    raise ProviderError(
        f"The video service did not finish this shot within "
        f"{settings.step_timeout_seconds}s.",
        details={"shot_id": shot.id, "task_id": task_id},
        retryable=True,
    )


def _set_stage(ctx: StepContext, stage: str) -> None:
    from ...pipeline import state_machine as sm
    from ...services.jobs import transition

    if ctx.job.stage != stage and sm.can_transition(ctx.job.stage, stage):
        transition(ctx.db, ctx.job, stage)
        ctx.db.commit()


def _enforce_budget(ctx: StepContext, estimate_cents: int) -> None:
    settings = ctx.settings
    spent = ctx.job.spent_cents or 0
    if settings.max_job_spend_cents and spent + estimate_cents > settings.max_job_spend_cents:
        raise BudgetExceeded(
            "This job would exceed the configured per-job spending limit.",
            details={"spent_cents": spent, "estimate_cents": estimate_cents,
                     "limit_cents": settings.max_job_spend_cents},
        )
    if settings.daily_spend_ceiling_cents:
        from datetime import timedelta

        from ...db import utcnow

        today = spend_since(ctx.db, utcnow() - timedelta(days=1))
        if today + estimate_cents > settings.daily_spend_ceiling_cents:
            raise BudgetExceeded(
                "The daily spending ceiling has been reached. Try again later or raise "
                "DAILY_SPEND_CEILING_CENTS.",
                details={"daily_spent_cents": today,
                         "limit_cents": settings.daily_spend_ceiling_cents},
            )


def _charge_attempt(ctx, shot, provider_name, model, cents, basis, estimated, attempt, *, failed):
    charge(
        ctx.db, ctx.job, provider=provider_name,
        operation=f"generate:{shot.type}" + (":retry" if failed else ""),
        cost_cents=cents, units=float(shot.duration_s), unit_name="second",
        basis=basis, estimated=estimated,
        detail={"shot_id": shot.id, "attempt": attempt, "model": model, "failed": failed},
    )


def _backoff(attempt: int) -> None:
    time.sleep(min(8.0, 0.5 * (2 ** attempt)))


def _utcnow():
    from ...db import utcnow

    return utcnow()
