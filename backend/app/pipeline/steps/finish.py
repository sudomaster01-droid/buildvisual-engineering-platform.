"""Steps 9–11: assemble, quality-gate, publish an immutable video version."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from ...db import utcnow
from ...errors import QualityGateFailed
from ...models import GeneratedClip, Job, QualityResult, VideoVersion
from ...services.jobs import transition
from ...services.media.assembly import ClipInput, assemble
from ...services.media.ffmpeg import MediaInfo
from ...services.qa.evaluator import evaluate_final, summarise
from ...spec_support import next_version
from ...storage import key_for_poster, key_for_video
from ..context import StepContext

log = logging.getLogger("buildvisual.steps.finish")


def _accepted_clips(ctx: StepContext) -> list[GeneratedClip]:
    generation = (ctx.job.diagnostics or {}).get("generation", {})
    shots = generation.get("shots", {})
    rows = ctx.db.execute(
        select(GeneratedClip).where(
            GeneratedClip.job_id == ctx.job.id,
            GeneratedClip.status == "accepted",
        )
    ).scalars().all()
    # Order by the shot plan, not by insertion time.
    order = {shot_id: index for index, shot_id in enumerate(shots)}
    return sorted(rows, key=lambda row: order.get(row.shot_id, 999))


def assemble_film(ctx: StepContext) -> dict[str, Any]:
    clips = _accepted_clips(ctx)
    if not clips:
        raise QualityGateFailed("No acceptable clips were produced, so nothing could be assembled.")
    generation = (ctx.job.diagnostics or {}).get("generation", {})
    planned = len(generation.get("shots", {}))
    missing = [
        shot_id for shot_id, info in generation.get("shots", {}).items()
        if not info.get("qa_passed")
    ]

    workdir = Path(tempfile.mkdtemp(prefix="bv_final_"))
    try:
        inputs: list[ClipInput] = []
        for clip in clips:
            local = ctx.storage.local_path(clip.storage_key)
            if local is None:
                local = workdir / f"{clip.id}.mp4"
                local.write_bytes(ctx.storage.open_reader(clip.storage_key).read())
            if not local.exists():
                raise QualityGateFailed(
                    f"The clip for one scene is missing from storage ({clip.shot_id})."
                )
            inputs.append(ClipInput(
                shot_id=clip.shot_id, path=local,
                expected_duration_s=clip.duration_s,
            ))

        output = workdir / "film.mp4"
        result = assemble(
            inputs, output,
            width=ctx.settings.video_width,
            height=ctx.settings.video_height,
            fps=ctx.settings.video_fps,
        )
        video_key = key_for_video(ctx.job.project_id, ctx.job.id)
        stored_video = ctx.storage.put_file(video_key, output)
        poster_key = None
        if result.poster_path and result.poster_path.exists():
            poster_key = key_for_poster(ctx.job.project_id, ctx.job.id)
            ctx.storage.put_file(poster_key, result.poster_path)

        ctx.job.diagnostics = {
            **(ctx.job.diagnostics or {}),
            "assembly": {
                **result.config,
                "storage_key": stored_video.key,
                "poster_key": poster_key,
                "checksum": stored_video.checksum,
                "missing_shots": missing,
                "planned_shots": planned,
            },
        }
        ctx.db.flush()
        ctx.note(
            "Video assembled.",
            data={
                "duration_s": round(result.info.duration_s, 1),
                "clips": len(inputs),
                "missing_shots": missing,
                "warnings": result.warnings[:4],
            },
        )
        return {
            "video_key": stored_video.key,
            "poster_key": poster_key,
            "duration_s": result.info.duration_s,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def quality_gate(ctx: StepContext) -> dict[str, Any]:
    assembly = (ctx.job.diagnostics or {}).get("assembly") or {}
    key = assembly.get("storage_key")
    if not key:
        raise QualityGateFailed("The assembled film is missing, so it cannot be released.")
    local = ctx.storage.local_path(key)
    if local is None:
        local = Path(tempfile.mkstemp(suffix=".mp4")[1])
        local.write_bytes(ctx.storage.open_reader(key).read())

    report = evaluate_final(
        local,
        video_id=ctx.job.id,
        expected_duration_s=assembly.get("expected_duration_s"),
        expected_width=ctx.settings.video_width,
        expected_height=ctx.settings.video_height,
        expected_fps=ctx.settings.video_fps,
    )
    row = QualityResult(
        job_id=ctx.job.id, target="final", target_id=ctx.job.id,
        evaluator_version=report.evaluator_version, passed=report.passed,
        severity=report.severity, signals=report.signals,
        defects=[d.as_dict() for d in report.defects], reason_code=report.reason_code,
    )
    ctx.db.add(row)
    ctx.job.diagnostics = {
        **(ctx.job.diagnostics or {}),
        "quality": report.as_dict(),
    }
    ctx.db.flush()

    if not report.passed:
        ctx.note(
            "The video did not pass the quality checks.",
            level="error",
            data={"defects": [d.code for d in report.defects][:6],
                  "reason": report.reason_code},
        )
        raise QualityGateFailed(
            "The assembled video failed the required quality checks: "
            + ", ".join(sorted({d.code for d in report.defects})) ,
            details={"defects": [d.as_dict() for d in report.defects][:8],
                     "retryable": True},
        )
    ctx.note(
        "Video checked and ready for review.",
        data={
            "defects": [d.code for d in report.defects],
            "signals": {
                key_: value for key_, value in report.signals.items()
                if key_ in ("duration_ratio", "sharpness", "black_frame_ratio",
                            "continuity_delta", "luminance_range")
            },
        },
    )
    return {"passed": True, "report": report.as_dict()}


def publish_version(ctx: StepContext) -> dict[str, Any]:
    job = ctx.job
    assembly = (job.diagnostics or {}).get("assembly") or {}
    quality = (job.diagnostics or {}).get("quality") or {}
    generation = (job.diagnostics or {}).get("generation") or {}
    key = assembly.get("storage_key")
    if not key or not quality.get("passed"):
        raise QualityGateFailed(
            "A video version can only be published after assembly and quality checks pass."
        )

    parent = ctx.db.get(VideoVersion, job.parent_video_version_id) if job.parent_video_version_id else None
    version = next_version(ctx.db, VideoVersion, job.project_id)
    provider = generation.get("provider", "local_render")
    external = bool(generation.get("external"))

    row = VideoVersion(
        project_id=job.project_id,
        job_id=job.id,
        parent_id=parent.id if parent else None,
        version=version,
        model_version_id=job.model_version_id or "",
        scene_manifest_id=job.scene_manifest_id or "",
        shot_plan_id=job.shot_plan_id or "",
        provider=provider,
        generation_mode="external_ai" if external else "model_render",
        storage_key=key,
        poster_key=assembly.get("poster_key"),
        duration_s=float(assembly.get("output_duration_s") or 0.0),
        width=int(assembly.get("width") or ctx.settings.video_width),
        height=int(assembly.get("height") or ctx.settings.video_height),
        fps=float(assembly.get("fps") or ctx.settings.video_fps),
        size_bytes=ctx.storage.size(key),
        checksum=assembly.get("checksum", ""),
        state="ready",
        qa_passed=True,
        qa_summary={
            "severity": quality.get("severity"),
            "reason_codes": quality.get("reason_codes"),
            "defects": [d.get("code") for d in quality.get("defects", [])],
            "evaluator_version": quality.get("evaluator_version"),
        },
        cost_cents=int(job.spent_cents or 0),
        cost_basis="configured_rate" if external else "unmetered",
        title=f"{_project_name(ctx)} — v{version}",
    )
    ctx.db.add(row)
    ctx.db.flush()

    project = ctx.db.get(__import__("app.models", fromlist=["Project"]).Project, job.project_id)
    if project is not None:
        project.latest_video_version_id = row.id
        project.status = "ready"
        if project.name in ("Untitled house", "") and job.house_spec_id:
            from ...models import HouseSpec
            from ...spec_support import spec_from_row

            spec_row = ctx.db.get(HouseSpec, job.house_spec_id)
            if spec_row is not None:
                try:
                    project.name = spec_from_row(spec_row).title[:150]
                except Exception:  # pragma: no cover - cosmetic only
                    pass

    job.video_version_id = row.id
    job.finished_at = utcnow()
    ctx.db.flush()
    transition(
        ctx.db, job, "ready_for_review",
        message="Your walkthrough is ready to review.",
        data={"video_version": version, "video_version_id": row.id},
    )
    transition(ctx.db, job, "completed", message="Completed.")
    ctx.note(
        "Saved to the project.",
        data={"video_version": version, "cost_cents": row.cost_cents},
    )
    return {"video_version_id": row.id, "version": version}


def _project_name(ctx: StepContext) -> str:
    from ...models import Project

    project = ctx.db.get(Project, ctx.job.project_id)
    return (project.name if project else "House")[:80]
