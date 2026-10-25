"""Steps 1–2: validate the request and interpret the house."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select

from ...db import utcnow
from ...errors import (
    BuildVisualError,
    CapabilityUnavailable,
    InterpretationError,
    MissingInput,
    ValidationError,
)
from ...models import Job, SourceArtifact, Transcript, VideoVersion
from ...services.intake.validators import pdf_to_images
from ...services.interpret.drawing import interpret_drawing
from ...services.interpret.llm import build_llm_client, llm_available
from ...services.interpret.rules import build_spec_from_requirements
from ...services.jobs import charge, record_event
from ...services.spec.schema import HouseSpec
from ...services.speech.transcribe import transcribe, transcription_available
from ...spec_support import spec_from_row, spec_to_row
from ..context import StepContext

log = logging.getLogger("buildvisual.steps.intake")


def validate_input(ctx: StepContext) -> dict[str, Any]:
    job = ctx.job
    if job.kind == "rectify":
        return _validate_rectification(ctx)

    artifacts = ctx.db.execute(
        select(SourceArtifact).where(SourceArtifact.id.in_(job.source_artifact_ids or []))
    ).scalars().all()
    valid_artifacts = [a for a in artifacts if a.validation_status == "valid"]
    rejected = [a for a in artifacts if a.validation_status != "valid"]
    if rejected:
        raise ValidationError(
            "One of the uploaded files failed validation and cannot be processed.",
            details={"rejected": [a.original_filename for a in rejected]},
        )

    drawing = next((a for a in valid_artifacts if a.kind == "drawing"), None)
    audio = next((a for a in valid_artifacts if a.kind == "audio"), None)
    text = (job.text_requirements or "").strip()

    transcript_id = job.transcript_id
    if audio is not None and not transcript_id:
        if not transcription_available():
            raise CapabilityUnavailable(
                "Local voice transcription is not ready on this deployment. "
                "Type the requirements instead."
            )
        payload_keys = audio.validation_report or {}
        _ = payload_keys
        data = ctx.storage.open_reader(audio.storage_key).read()
        result = transcribe(data, audio.original_filename or "audio.webm", language=job.locale)
        transcript = Transcript(
            project_id=job.project_id,
            source_artifact_id=audio.id,
            text=result.text,
            language=result.language,
            provider=result.provider,
            model=result.model,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
        )
        ctx.db.add(transcript)
        ctx.db.flush()
        transcript_id = transcript.id
        job.transcript_id = transcript_id
        text = result.text
        ctx.note("Recording transcribed.", data={"confidence": result.confidence})
        charge(
            ctx.db, job, provider="speech", operation="transcribe",
            cost_cents=0, units=result.duration_seconds or 0.0, unit_name="second",
            basis="unmetered", estimated=True,
            detail={"model": result.model, "characters": len(result.text)},
        )

    if not drawing and not text:
        raise MissingInput()

    if text:
        from ...config import get_settings

        limit = get_settings().max_text_chars
        if len(text) > limit:
            raise ValidationError(
                f"The description is {len(text)} characters; the limit is {limit}."
            )
        job.text_requirements = text
    job.input_mode = "drawing" if drawing and not text else ("voice" if transcript_id else "text")
    ctx.db.flush()
    ctx.note(
        "House information received.",
        data={
            "has_drawing": drawing is not None,
            "has_text": bool(text),
            "artifacts": len(valid_artifacts),
        },
    )
    return {"has_drawing": drawing is not None, "has_text": bool(text)}


def _validate_rectification(ctx: StepContext) -> dict[str, Any]:
    job = ctx.job
    if not job.parent_video_version_id:
        raise ValidationError("A correction needs the video version it applies to.")
    parent = ctx.db.get(VideoVersion, job.parent_video_version_id)
    if parent is None or parent.project_id != job.project_id:
        raise ValidationError("The video this correction refers to could not be found.")
    if parent.state != "ready":
        raise ValidationError("The previous video is not available, so it cannot be corrected.")
    if not (job.rectification_text or "").strip():
        raise ValidationError("Describe the change you want.")
    ctx.note("Correction request received.")
    return {"parent_video_version_id": parent.id, "parent_model_version_id": parent.model_version_id}


def interpret_house(ctx: StepContext) -> dict[str, Any]:
    job = ctx.job
    rect = ctx.rectification

    # --- reuse the approved model for non-structural corrections ----------
    if job.kind == "rectify" and not rect.get("requires_new_model"):
        parent_spec_id = rect.get("parent_house_spec_id")
        parent_model_id = rect.get("parent_model_version_id")
        if not parent_spec_id or not parent_model_id:
            raise ValidationError("The previous model for this project could not be located.")
        job.house_spec_id = parent_spec_id
        job.model_version_id = parent_model_id
        ctx.db.flush()
        ctx.note(
            "Keeping the approved house model; only the presentation is rebuilt.",
            data={"model_version_id": parent_model_id},
        )
        return {"reused_model": True}

    if job.kind == "rectify" and rect.get("structural_spec"):
        # The plan is persisted as JSON on the job; re-validate it into the typed
        # schema so a corrupted record can never reach the geometry builder.
        spec = HouseSpec.model_validate(rect["structural_spec"])
        row = spec_to_row(
            ctx.db, job, spec, note="structural correction",
            report={"source": "rectification", "patch": rect.get("spec_patch") or {}},
        )
        job.house_spec_id = row.id
        ctx.db.flush()
        ctx.note(
            "Structural change applied to the house definition; a new model version will be built.",
            data={"house_spec_version": row.version},
        )
        return {"house_spec_id": row.id, "structural": True}

    # --- fresh interpretation --------------------------------------------
    drawing = None
    if job.source_artifact_ids:
        artifacts = ctx.db.execute(
            select(SourceArtifact).where(SourceArtifact.id.in_(job.source_artifact_ids))
        ).scalars().all()
        drawing = next((a for a in artifacts if a.kind == "drawing"), None)

    if drawing is not None:
        spec, report = _interpret_drawing(ctx, drawing)
        source = "drawing"
    elif (job.text_requirements or "").strip():
        spec, parsed = build_spec_from_requirements(
            job.text_requirements, seed=len(job.text_requirements)
        )
        report = {
            "parser": "rules:v1",
            "warnings": parsed.warnings,
            "explicit_facts": parsed.explicit,
            "rooms": len(spec.rooms),
            "levels": len(spec.levels),
            "note": (
                "No drawing was supplied. The plan is an indicative arrangement "
                "generated from the stated requirements."
            ),
        }
        if llm_available():
            report["llm_available"] = True
        source = "requirements"
    else:
        raise MissingInput()

    row = spec_to_row(ctx.db, job, spec, note=f"interpreted from {source}", report=report)
    job.house_spec_id = row.id
    job.diagnostics = {
        **(job.diagnostics or {}),
        "interpretation": {
            "source": source,
            "report": report,
            "confidence": row.confidence,
        },
    }
    ctx.db.flush()
    ctx.note(
        "Design understood.",
        data={
            "rooms": len(spec.rooms),
            "levels": len(spec.levels),
            "source": source,
        },
    )
    return {"house_spec_id": row.id, "source": source}


def _interpret_drawing(ctx: StepContext, artifact: SourceArtifact):
    if not llm_available():
        raise CapabilityUnavailable(
            "Reading a drawing needs a vision model, and none is configured on this "
            "deployment. Type or speak the house requirements instead, or set "
            "LLM_PROVIDER and LLM_API_KEY.",
            details={"capability": "drawing_interpretation"},
        )
    data = ctx.storage.open_reader(artifact.storage_key).read()
    if artifact.mime == "application/pdf":
        images = pdf_to_images(data)
    else:
        images = [(data, artifact.mime)]
    client = build_llm_client()
    spec, report = interpret_drawing(
        images, client,
        requirements_text=ctx.job.text_requirements,
        locale=ctx.job.locale,
    )
    usage = report.get("usage") or {}
    charge(
        ctx.db, ctx.job, provider=report.get("provider", "llm"),
        operation="interpret_drawing", cost_cents=0,
        units=float(usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
        unit_name="token", basis="unmetered", estimated=True,
        detail={"model": report.get("model"), "tokens": usage},
    )
    return spec, report
