"""Steps 5–7: shot plan, model-derived references, and private prompts."""

from __future__ import annotations

import logging
from typing import Any

from ...models import (
    HouseSpec,
    ModelVersion,
    PromptVersion,
    ReferenceArtifact,
    SceneManifest,
    ShotPlan as ShotPlanRow,
)
from ...services.design.compose import compose_scene, styled_scene_mesh
from ...services.film.motion import sample_camera
from ...services.film.prompts import compose_prompts
from ...services.film.shotplan import plan_film
from ...services.modeling.geometry import build_model
from ...services.modeling.raster import (
    colour_to_png_bytes,
    depth_to_png_bytes,
    edge_to_png_bytes,
    render,
    segmentation_to_png_bytes,
)
from ...spec_support import next_version, spec_from_row
from ...storage import key_for_reference
from ..context import StepContext

log = logging.getLogger("buildvisual.steps.reference")


def load_model_and_scene(ctx: StepContext):
    """Rebuild the deterministic model + scene from their stored versions.

    Rebuilding (rather than pickling meshes into the database) keeps the model
    artifact as the single source of truth and makes a worker restart safe: the
    same spec + builder version reproduce the same geometry hash.
    """
    model_row = ctx.db.get(ModelVersion, ctx.job.model_version_id)
    scene_row = ctx.db.get(SceneManifest, ctx.job.scene_manifest_id)
    if model_row is None or scene_row is None:
        raise RuntimeError("The model or scene manifest is missing for this job.")
    spec_row = ctx.db.get(HouseSpec, model_row.house_spec_id)
    spec = spec_from_row(spec_row)
    model = build_model(spec)
    manifest = scene_row.manifest
    # Recompute the scene from the recorded manifest so materials, sun and
    # placements match exactly what was approved (a restart-safe rebuild).
    scene = compose_scene(
        model,
        attempt=int(manifest.get("attempt", 0)),
        seed_material=manifest.get("seed"),
        palette_override=manifest.get("palette_id"),
        lighting_overrides=(manifest.get("lighting") or {}).get("requested_overrides") or {},
        staging=(manifest.get("staging") or {}),
    )
    return model, scene, scene_row


def plan_shots(ctx: StepContext) -> dict[str, Any]:
    job = ctx.job
    model, scene, scene_row = load_model_and_scene(ctx)
    adjustments = ctx.rectification.get("shot_adjustments") or {}
    plan = plan_film(
        model, scene,
        max_shots=ctx.settings.max_shots,
        clip_seconds=ctx.settings.shot_seconds,
        version=1,
        adjustments=adjustments,
    )
    version = next_version(ctx.db, ShotPlanRow, job.project_id)
    row = ShotPlanRow(
        project_id=job.project_id,
        model_version_id=scene_row.model_version_id,
        scene_manifest_id=scene_row.id,
        version=version,
        planner_version=plan.manifest["planner_version"],
        plan=plan.manifest,
    )
    ctx.db.add(row)
    ctx.db.flush()
    job.shot_plan_id = row.id
    ctx.db.flush()
    ctx.note(
        "Walkthrough planned.",
        data={
            "shots": len(plan.shots),
            "sequence": plan.manifest["sequence"],
            "duration_s": plan.manifest["total_duration_s"],
            "adaptations": plan.manifest["adaptations"],
        },
    )
    return {"shot_plan_id": row.id, "shots": len(plan.shots)}


def render_references(ctx: StepContext) -> dict[str, Any]:
    job = ctx.job
    model, scene, scene_row = load_model_and_scene(ctx)
    plan_row = ctx.db.get(ShotPlanRow, job.shot_plan_id)
    if plan_row is None:
        raise RuntimeError("The shot plan is missing.")
    plan = plan_film(
        model, scene, max_shots=ctx.settings.max_shots,
        clip_seconds=ctx.settings.shot_seconds,
        adjustments=ctx.rectification.get("shot_adjustments") or {},
    )
    styled = styled_scene_mesh(model, scene)
    sun = tuple(scene.manifest["lighting"]["sun"]["direction"])
    width, height = ctx.settings.reference_width, ctx.settings.reference_height

    interior_light = scene.manifest["lighting"]["interior"]
    exterior_light = scene.manifest["lighting"]["exterior"]
    interior_types = {"entrance_walkthrough", "room_walkthrough", "upper_floor"}

    rendered: dict[str, dict[str, str]] = {}
    for shot in plan.shots:
        ctx.heartbeat()
        light = interior_light if shot.type in interior_types else exterior_light
        for role, t in (("first", 0.0), ("last", 1.0)):
            camera = sample_camera(shot, t)
            result = render(
                styled, camera, width, height, sun_direction=sun,
                ambient=float(light["ambient"]),
                exposure=float(light["exposure"]),
            )
            artifacts = {
                "color": colour_to_png_bytes(result),
                "depth": depth_to_png_bytes(result),
                "edge": edge_to_png_bytes(result),
                "segmentation": segmentation_to_png_bytes(result),
            }
            if role == "last":
                # Keyframes: the colour anchor is what a video model can use; the
                # derived maps stay as internal QA evidence.
                artifacts = {"color": artifacts["color"]}
            for kind, payload in artifacts.items():
                key = key_for_reference(job.project_id, plan_row.id, shot.id, f"{role}_{kind}")
                stored = ctx.storage.put_bytes(key, payload)
                ctx.db.add(ReferenceArtifact(
                    shot_plan_id=plan_row.id,
                    shot_id=shot.id,
                    kind=kind,
                    frame_role=role,
                    storage_key=stored.key,
                    width=width,
                    height=height,
                    checksum=stored.checksum,
                ))
            rendered.setdefault(shot.id, {})[role] = (
                f"{plan_row.id}:{shot.id}:{role}_color"
            )
    ctx.db.flush()
    ctx.note(
        "Structural references rendered from the model.",
        data={
            "shots": len(plan.shots),
            "per_shot": ["colour", "depth", "edge", "segmentation"],
            "note": (
                "Reference frames guide the video model's composition. No configured "
                "provider treats them as a hard geometric constraint."
            ),
        },
    )
    return {"rendered": len(rendered)}


def compose_prompts_step(ctx: StepContext) -> dict[str, Any]:
    job = ctx.job
    model, scene, _scene_row = load_model_and_scene(ctx)
    plan = plan_film(
        model, scene, max_shots=ctx.settings.max_shots,
        clip_seconds=ctx.settings.shot_seconds,
        adjustments=ctx.rectification.get("shot_adjustments") or {},
    )
    correction = None
    if job.kind == "rectify":
        correction = {
            "category": job.rectification_category or "presentation",
            "text": job.rectification_text,
            "target": ", ".join(filter(None, [
                (ctx.rectification.get("description") or "")[:80],
            ])) or None,
        }
    prompts = compose_prompts(model, scene, plan, correction=correction)
    provider = ctx.settings.resolved_video_provider
    snapshot = {
        "provider": provider,
        "template_version": prompts[0].template_version if prompts else "",
        "shots": [
            {
                "shot_id": p.shot_id,
                "shot_type": p.shot_type,
                # Prompts stay server-side: only the hash is retained for audit.
                "hash": p.hash,
                "characters": len(p.prompt),
            }
            for p in prompts
        ],
    }
    row = PromptVersion(
        job_id=job.id,
        shot_plan_id=job.shot_plan_id or "",
        template_version=snapshot["template_version"],
        provider=provider,
        prompts={"items": [p.as_dict() for p in prompts], "snapshot": snapshot},
        redacted_prompt_hash="".join(p.hash[:8] for p in prompts)[:64],
    )
    ctx.db.add(row)
    ctx.db.flush()
    ctx.note(
        "Scene instructions prepared.",
        data={
            "template_version": snapshot["template_version"],
            "shots": len(prompts),
            "provider": provider,
        },
    )
    return {"prompt_version_id": row.id, "count": len(prompts)}
