"""Steps 3–4: build and lock the model, then complete the design."""

from __future__ import annotations

import json
import logging
from typing import Any

from ...models import HouseSpec, ModelVersion, SceneManifest
from ...services.design.compose import compose_scene
from ...services.jobs import record_event
from ...services.modeling.geometry import BUILDER_VERSION, build_model, model_to_glb_bytes
from ...services.modeling.validate import assert_valid
from ...services.spec.schema import HouseSpec as HouseSpecModel
from ...spec_support import next_version, spec_from_row
from ...storage import key_for_model
from ..context import StepContext

log = logging.getLogger("buildvisual.steps.model")


def build_locked_model(ctx: StepContext) -> dict[str, Any]:
    job = ctx.job
    if job.kind == "rectify" and job.model_version_id and not ctx.rectification.get("requires_new_model"):
        existing = ctx.db.get(ModelVersion, job.model_version_id)
        if existing is not None:
            ctx.note(
                "Keeping the approved model version.",
                data={"model_version": existing.version},
            )
            return {"model_version_id": existing.id, "reused": True}

    if not job.house_spec_id:
        raise RuntimeError("interpret_house must run before build_locked_model")
    row = ctx.db.get(HouseSpec, job.house_spec_id)
    if row is None:
        raise RuntimeError("The house specification row disappeared.")
    spec = spec_from_row(row)

    report = assert_valid(spec)  # raises GeometryError on critical findings
    model = build_model(spec)

    glb = model_to_glb_bytes(model)
    version = next_version(ctx.db, ModelVersion, job.project_id)
    key = key_for_model(job.project_id, f"{job.project_id}_v{version}")
    stored = ctx.storage.put_bytes(key, glb)
    manifest = model.manifest()
    manifest["artifact"] = {"key": stored.key, "bytes": stored.size, "sha256": stored.checksum}

    parent = ctx.db.get(ModelVersion, job.model_version_id) if job.model_version_id else None
    row_mv = ModelVersion(
        project_id=job.project_id,
        house_spec_id=row.id,
        parent_id=parent.id if parent else None,
        version=version,
        builder_version=BUILDER_VERSION,
        geometry_hash=model.geometry_hash,
        artifact_key=stored.key,
        manifest=manifest,
        metrics=model.metrics,
        warnings=model.warnings + [f.as_dict() for f in report.findings if f.severity != "info"],
    )
    ctx.db.add(row_mv)
    ctx.db.flush()
    job.model_version_id = row_mv.id
    ctx.db.flush()

    ctx.note(
        "3D model built and locked.",
        data={
            "model_version": version,
            "geometry_hash": model.geometry_hash[:16],
            "triangles": model.metrics.get("triangles"),
            "floor_area_m2": model.metrics.get("floor_area_m2"),
            "warnings": len(model.warnings),
        },
    )
    return {
        "model_version_id": row_mv.id,
        "geometry_hash": model.geometry_hash,
        "manifest": manifest,
    }


def complete_design(ctx: StepContext) -> dict[str, Any]:
    job = ctx.job
    model_row = ctx.db.get(ModelVersion, job.model_version_id)
    if model_row is None:
        raise RuntimeError("The model version is missing.")
    spec_row = ctx.db.get(HouseSpec, model_row.house_spec_id)
    spec = spec_from_row(spec_row)
    model = build_model(spec)  # deterministic rebuild from the locked spec

    overrides = ctx.rectification.get("design_overrides") or {}
    attempt = int(ctx.rectification.get("attempt") or 0)
    if job.kind == "rectify" and not overrides.get("palette_override") and not overrides.get("staging"):
        attempt += 1  # a bare "make it different" still produces a different palette

    scene = compose_scene(
        model,
        attempt=attempt,
        palette_override=overrides.get("palette_override"),
        lighting_overrides=overrides.get("lighting_overrides") or {},
        staging=overrides.get("staging") or {},
    )
    version = next_version(ctx.db, SceneManifest, job.project_id)
    row = SceneManifest(
        project_id=job.project_id,
        model_version_id=model_row.id,
        version=version,
        seed=int(scene.manifest.get("attempt", 0)),
        palette_id=scene.manifest["palette_id"],
        manifest=scene.manifest,
    )
    ctx.db.add(row)
    ctx.db.flush()
    job.scene_manifest_id = row.id
    ctx.db.flush()

    ctx.note(
        "Design completed automatically.",
        data={
            "palette": scene.manifest["palette_id"],
            "furniture": len(scene.manifest["objects"]),
            "trees": scene.manifest["landscape"]["counts"]["trees"],
            "skipped": len(scene.manifest["skipped_items"]),
            "model_unchanged": not scene.manifest["structural_geometry_modified"],
        },
    )
    return {
        "scene_manifest_id": row.id,
        "palette": scene.manifest["palette_id"],
        "objects": len(scene.manifest["objects"]),
    }
