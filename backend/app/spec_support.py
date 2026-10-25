"""House-spec persistence helpers.

A spec is stored as a validated JSON snapshot plus derived metadata so that:
* a later parser/model upgrade cannot silently change the meaning of an old job
  (blueprint p.34) — old rows keep their own snapshot and schema version;
* the provenance and confidence travel with the geometry that was built from it.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import utcnow
from .models import HouseSpec, Job
from .services.spec.schema import SCHEMA_VERSION, HouseSpec as HouseSpecModel, merge_confidence


def next_version(db: Session, model, project_id: str) -> int:
    current = db.execute(
        select(func.max(model.version)).where(model.project_id == project_id)
    ).scalar()
    return int(current or 0) + 1


def spec_to_row(
    db: Session,
    job: Job,
    spec: HouseSpecModel,
    *,
    note: str = "",
    report: dict[str, Any] | None = None,
) -> HouseSpec:
    """Persist a new immutable spec version for this project."""
    version = next_version(db, HouseSpec, job.project_id)
    parent = db.execute(
        select(HouseSpec)
        .where(HouseSpec.project_id == job.project_id)
        .order_by(HouseSpec.version.desc())
        .limit(1)
    ).scalar_one_or_none()
    payload = spec.model_dump(mode="json")
    row = HouseSpec(
        project_id=job.project_id,
        parent_id=parent.id if parent else None,
        version=version,
        schema_version=SCHEMA_VERSION,
        spec=payload,
        confidence=merge_confidence(spec),
        unresolved=[u.model_dump(mode="json") for u in spec.unresolved],
        provenance={
            **spec.provenance,
            "note": note,
            "report": report or {},
            "created_at": utcnow().isoformat(),
        },
        job_id=job.id,
    )
    db.add(row)
    db.flush()
    return row


def spec_from_row(row: HouseSpec) -> HouseSpecModel:
    return HouseSpecModel.model_validate(row.spec)


def spec_snapshot_hash(row: HouseSpec) -> str:
    """Stable hash of the stored snapshot (used for traceability and dedupe)."""
    import hashlib

    payload = json.dumps(row.spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
