"""Applying a classified change without breaking the model-first contract.

Blueprint p.43/p.68/p.76. The routing table below is the whole policy:

    presentation      → new scene manifest;           model version unchanged
    scene_composition → new scene manifest;           model version unchanged
    camera            → new shot plan + references;   model version unchanged
    structural        → new house spec → NEW model version → new references

Only structural changes may touch geometry, and they always go through
re-validation, so a correction can never silently deform the locked model.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

from ...errors import ClarificationRequired, ValidationError
from ..spec.schema import HouseSpec, RoofType, RoomType, SourceType
from .classify import ChangeIntent

log = logging.getLogger("buildvisual.rectify.apply")

#: Palettes reachable from ordinary language.
_PALETTE_HINTS = {
    "warm": "warm_natural",
    "cosy": "warm_natural",
    "cozy": "warm_natural",
    "natural": "warm_natural",
    "timber": "warm_natural",
    "oak": "warm_natural",
    "cool": "cool_modern",
    "modern": "cool_modern",
    "contemporary": "cool_modern",
    "minimal": "cool_modern",
    "grey": "cool_modern",
    "gray": "cool_modern",
    "bright": "crisp_estate",
    "crisp": "crisp_estate",
    "classic": "crisp_estate",
    "estate": "crisp_estate",
    "mediterranean": "mediterranean",
    "terracotta": "mediterranean",
    "spanish": "mediterranean",
    "olive": "mediterranean",
}


@dataclass
class RectificationPlan:
    """Everything the pipeline needs to produce the corrected version."""

    category: str
    description: str
    design_overrides: dict[str, Any]
    shot_adjustments: dict[str, Any]
    spec_patch: dict[str, Any]
    requires_new_model: bool
    structural_spec: HouseSpec | None = None
    affected_shots: list[str] | None = None
    notes: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "design_overrides": self.design_overrides,
            "shot_adjustments": self.shot_adjustments,
            "spec_patch": self.spec_patch,
            "requires_new_model": self.requires_new_model,
            "affected_shots": self.affected_shots or [],
            "notes": self.notes or [],
        }


def build_plan(intent: ChangeIntent, spec: HouseSpec) -> RectificationPlan:
    category = intent.category
    if category == "clarification":
        raise ClarificationRequired(
            intent.question or "This change needs one clarification before it can run.",
            details={"question": intent.question, "category": "clarification"},
        )
    if category == "structural":
        new_spec, patch, notes = apply_structural(intent, spec)
        return RectificationPlan(
            category="structural",
            description=f"Structural change: {intent.text}",
            design_overrides={},
            shot_adjustments={},
            spec_patch=patch,
            requires_new_model=True,
            structural_spec=new_spec,
            notes=notes,
        )
    if category == "camera":
        adjustments = camera_adjustments(intent)
        return RectificationPlan(
            category="camera",
            description=f"Camera change: {intent.text}",
            design_overrides={},
            shot_adjustments=adjustments,
            spec_patch={},
            requires_new_model=False,
            affected_shots=intent.affected_shots or None,
            notes=["The locked model is unchanged; only the shot plan and references are rebuilt."],
        )
    if category == "scene_composition":
        overrides = {
            "palette_override": None,
            "lighting_overrides": {},
            "staging": _staging_hint(intent),
        }
        return RectificationPlan(
            category="scene_composition",
            description=f"Staging change: {intent.text}",
            design_overrides=overrides,
            shot_adjustments={},
            spec_patch={},
            requires_new_model=False,
            affected_shots=intent.affected_shots or None,
            notes=["The locked model is unchanged; furniture and styling are recomposed."],
        )
    overrides = {
        "palette_override": palette_from_text(intent.text),
        "lighting_overrides": lighting_from_text(intent.text),
        "staging": {},
    }
    return RectificationPlan(
        category="presentation",
        description=f"Presentation change: {intent.text}",
        design_overrides=overrides,
        shot_adjustments={},
        spec_patch={},
        requires_new_model=False,
        affected_shots=intent.affected_shots or None,
        notes=["The locked model is unchanged; lighting, palette or finishes are recommitted."],
    )


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------


def palette_from_text(text: str) -> str | None:
    lowered = text.lower()
    for word, palette in _PALETTE_HINTS.items():
        if word in lowered:
            return palette
    return None


def lighting_from_text(text: str) -> dict[str, Any]:
    lowered = text.lower()
    overrides: dict[str, Any] = {}
    if any(word in lowered for word in ("warmer", "warm", "cosy", "cozy", "golden")):
        overrides["temperature_bias"] = "warm"
        overrides["temperature_k"] = 2700
    if any(word in lowered for word in ("cooler", "cool", "crisp", "neutral white")):
        overrides["temperature_bias"] = "cool"
        overrides["temperature_k"] = 4200
    if any(word in lowered for word in ("brighter", "lighter", "more light", "airy")):
        overrides["exposure_delta"] = 0.12
    if any(word in lowered for word in ("darker", "moodier", "dramatic", "moody")):
        overrides["exposure_delta"] = -0.12
        overrides["contrast_delta"] = 0.06
    if "sunset" in lowered or "golden hour" in lowered:
        overrides["time_of_day"] = "golden_hour"
        overrides["temperature_k"] = 2400
    if "evening" in lowered or "dusk" in lowered:
        overrides["time_of_day"] = "evening"
    return overrides


def _staging_hint(intent: ChangeIntent) -> dict[str, Any]:
    lowered = intent.text.lower()
    hint: dict[str, Any] = {
        "attempt_bump": True,
        "rooms": intent.targets.get("rooms", []),
    }
    if "minimal" in lowered or "less furniture" in lowered or "sparse" in lowered:
        hint["density"] = "sparse"
    elif "fuller" in lowered or "more furniture" in lowered or "lived in" in lowered:
        hint["density"] = "fuller"
    if "sofa" in lowered or "couch" in lowered:
        hint["swap"] = "sofa"
    if "bed" in lowered:
        hint["swap"] = "bed"
    if "table" in lowered:
        hint["swap"] = "table"
    return hint


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def camera_adjustments(intent: ChangeIntent) -> dict[str, Any]:
    lowered = intent.text.lower()
    adjustments: dict[str, Any] = {}
    drop: list[str] = []
    favour: list[str] = []
    if any(word in lowered for word in ("remove", "drop", "skip", "without")):
        for key, shot_type in (
            ("aerial", "aerial_reveal"),
            ("drone", "drone_orbit"),
            ("orbit", "drone_orbit"),
            ("closing", "closing"),
            ("compound", "compound_move"),
        ):
            if key in lowered:
                drop.append(shot_type)
    if any(word in lowered for word in ("entrance", "front door", "doorway", "gate")):
        favour.append("entrance_walkthrough")
    if any(word in lowered for word in ("inside", "interior", "rooms", "walkthrough")):
        favour.append("room_walkthrough")
    if "exterior" in lowered or "facade" in lowered or "outside" in lowered:
        favour.append("exterior_walkaround")
    if "aerial" in lowered or "overview" in lowered:
        favour.append("aerial_reveal")
    if "upper" in lowered or "upstairs" in lowered or "first floor" in lowered:
        favour.append("upper_floor")
    if favour:
        adjustments["reorder"] = favour
    if drop:
        adjustments["drop"] = drop
    if any(word in lowered for word in ("shorter", "less time", "quick", "brief")):
        adjustments["duration_scale"] = 0.85
    if any(word in lowered for word in ("longer", "more time", "slower", "linger")):
        adjustments["duration_scale"] = 1.0  # clip length is provider-fixed; see note
        adjustments["note"] = (
            "Provider clip length is fixed by configuration, so 'longer' is handled by "
            "prioritising the shot and keeping it in the cut rather than extending it."
        )
    return adjustments


# ---------------------------------------------------------------------------
# Structural transforms
# ---------------------------------------------------------------------------


def apply_structural(intent: ChangeIntent, spec: HouseSpec) -> tuple[HouseSpec, dict, list[str]]:
    """Return a new spec. Never mutates the input object."""
    targets = intent.targets or {}
    patch: dict[str, Any] = {}
    notes: list[str] = []
    new_spec = copy.deepcopy(spec)

    roof_type = targets.get("roof_type")
    if roof_type:
        new_spec.roof.type = RoofType(roof_type)
        new_spec.roof.source = SourceType.USER_STATED
        new_spec.roof.confidence = 0.95
        patch["roof_type"] = roof_type
        notes.append(f"Roof changed to {roof_type}.")

    pitch = targets.get("roof_pitch_deg")
    if pitch is not None:
        new_spec.roof.pitch_deg = max(0.0, min(70.0, float(pitch)))
        new_spec.roof.source = SourceType.USER_STATED
        patch["roof_pitch_deg"] = new_spec.roof.pitch_deg
        notes.append(f"Roof pitch set to {new_spec.roof.pitch_deg:.0f}°.")

    wanted_storeys = targets.get("storeys")
    wanted_bedrooms = targets.get("bedrooms")
    if wanted_storeys or wanted_bedrooms:
        if not _is_synthesised(spec):
            raise ClarificationRequired(
                "This plan was interpreted from a drawing, so changing the number of "
                "storeys or bedrooms needs an updated drawing or explicit dimensions. "
                "Upload a revised plan and the model will be rebuilt.",
                details={
                    "reason": "structural_edit_on_drawing_derived_plan",
                    "supported": ["roof_type", "roof_pitch_deg"],
                },
            )
        new_spec = _resynthesise(spec, wanted_storeys, wanted_bedrooms, notes)
        patch["storeys"] = wanted_storeys or len(spec.levels)
        patch["bedrooms"] = wanted_bedrooms
        if wanted_storeys:
            notes.append(f"Rebuilt the plan with {wanted_storeys} storey(s).")
        if wanted_bedrooms:
            notes.append(f"Rebuilt the plan with {wanted_bedrooms} bedroom(s).")

    if not patch:
        raise ClarificationRequired(
            "That structural change is not one this build can apply automatically. "
            "Supported structural edits are roof type, roof pitch, storey count and "
            "bedroom count on a plan generated from requirements.",
            details={"targets": targets},
        )
    new_spec.provenance = {
        **spec.provenance,
        "parent_schema_version": spec.schema_version,
        "structural_edit": patch,
    }
    return new_spec, patch, notes


def _is_synthesised(spec: HouseSpec) -> bool:
    return spec.provenance.get("method") == "deterministic_synthesis"


def _resynthesise(
    spec: HouseSpec, storeys: int | None, bedrooms: int | None, notes: list[str]
) -> HouseSpec:
    from ..spec.templates import PlanRequest, synthesise_plan

    current_bedrooms = sum(1 for r in spec.rooms if r.type is RoomType.BEDROOM)
    request = PlanRequest(
        bedrooms=bedrooms or max(1, current_bedrooms),
        bathrooms=max(1, sum(1 for r in spec.rooms if r.type is RoomType.BATHROOM)),
        storeys=storeys or len(spec.levels),
        living=any(r.type is RoomType.LIVING for r in spec.rooms),
        dining=any(r.type is RoomType.DINING for r in spec.rooms),
        kitchen=any(r.type is RoomType.KITCHEN for r in spec.rooms),
        study=any(r.type is RoomType.STUDY for r in spec.rooms),
        garage=any(r.type is RoomType.GARAGE for r in spec.rooms),
        style_hint=spec.design_intent.style,
    )
    rebuilt = synthesise_plan(request)
    rebuilt.title = spec.title
    rebuilt.design_intent = spec.design_intent
    rebuilt.stated_requirements = spec.stated_requirements
    rebuilt.provenance = {
        **rebuilt.provenance,
        "resynthesised_from": "previous_spec",
        "previous_levels": len(spec.levels),
    }
    notes.append(
        "The room layout was regenerated from the updated requirements; the previous "
        "model version remains available for its existing videos."
    )
    return rebuilt


def validate_plan(plan: RectificationPlan) -> None:
    if plan.requires_new_model and plan.structural_spec is None:
        raise ValidationError("A structural change must produce a new house specification.")
