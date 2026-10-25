"""Deterministic parsing of typed/spoken house requirements.

This is the no-credentials path: it understands the requirements the engineer
actually writes ("four bedroom two storey house with a garage, open plan living,
flat roof") and turns them into a labelled, inferred plan via
``services.spec.templates``. It never claims the result is measured.

It is also the fallback whenever an LLM parse fails validation, so text input
always has a working path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..spec.schema import HouseSpec, RoofType, SourceType
from ..spec.templates import PlanRequest, synthesise_plan

_NUMBER_WORDS = {
    "no": 0, "zero": 0, "one": 1, "single": 1, "two": 2, "double": 2, "three": 3,
    "triple": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_STYLE_HINTS = {
    "modern": "modern", "contemporary": "modern", "flat roof": "modern flat",
    "minimal": "modern", "traditional": "traditional", "classic": "traditional",
    "mediterranean": "mediterranean", "spanish": "mediterranean",
    "scandinavian": "scandinavian", "colonial": "colonial",
}


@dataclass
class RequirementParse:
    request: PlanRequest
    stated: list[str] = field(default_factory=list)
    explicit: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _number(text: str) -> int | None:
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    return _NUMBER_WORDS.get(text)


def parse_requirements(text: str) -> RequirementParse:
    """Extract a house program from free text. Deterministic and order-independent."""
    lowered = f" {text.lower()} "
    explicit: dict[str, object] = {}
    stated: list[str] = []
    warnings: list[str] = []

    bedroom_match = re.search(
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)[\s-]*bed(?:room)?s?\b",
        lowered,
    )
    if bedroom_match:
        explicit["bedrooms"] = _number(bedroom_match.group(1))
    else:
        alt = re.search(r"\b(\w+)[\s-]*bedroom\b", lowered)
        if alt and _number(alt.group(1)) is not None:
            explicit["bedrooms"] = _number(alt.group(1))

    bathroom_match = re.search(
        r"(\d+|one|two|three|four|five)[\s-]*(?:bath|bathroom|toilet|ensuite)s?\b", lowered
    )
    if bathroom_match:
        explicit["bathrooms"] = _number(bathroom_match.group(1))

    storey_match = re.search(
        r"(\d+|one|two|three|single|double|triple)[\s-]*(?:storey|story|storeys|stories|level|floors?)\b",
        lowered,
    )
    if storey_match:
        explicit["storeys"] = _number(storey_match.group(1))
    elif "bungalow" in lowered or "single storey" in lowered or "single-storey" in lowered:
        explicit["storeys"] = 1

    area_match = re.search(r"(\d{2,4})\s*(?:m2|m²|sqm|square met|sq\.? ?m)", lowered)
    if area_match:
        explicit["area_m2"] = float(area_match.group(1))

    explicit["garage"] = bool(re.search(r"\bgarage|carport|parking\b", lowered))
    explicit["study"] = bool(re.search(r"\bstudy|home office|office\b", lowered))
    explicit["living"] = not bool(re.search(r"\bno living|living only\b", lowered))
    explicit["dining"] = not bool(re.search(r"\bno dining\b", lowered))
    explicit["kitchen"] = not bool(re.search(r"\bno kitchen\b", lowered))

    for word, style in _STYLE_HINTS.items():
        if word in lowered:
            explicit["style"] = style
            break

    if re.search(r"\b(gable|pitched|tiled) roof\b", lowered):
        explicit["roof"] = "gable"
    elif re.search(r"\bflat roof\b", lowered):
        explicit["roof"] = "flat"

    # Statements worth carrying through to the prompt as client intent.
    for marker in (
        "open plan", "double volume", "high ceiling", "large windows", "floor to ceiling",
        "balcony", "veranda", "terrace", "pool", "courtyard", "ensuite", "walk-in",
        "farmhouse kitchen", "island", "roof terrace", "carport", "annexe", "granny flat",
    ):
        if marker in lowered:
            stated.append(marker)

    if not explicit.get("bedrooms"):
        warnings.append(
            "No bedroom count was stated; a three-bedroom house was assumed."
        )
    if not explicit.get("storeys"):
        warnings.append("No storey count was stated; a single storey was assumed.")

    request = PlanRequest(
        bedrooms=int(explicit.get("bedrooms") or 3),
        bathrooms=int(explicit.get("bathrooms") or 2),
        storeys=int(explicit.get("storeys") or 1),
        living=bool(explicit.get("living", True)),
        dining=bool(explicit.get("dining", True)),
        kitchen=bool(explicit.get("kitchen", True)),
        study=bool(explicit.get("study", False)),
        garage=bool(explicit.get("garage", False)),
        target_area_m2=float(explicit["area_m2"]) if explicit.get("area_m2") else None,
        style_hint=explicit.get("style") if isinstance(explicit.get("style"), str) else None,
    )
    return RequirementParse(request=request, stated=stated, explicit=explicit, warnings=warnings)


def build_spec_from_requirements(text: str, *, seed: int = 0) -> tuple[HouseSpec, RequirementParse]:
    parsed = parse_requirements(text)
    spec = synthesise_plan(parsed.request, seed=seed)
    spec.stated_requirements = parsed.stated + [
        line.strip() for line in re.split(r"[.\n;]", text) if 20 <= len(line.strip()) <= 160
    ][:6]
    roof = parsed.explicit.get("roof")
    if roof == "flat":
        spec.roof.type = RoofType.FLAT
        spec.roof.pitch_deg = 3.0
        spec.roof.source = SourceType.USER_STATED
        spec.roof.confidence = 0.9
    spec.provenance = {
        **spec.provenance,
        "method": "deterministic_synthesis",
        "parser": "rules:v1",
        "explicit_facts": {k: v for k, v in parsed.explicit.items() if v not in (None, False)},
        "parser_warnings": parsed.warnings,
        "requirements_excerpt": text[:400],
    }
    return spec, parsed
