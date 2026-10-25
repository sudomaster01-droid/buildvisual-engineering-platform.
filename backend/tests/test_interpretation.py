from __future__ import annotations

from app.services.interpret.drawing import spec_from_observations
from app.services.interpret.rules import build_spec_from_requirements
from app.services.spec.schema import SourceType


def test_text_requirements_produce_a_valid_traceable_spec_without_api():
    spec, parsed = build_spec_from_requirements(
        "Four bedroom, two-storey house with 180 sqm, a garage and flat roof",
        seed=14,
    )

    validated = spec.__class__.model_validate(spec.model_dump())
    assert validated.storey_count == 2
    assert parsed.explicit["bedrooms"] == 4
    assert parsed.explicit["storeys"] == 2
    assert parsed.explicit["area_m2"] == 180
    assert validated.roof.source is SourceType.USER_STATED
    assert validated.provenance["method"] == "deterministic_synthesis"
    assert all(room.source is SourceType.INFERRED for room in validated.rooms)


def test_drawing_observations_keep_estimated_dimensions_inferred_and_ids_stable():
    observations = {
        "title": "Sample plan",
        "levels": [{
            "index": 0,
            "name": "Ground floor",
            "rooms": [{
                "name": "Living room",
                "type": "living",
                "x": 0,
                "y": 0,
                "w": 5,
                "d": 4,
                "confidence": 0.95,
            }],
        }],
        "unresolved": ["The drawing scale is not shown."],
    }

    first, report = spec_from_observations(observations)
    second, _ = spec_from_observations(observations)

    assert first.__class__.model_validate(first.model_dump())
    assert first.rooms[0].source is SourceType.INFERRED
    assert first.rooms[0].confidence == 0.95
    assert first.rooms[0].id == second.rooms[0].id
    assert first.unresolved[0].description == "The drawing scale is not shown."
    assert report["measured_rooms"] == 0
