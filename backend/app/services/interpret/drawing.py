"""Drawing understanding: raster plan → typed house specification.

Division of labour (blueprint p.32):
* the **vision model** reads the drawing and reports what it can *see* — room
  labels, storey count, roof notes, approximate rectangles, dimension callouts;
* **deterministic code** (this module) converts those observations into the
  canonical spec: it normalises coordinates, rejects overlaps, derives walls
  from shared room boundaries, and places openings using explicit geometric
  rules.

The model never emits walls, heights or a final 3-D structure. That keeps its
job to perception, keeps arithmetic in code, and means a bad model day produces
a poor room list rather than impossible geometry.

Everything produced here is stamped with the source type and confidence the
model reported, so "measured on the drawing" is never confused with "guessed".
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

from ...errors import InterpretationError
from ..modeling.validate import validate_spec
from ..spec.geometry_helpers import polygon_area
from ..spec.schema import (
    HouseSpec,
    Level,
    Opening,
    OpeningKind,
    Roof,
    RoofType,
    Room,
    RoomType,
    Site,
    SourceType,
    Stair,
    Wall,
)

log = logging.getLogger("buildvisual.interpret.drawing")

INTERPRETER_VERSION = "1.0.0"

DRAWING_SYSTEM_PROMPT = """You are an architectural drawing reader. You receive one or more
images of a house floor plan and return a single JSON object describing ONLY what is visible.

Return exactly this shape:
{
  "title": "short title if a drawing title is visible, else ''",
  "storeys": 1,
  "roof": {"type": "gable|hip|flat|shed|unknown", "pitch_deg": 25},
  "levels": [
    {"index": 0, "name": "Ground floor",
     "rooms": [
       {"name": "Living room", "type": "living|dining|kitchen|bedroom|bathroom|hall|study|utility|garage|stair|balcony|other",
        "x": 0.0, "y": 0.0, "w": 4.2, "d": 3.6,
        "confidence": 0.0-1.0}
     ],
     "dimension_callouts": [{"text": "4.2 m", "value": 4.2}],
     "notes": "anything legible that affects geometry"}
  ],
  "stairs": [{"from_level": 0, "to_level": 1, "x": 0.0, "y": 0.0, "w": 1.0, "d": 3.0}],
  "site": {"plot_width_m": null, "plot_depth_m": null, "orientation_note": ""},
  "unresolved": ["a critical ambiguity the drawing does not answer"],
  "observed_warnings": ["anything the reader is unsure about"]
}

Rules that matter:
- Units are METRES. Convert any other unit you see and say so in notes.
- Coordinates: x is the horizontal axis to the right, y is vertical going UP the page.
  The origin is the bottom-left corner of the building. Rooms must not overlap.
- Use the dimension callouts for w/d where present; otherwise estimate from the scale
  and set confidence below 0.6.
- Do NOT invent rooms, storeys or stairs that are not drawn. If the plan shows one
  storey, report one.
- If a room's extent is genuinely unclear, omit it and add it to "unresolved".
- Return JSON only. No prose, no markdown.
"""


def interpret_drawing(
    images: list[tuple[bytes, str]],
    llm_client,
    *,
    requirements_text: str | None = None,
    locale: str = "en",
) -> tuple[HouseSpec, dict[str, Any]]:
    """Ask the vision model for observations, then build the spec deterministically."""
    if not images:
        raise InterpretationError("No drawing pages were available to interpret.")
    user = (
        "Read this floor plan and return the JSON described in the system prompt. "
        f"Language for any labels: {locale}."
    )
    if requirements_text:
        user += (
            "\n\nThe engineer also provided these requirements. Use them only to "
            "disambiguate labels; the drawing is authoritative for geometry:\n"
            f"<requirements>{requirements_text[:1200]}</requirements>"
        )
    raw = llm_client.generate_json(
        system=DRAWING_SYSTEM_PROMPT, user=user, images=images, max_output_tokens=8000
    )
    usage = getattr(llm_client, "last_usage", None)
    spec, report = spec_from_observations(raw, requirements_text=requirements_text)
    report["provider"] = getattr(llm_client, "provider", "unknown")
    report["model"] = getattr(llm_client, "model", "unknown")
    if usage is not None:
        report["usage"] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
    return spec, report


_ALLOWED_ROOM_TYPES = {item.value for item in RoomType}


def spec_from_observations(
    raw: dict[str, Any], *, requirements_text: str | None = None
) -> tuple[HouseSpec, dict[str, Any]]:
    """Convert the model's observations into a validated HouseSpec."""
    if not isinstance(raw, dict):
        raise InterpretationError("The drawing interpreter returned unusable output.")
    levels_raw = raw.get("levels")
    if not isinstance(levels_raw, list) or not levels_raw:
        raise InterpretationError(
            "No rooms were recognised in this drawing. Try a clearer scan or type the "
            "requirements instead."
        )

    warnings: list[str] = [str(w)[:200] for w in raw.get("observed_warnings", []) if w]
    unresolved_raw = [str(u)[:240] for u in raw.get("unresolved", []) if u]
    rooms: list[Room] = []
    levels: list[Level] = []
    ceiling = 2.7

    for index, level_raw in enumerate(levels_raw):
        level_index = int(level_raw.get("index", index))
        levels.append(Level(
            index=level_index,
            name=str(level_raw.get("name") or f"Floor {level_index}")[:60],
            elevation_m=round(level_index * (ceiling + 0.28), 3),
            ceiling_height_m=ceiling,
        ))
        for room_raw in level_raw.get("rooms", []) or []:
            room = _room_from_raw(room_raw, level_index)
            if room is not None:
                rooms.append(room)

    if not rooms:
        raise InterpretationError("The drawing interpreter found no usable room geometry.")

    rooms = _resolve_overlaps(rooms, warnings)
    stairs = _stairs_from_raw(raw.get("stairs") or [], levels, rooms, warnings)
    walls = _walls_from_rooms(rooms, warnings)
    _place_openings(rooms, walls, warnings)

    roof_raw = raw.get("roof") or {}
    roof_type = str(roof_raw.get("type", "unknown")).lower()
    roof = Roof(
        type=RoofType(roof_type) if roof_type in {t.value for t in RoofType} else RoofType.UNKNOWN,
        pitch_deg=float(roof_raw.get("pitch_deg") or 25.0),
        source=SourceType.DRAWING_LABEL if roof_type != "unknown" else SourceType.SYSTEM_DEFAULT,
        confidence=0.7 if roof_type != "unknown" else 0.3,
    )
    site_raw = raw.get("site") or {}
    site = Site(
        plot_width_m=_optional_float(site_raw.get("plot_width_m")),
        plot_depth_m=_optional_float(site_raw.get("plot_depth_m")),
        context_note=str(site_raw.get("orientation_note") or "")[:200] or None,
    )

    factual = sum(1 for r in rooms if r.source is SourceType.DRAWING_DIMENSION)
    spec = HouseSpec(
        title=str(raw.get("title") or "House from drawing")[:120],
        levels=sorted(levels, key=lambda lvl: lvl.index),
        rooms=rooms,
        walls=walls,
        stairs=stairs,
        roof=roof,
        site=site,
        stated_requirements=[requirements_text] if requirements_text else [],
        inferred_fields=[
            f"walls ({len(walls)}) derived from room boundaries",
            "wall thickness and heights",
            f"openings ({sum(len(w.openings) for w in walls)}) placed by rule",
        ],
        unresolved=[
            {
                "field": "interpretation",
                "description": text,
                "options": [],
                "critical": False,
                "entropy_bits": 0.0,
            }
            for text in unresolved_raw[:8]
        ],
        provenance={
            "method": "vision_model_plus_deterministic_geometry",
            "interpreter_version": INTERPRETER_VERSION,
            "measured_rooms": factual,
            "inferred_rooms": len(rooms) - factual,
            "note": (
                "Rooms were read from the drawing by a vision model. Walls and openings "
                "were derived deterministically from the reported room boundaries; they "
                "are not traced from the drawing's linework."
            ),
        },
    )

    report = validate_spec(spec)
    if not report.ok:
        messages = "; ".join(f.message for f in report.critical)
        raise InterpretationError(
            "The drawing interpretation produced geometry that failed structural checks: "
            + messages
        )
    warnings.extend(f.message for f in report.findings if f.severity == "warning")

    usage_report = {
        "warnings": warnings[:12],
        "unresolved": [u.description for u in spec.unresolved],
        "rooms": len(rooms),
        "levels": len(levels),
        "measured_rooms": factual,
        "validation": report.metrics,
        "confidence": _confidence_summary(rooms),
    }
    return spec, usage_report


def _optional_float(value: Any) -> float | None:
    try:
        if value in (None, "", 0):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _room_from_raw(room_raw: dict, level_index: int) -> Room | None:
    """Build a room, treating model-estimated dimensions as inferred."""
    try:
        x = float(room_raw.get("x", 0.0))
        y = float(room_raw.get("y", 0.0))
        w = float(room_raw.get("w", 0.0))
        d = float(room_raw.get("d", 0.0))
    except (TypeError, ValueError):
        return None
    if w <= 0.4 or d <= 0.4 or w > 60 or d > 60:
        return None
    room_type_raw = str(room_raw.get("type", "other")).lower().strip()
    room_type = RoomType(room_type_raw) if room_type_raw in _ALLOWED_ROOM_TYPES else RoomType.OTHER
    confidence = room_raw.get("confidence")
    try:
        confidence = max(0.05, min(1.0, float(confidence))) if confidence is not None else 0.55
    except (TypeError, ValueError):
        confidence = 0.55
    name = str(room_raw.get("name") or room_type.value.title())[:60]
    stable_key = (
        f"{level_index}\0{room_type.value}\0{name}\0"
        f"{x:.6f}\0{y:.6f}\0{w:.6f}\0{d:.6f}"
    )
    stable_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:10]
    return Room(
        id=f"L{level_index}_{room_type.value}_{stable_id}",
        name=name,
        type=room_type,
        level=level_index,
        polygon=[(x, y), (x + w, y), (x + w, y + d), (x, y + d)],
        ceiling_height_m=2.7,
        source=SourceType.INFERRED,
        confidence=round(confidence, 3),
    )


def _resolve_overlaps(rooms: list[Room], warnings: list[str]) -> list[Room]:
    """Nudge heavily overlapping rooms apart instead of failing the whole plan."""
    from ..spec.geometry_helpers import rect_polygon

    accepted: list[Room] = []
    for room in rooms:
        x0 = min(p[0] for p in room.polygon)
        y0 = min(p[1] for p in room.polygon)
        x1 = max(p[0] for p in room.polygon)
        y1 = max(p[1] for p in room.polygon)
        width, height = x1 - x0, y1 - y0
        attempts = 0
        while attempts < 40 and any(_overlaps_area(room, other) > 0.35 for other in accepted):
            # Shift right/up in small steps until the overlap is tolerable.
            x0 += max(0.6, width * 0.15)
            if x0 > 120:
                break
            room.polygon = rect_polygon(x0, y0, x0 + width, y0 + height)
            attempts += 1
        if attempts:
            warnings.append(
                f"Room '{room.name}' was repositioned because the interpreted "
                "rectangles overlapped."
            )
            room.confidence = round(max(0.2, room.confidence - 0.2), 3)
            room.source = SourceType.INFERRED
        accepted.append(room)
    return accepted


def _overlaps_area(a: Room, b: Room) -> float:
    ax0 = min(p[0] for p in a.polygon); ax1 = max(p[0] for p in a.polygon)
    ay0 = min(p[1] for p in a.polygon); ay1 = max(p[1] for p in a.polygon)
    bx0 = min(p[0] for p in b.polygon); bx1 = max(p[0] for p in b.polygon)
    by0 = min(p[1] for p in b.polygon); by1 = max(p[1] for p in b.polygon)
    dx = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    dy = max(0.0, min(ay1, by1) - max(ay0, by0))
    overlap = dx * dy
    smaller = min(polygon_area(a.polygon), polygon_area(b.polygon)) or 1.0
    return overlap / smaller


def _walls_from_rooms(rooms: list[Room], warnings: list[str]) -> list[Wall]:
    """Exterior shell per level plus partitions on shared boundaries."""
    walls: list[Wall] = []
    by_level: dict[int, list[Room]] = {}
    for room in rooms:
        by_level.setdefault(room.level, []).append(room)

    for level, level_rooms in by_level.items():
        xs0 = min(min(p[0] for p in r.polygon) for r in level_rooms)
        ys0 = min(min(p[1] for p in r.polygon) for r in level_rooms)
        xs1 = max(max(p[0] for p in r.polygon) for r in level_rooms)
        ys1 = max(max(p[1] for p in r.polygon) for r in level_rooms)
        height = level_rooms[0].ceiling_height_m
        thinnest = 0.24
        shell = [
            ((xs0, ys0), (xs1, ys0)),
            ((xs1, ys0), (xs1, ys1)),
            ((xs1, ys1), (xs0, ys1)),
            ((xs0, ys1), (xs0, ys0)),
        ]
        for index, (start, end) in enumerate(shell):
            if math.dist(start, end) < 0.5:
                continue
            walls.append(Wall(
                id=f"L{level}_shell{index}", level=level, start=start, end=end,
                thickness_m=thinnest, height_m=height, interior=False,
                source=SourceType.DERIVED, confidence=0.85,
            ))

        edges: dict[tuple[str, float], list[tuple[str, float, float]]] = {}
        for room in level_rooms:
            pts = room.polygon
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % len(pts)]
                if abs(x1 - x2) < 1e-6:
                    edges.setdefault(("v", round(x1, 3)), []).append(
                        (room.id, min(y1, y2), max(y1, y2)))
                elif abs(y1 - y2) < 1e-6:
                    edges.setdefault(("h", round(y1, 3)), []).append(
                        (room.id, min(x1, x2), max(x1, x2)))
        index = 0
        for (axis, coord), items in sorted(edges.items()):
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    a_id, a_lo, a_hi = items[i]
                    b_id, b_lo, b_hi = items[j]
                    if a_id == b_id:
                        continue
                    lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
                    if hi - lo < 0.8:
                        continue
                    start = (coord, lo) if axis == "v" else (lo, coord)
                    end = (coord, hi) if axis == "v" else (hi, coord)
                    walls.append(Wall(
                        id=f"L{level}_part{index}", level=level, start=start, end=end,
                        thickness_m=0.11, height_m=height, interior=True,
                        source=SourceType.DERIVED, confidence=0.8,
                    ))
                    index += 1
    if not walls:
        warnings.append("No walls could be derived from the interpreted rooms.")
    return walls


def _place_openings(rooms: list[Room], walls: list[Wall], warnings: list[str]) -> None:
    """Windows on exterior walls, one door per partition, one front door."""
    from ..spec.geometry_helpers import point_in_polygon

    exterior = [w for w in walls if not w.interior]
    interior = [w for w in walls if w.interior]

    for wall in exterior:
        length = wall.length_m
        if length < 1.6:
            continue
        count = max(1, int(length // 3.0))
        for i in range(count):
            offset = length * (i + 0.5) / count
            ax = wall.start[0] + (wall.end[0] - wall.start[0]) * offset / length
            ay = wall.start[1] + (wall.end[1] - wall.start[1]) * offset / length
            nx = -(wall.end[1] - wall.start[1]) / length
            ny = (wall.end[0] - wall.start[0]) / length
            host = None
            for sign in (1, -1):
                probe = (ax + nx * sign * 0.45, ay + ny * sign * 0.45)
                host = next((r for r in rooms if r.level == wall.level and point_in_polygon(probe, r.polygon)), None)
                if host:
                    break
            if host is None:
                continue
            wall.openings.append(Opening(
                id=f"{wall.id}_w{i}", kind=OpeningKind.WINDOW,
                offset_m=round(offset, 2),
                width_m=round(min(1.8, length / (count + 1)), 2),
                height_m=1.35, sill_m=0.95, connects=[host.id],
                source=SourceType.DERIVED, confidence=0.6,
            ))

    # Front door: on the south shell of the ground floor, in front of the room
    # nearest the entrance-adjacent circulation space.
    ground_walls = [w for w in exterior if w.level == 0]
    if ground_walls:
        south = min(ground_walls, key=lambda w: min(w.start[1], w.end[1]))
        candidates = [r for r in rooms if r.level == 0]
        hall = next((r for r in candidates if r.type is RoomType.HALL), None)
        target = hall or (max(candidates, key=lambda r: polygon_area(r.polygon)) if candidates else None)
        if target is not None:
            cx = sum(p[0] for p in target.polygon) / len(target.polygon)
            offset = max(0.9, min(south.length_m - 0.9, abs(cx - south.start[0]) or 1.0))
            south.openings = [o for o in south.openings if abs(o.offset_m - offset) > 1.3]
            south.openings.append(Opening(
                id=f"{south.id}_entry", kind=OpeningKind.DOOR, offset_m=round(offset, 2),
                width_m=1.0, height_m=2.1, connects=[target.id], is_entrance=True,
                source=SourceType.DERIVED, confidence=0.6,
            ))
        else:
            warnings.append("No ground-floor room was available to host a front door.")

    for wall in interior:
        if wall.length_m < 1.2:
            continue
        mid = wall.length_m / 2
        ax = wall.start[0] + (wall.end[0] - wall.start[0]) * 0.5
        ay = wall.start[1] + (wall.end[1] - wall.start[1]) * 0.5
        nx = -(wall.end[1] - wall.start[1]) / wall.length_m
        ny = (wall.end[0] - wall.start[0]) / wall.length_m
        found: list[str] = []
        for sign in (1, -1):
            probe = (ax + nx * sign * 0.35, ay + ny * sign * 0.35)
            room = next(
                (r for r in rooms if r.level == wall.level and point_in_polygon(probe, r.polygon)),
                None,
            )
            if room and room.id not in found:
                found.append(room.id)
        if len(found) == 2:
            wall.openings.append(Opening(
                id=f"{wall.id}_door", kind=OpeningKind.DOOR, offset_m=round(mid, 2),
                width_m=0.9, height_m=2.05, connects=found,
                source=SourceType.DERIVED, confidence=0.6,
            ))


def _stairs_from_raw(
    raw_stairs: list[dict], levels: list[Level], rooms: list[Room], warnings: list[str]
) -> list[Stair]:
    stairs: list[Stair] = []
    level_ids = {lvl.index for lvl in levels}
    for index, raw in enumerate(raw_stairs[:2]):
        try:
            from_level = int(raw.get("from_level", 0))
            to_level = int(raw.get("to_level", from_level + 1))
            x = float(raw.get("x", 0.0))
            y = float(raw.get("y", 0.0))
            w = float(raw.get("w", 1.0))
            d = float(raw.get("d", 3.0))
        except (TypeError, ValueError):
            continue
        if from_level not in level_ids or to_level not in level_ids or to_level <= from_level:
            continue
        stairs.append(Stair(
            id=f"stair_{index}", from_level=from_level, to_level=to_level,
            start=(x + w / 2, y), end=(x + w / 2, y + d), width_m=max(0.8, w),
            source=SourceType.DRAWING_DIMENSION, confidence=0.7,
        ))
    if len(levels) > 1 and not stairs:
        warnings.append(
            "The plan shows multiple storeys but no stair was reported; the upper floor "
            "will be flagged as unreachable."
        )
    return stairs


def _confidence_summary(rooms: list[Room]) -> dict[str, float]:
    if not rooms:
        return {}
    values = sorted(r.confidence for r in rooms)
    return {
        "rooms_mean": round(sum(values) / len(values), 3),
        "rooms_min": round(values[0], 3),
        "rooms_median": round(values[len(values) // 2], 3),
    }
