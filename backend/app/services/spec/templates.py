"""Deterministic plan synthesis for text/voice-only input.

When the engineer describes a house in words and supplies no drawing, the system
must still produce *a* house — but it must never present invented geometry as if
it were measured. Everything generated here is stamped
``source=SourceType.INFERRED`` with a low confidence, and the job carries a
warning that the plan is indicative. The drawing path never uses this module.

The layout logic is deterministic (grid subdivision driven by room program), so
the same requirements always produce the same geometry hash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schema import (
    HouseSpec,
    Level,
    Opening,
    OpeningKind,
    Provenanced,
    Roof,
    RoofType,
    Room,
    RoomType,
    Site,
    SourceType,
    Stair,
    Wall,
)

INFERRED_CONFIDENCE = 0.55


@dataclass
class PlanRequest:
    bedrooms: int = 3
    bathrooms: int = 2
    storeys: int = 1
    living: bool = True
    dining: bool = True
    kitchen: bool = True
    study: bool = False
    garage: bool = False
    #: total habitable area; if None it is derived from the room program
    target_area_m2: float | None = None
    style_hint: str | None = None


def synthesise_plan(request: PlanRequest, *, seed: int = 0) -> HouseSpec:
    """Build an indicative, clearly-labelled inferred house."""
    bedrooms = max(1, min(request.bedrooms, 10))
    bathrooms = max(1, min(request.bathrooms, bedrooms + 2))
    storeys = max(1, min(request.storeys, 3))

    # Area budget: room program → area → side lengths (near-square plan reads well
    # in an aerial reveal and keeps circulation simple).
    area = request.target_area_m2 or (
        70
        + bedrooms * 16
        + bathrooms * 6
        + (40 if request.living else 0)
        + (18 if request.dining else 0)
        + (16 if request.kitchen else 0)
        + (14 if request.study else 0)
        + (22 if request.garage else 0)
    )
    area = max(45.0, min(area, 900.0))
    per_storey = area / storeys
    aspect = 1.0 + 0.06 * ((seed % 3) - 1)  # tiny deterministic variation
    width = round(math.sqrt(per_storey * aspect) / 0.6) * 0.6
    depth = round((per_storey / width) / 0.6) * 0.6

    levels: list[Level] = []
    rooms: list[Room] = []
    walls: list[Wall] = []
    stairs: list[Stair] = []

    per_storey_area = per_storey
    for level_index in range(storeys):
        elevation = 0.0 if level_index == 0 else levels[-1].elevation_m + levels[-1].ceiling_height_m + 0.28
        levels.append(Level(
            index=level_index,
            name="Ground floor" if level_index == 0 else f"Floor {level_index + 1}",
            elevation_m=round(elevation, 3),
            ceiling_height_m=2.7 if level_index == 0 else 2.6,
        ))
        storey_rooms, storey_walls = _layout_storey(
            width=width,
            depth=depth,
            level_index=level_index,
            request=request,
            room_split=_room_program(
                request, bedrooms, bathrooms, storeys, level_index, per_storey_area
            ),
        )
        rooms.extend(storey_rooms)
        walls.extend(storey_walls)

    if storeys > 1:
        # Stair in the entrance hall of each lower floor, running north.
        for level_index in range(storeys - 1):
            hall = next(
                (r for r in rooms if r.level == level_index and r.type is RoomType.HALL),
                next(r for r in rooms if r.level == level_index),
            )
            cx = sum(p[0] for p in hall.polygon) / len(hall.polygon)
            cy = sum(p[1] for p in hall.polygon) / len(hall.polygon)
            stairs.append(Stair(
                id=f"stair_L{level_index}",
                from_level=level_index,
                to_level=level_index + 1,
                start=(round(cx - 0.9, 2), round(cy - 1.4, 2)),
                end=(round(cx - 0.9, 2), round(cy + 1.4, 2)),
                width_m=1.0,
                source=SourceType.INFERRED,
                confidence=INFERRED_CONFIDENCE,
            ))

    spec = HouseSpec(
        title=f"{bedrooms}-bedroom {'house' if storeys == 1 else 'storey house'}",
        levels=levels,
        rooms=rooms,
        walls=walls,
        stairs=stairs,
        roof=Roof(
            type=RoofType.GABLE if request.style_hint != "modern flat" else RoofType.FLAT,
            pitch_deg=32.0 if request.style_hint != "modern flat" else 3.0,
            overhang_m=0.7,
            source=SourceType.SYSTEM_DEFAULT,
            confidence=0.45,
        ),
        site=Site(
            plot_width_m=round(width + 12, 1),
            plot_depth_m=round(depth + 14, 1),
        ),
        inferred_fields=[
            "levels", "rooms", "walls", "openings", "roof", "site",
        ],
        provenance={
            "method": "deterministic_synthesis",
            "note": (
                "No drawing was supplied. This plan is an indicative arrangement "
                "generated from the stated requirements, not a measured survey. "
                "Rooms, walls and openings are inferred."
            ),
        },
    )
    return spec


def _room_program(request, bedrooms, bathrooms, storeys, level_index, target_area) -> list[tuple[RoomType, str, float]]:
    """Return [(type, name, weight)] for one storey."""
    program: list[tuple[RoomType, str, float]] = []
    if level_index == 0:
        if request.living:
            program.append((RoomType.LIVING, "Living room", 1.9))
        if request.dining:
            program.append((RoomType.DINING, "Dining", 1.0))
        if request.kitchen:
            program.append((RoomType.KITCHEN, "Kitchen", 0.9))
        program.append((RoomType.HALL, "Entrance hall", 0.6))
        program.append((RoomType.BATHROOM, "Ground bathroom", 0.45))
        if request.study:
            program.append((RoomType.STUDY, "Study", 0.7))
        if request.garage:
            program.append((RoomType.GARAGE, "Garage", 1.3))
        if storeys == 1:
            for i in range(bedrooms):
                program.append((RoomType.BEDROOM, f"Bedroom {i + 1}", 1.15))
            for i in range(max(0, bathrooms - 1)):
                program.append((RoomType.BATHROOM, f"Bathroom {i + 1}", 0.45))
        else:
            for i in range(min(bedrooms, 1)):
                program.append((RoomType.BEDROOM, f"Bedroom {i + 1}", 1.15))
    else:
        remaining = bedrooms - (1 if level_index == 1 else 0)
        per = max(1, math.ceil(remaining / max(1, storeys - level_index)))
        for i in range(per):
            program.append((RoomType.BEDROOM, f"Bedroom {level_index * 10 + i + 1}", 1.15))
        for i in range(min(bathrooms, 2)):
            program.append((RoomType.BATHROOM, f"Bathroom {level_index * 10 + i + 1}", 0.45))
        program.append((RoomType.HALL, "Landing", 0.5))
    if not program:  # pragma: no cover - defensive
        program.append((RoomType.LIVING, "Living room", 1.0))
    return program


def _layout_storey(
    *, width: float, depth: float, level_index: int, request: PlanRequest,
    room_split: list[tuple[RoomType, str, float]],
) -> tuple[list[Room], list[Wall]]:
    """Slice a rectangle into rooms with a deterministic binary split."""
    base_y = 0.0
    bay = depth / 2.0
    local_elev = level_index

    cells = _slice_rect(
        (0.0, base_y, width, bay), [w for _, _, w in room_split[: max(1, len(room_split) // 2 + len(room_split) % 2)]]
    )
    lower = _slice_rect(
        (0.0, base_y + bay, width, base_y + depth),
        [w for _, _, w in room_split[len(cells):]] or [1.0],
    )
    cells = cells + lower

    rooms: list[Room] = []
    for i, (kind, name, _weight) in enumerate(room_split):
        rect = cells[i] if i < len(cells) else cells[-1]
        x0, y0, x1, y1 = rect
        rooms.append(Room(
            id=f"L{level_index}_{kind.value}_{i}",
            name=name,
            type=kind,
            level=local_elev,
            polygon=[(round(x0, 2), round(y0, 2)), (round(x1, 2), round(y0, 2)),
                     (round(x1, 2), round(y1, 2)), (round(x0, 2), round(y1, 2))],
            ceiling_height_m=2.7 if level_index == 0 else 2.6,
            source=SourceType.INFERRED,
            confidence=INFERRED_CONFIDENCE,
        ))

    walls = _walls_for_storey(rooms, width, depth, level_index)
    _add_openings(rooms, walls, level_index, request)
    return rooms, walls


def _slice_rect(rect, weights: list[float]) -> list[tuple[float, float, float, float]]:
    """Split a rectangle along its longer axis using the given weights."""
    x0, y0, x1, y1 = rect
    weights = [max(0.15, w) for w in weights] or [1.0]
    total = sum(weights)
    cells = []
    if (x1 - x0) >= (y1 - y0):
        cursor = x0
        for w in weights:
            cut = cursor + (x1 - x0) * w / total
            cells.append((cursor, y0, cut, y1))
            cursor = cut
    else:
        cursor = y0
        for w in weights:
            cut = cursor + (y1 - y0) * w / total
            cells.append((x0, cursor, x1, cut))
            cursor = cut
    return cells


def _walls_for_storey(rooms: list[Room], width: float, depth: float, level_index: int) -> list[Wall]:
    """Exterior shell + interior partitions along shared room boundaries."""
    walls: list[Wall] = []
    height = 2.7 if level_index == 0 else 2.6
    shell = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, depth)),
        ((width, depth), (0.0, depth)),
        ((0.0, depth), (0.0, 0.0)),
    ]
    for i, (start, end) in enumerate(shell):
        walls.append(Wall(
            id=f"L{level_index}_ext{i}",
            level=level_index,
            start=start,
            end=end,
            thickness_m=0.24,
            height_m=height,
            interior=False,
            source=SourceType.INFERRED,
            confidence=INFERRED_CONFIDENCE,
        ))

    for index, (_a, _b, axis, coord, lo, hi) in enumerate(_shared_boundaries(rooms)):
        if hi - lo < 0.6:
            continue
        start = (coord, lo) if axis == "v" else (lo, coord)
        end = (coord, hi) if axis == "v" else (hi, coord)
        walls.append(Wall(
            id=f"L{level_index}_int{index}",
            level=level_index,
            start=start,
            end=end,
            thickness_m=0.1,
            height_m=height,
            interior=True,
            source=SourceType.INFERRED,
            confidence=INFERRED_CONFIDENCE,
        ))
    return walls


def _shared_boundaries(rooms: list[Room]) -> list[tuple[str, str, str, float, float, float]]:
    """Find wall lines between adjacent rooms.

    Returns tuples ``(room_a, room_b, axis, coord, lo, hi)`` where ``axis`` is
    ``v`` (wall at constant x) or ``h`` (wall at constant y).
    """
    edges: dict[tuple[str, float], list[tuple[str, float, float]]] = {}
    for room in rooms:
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
    shared: list[tuple[str, str, str, float, float, float]] = []
    for (axis, coord), items in edges.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a_id, a_lo, a_hi = items[i]
                b_id, b_lo, b_hi = items[j]
                if a_id == b_id:
                    continue
                lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
                if hi - lo > 0.6:
                    shared.append((a_id, b_id, axis, coord, lo, hi))
    shared.sort(key=lambda item: (item[2], item[3], item[4]))
    return shared


def _add_openings(rooms: list[Room], walls: list[Wall], level_index: int, request: PlanRequest) -> None:
    """Windows on exterior walls facing habitable rooms, doors on shared partitions."""
    from .geometry_helpers import point_in_polygon

    exterior = [w for w in walls if not w.interior]
    interior = [w for w in walls if w.interior]

    # Windows: for each exterior wall, place windows in front of rooms that touch it.
    for wall in exterior:
        length = wall.length_m
        if length < 2.0:
            continue
        count = max(1, int(length // 2.6))
        for i in range(count):
            offset = length * (i + 0.5) / count
            ax = wall.start[0] + (wall.end[0] - wall.start[0]) * (offset / length)
            ay = wall.start[1] + (wall.end[1] - wall.start[1]) * (offset / length)
            # nudge inward to find the room
            nx = -(wall.end[1] - wall.start[1]) / length
            ny = (wall.end[0] - wall.start[0]) / length
            host = None
            for sign in (1, -1):
                probe = (ax + nx * sign * 0.5, ay + ny * sign * 0.5)
                host = next((r for r in rooms if point_in_polygon(probe, r.polygon)), None)
                if host:
                    break
            if host is None:
                continue
            is_entrance_wall = wall.id.endswith("0")
            if is_entrance_wall and i == 0:
                continue  # keep the entrance bay clear for the front door
            wall.openings.append(Opening(
                id=f"{wall.id}_w{i}", kind=OpeningKind.WINDOW, offset_m=round(offset, 2),
                width_m=round(min(1.6, length / (count + 1)), 2), height_m=1.3,
                sill_m=0.95, connects=[host.id],
                source=SourceType.INFERRED, confidence=INFERRED_CONFIDENCE,
            ))

    # Entrance door on the ground-floor south wall only: an upper storey cannot
    # have a front door. This is a structural rule, not a preference.
    south = exterior[0] if (exterior and level_index == 0) else None
    hall = next((r for r in rooms if r.type is RoomType.HALL), rooms[0])
    if south is not None:
        cx = sum(p[0] for p in hall.polygon) / len(hall.polygon)
        offset = max(0.9, min(south.length_m - 0.9, abs(cx - south.start[0])))
        south.openings = [o for o in south.openings if abs(o.offset_m - offset) > 1.2]
        south.openings.append(Opening(
            id=f"{south.id}_entry", kind=OpeningKind.DOOR, offset_m=round(offset, 2),
            width_m=1.0, height_m=2.1, sill_m=0.0, connects=[hall.id], is_entrance=True,
            source=SourceType.INFERRED, confidence=INFERRED_CONFIDENCE,
        ))

    # One door per shared room boundary, so the connectivity graph is real.
    connection = {
        (a, b): (axis, coord, lo, hi)
        for a, b, axis, coord, lo, hi in _shared_boundaries(rooms)
    }
    for wall in interior:
        if wall.length_m < 1.0:
            continue
        twin = _boundary_for_wall(wall, rooms, connection)
        if twin is None:
            continue
        room_a, room_b = twin
        wall.openings.append(Opening(
            id=f"{wall.id}_door", kind=OpeningKind.DOOR,
            offset_m=round(wall.length_m / 2.0, 2),
            width_m=round(min(0.9, wall.length_m - 0.2), 2), height_m=2.05,
            connects=[room_a, room_b],
            source=SourceType.INFERRED, confidence=INFERRED_CONFIDENCE,
        ))


def _boundary_for_wall(wall: Wall, rooms: list[Room], connection: dict) -> tuple[str, str] | None:
    """Which two rooms does this partition separate?"""
    for (a_id, b_id), (axis, coord, lo, hi) in connection.items():
        if axis == "v" and abs(wall.start[0] - coord) < 1e-6 and abs(wall.start[1] - lo) < 1e-6:
            return (a_id, b_id)
        if axis == "h" and abs(wall.start[1] - coord) < 1e-6 and abs(wall.start[0] - lo) < 1e-6:
            return (a_id, b_id)
    return None
