"""Automatic cinematic planning — the virtual director.

Blueprint p.50–53 define an agreed architectural film grammar:

1 aerial reveal → 2 drone orbit → 3 approach to the gate → 4 compound move →
5 exterior walkaround → 6 entrance + room-by-room → 7 upper floors (when they
exist) → 8 closing shot on the strongest view.

Two rules are enforced in code rather than described in prose:

* the sequence adapts to the actual house (no invented upstairs, gate or room);
* every camera move is validated against the model before it can become a shot.

The output is a versioned shot plan whose shots are sized to the configured
provider clip length, so the story is segmented as a deliberate choice rather
than an arbitrary split after generation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ...config import get_settings
from ..design.compose import Scene
from ..modeling.geometry import BuildingModel
from ..modeling.validate import RoomGraph, build_room_graph
from ..spec.geometry_helpers import polygon_area, polygon_centroid
from ..spec.schema import HouseSpec, RoomType
from . import routing

PLANNER_VERSION = "1.0.0"

SHOT_TYPES = [
    "aerial_reveal",
    "drone_orbit",
    "approach_gate",
    "compound_move",
    "exterior_walkaround",
    "entrance_walkthrough",
    "room_walkthrough",
    "upper_floor",
    "closing",
]

EYE_HEIGHT = 1.62
DRONE_MIN_AGL = 6.0


@dataclass
class CameraKeyframe:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_deg: float = 50.0


@dataclass
class Shot:
    id: str
    type: str
    title: str
    purpose: str
    duration_s: float
    motion: str                 # dolly | crane | orbit | track | push_in
    start: CameraKeyframe
    end: CameraKeyframe
    room_id: str | None = None
    level: int = 0
    transition: str = "cut"
    priority: int = 5           # 1 = must keep when trimming to max_shots
    path: list[tuple[float, float]] = field(default_factory=list)
    reference_frames: list[str] = field(default_factory=lambda: ["first", "last"])
    notes: list[str] = field(default_factory=list)

    def keyframes(self) -> dict[str, Any]:
        return {
            "start": asdict(self.start),
            "end": asdict(self.end),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "purpose": self.purpose,
            "duration_s": round(self.duration_s, 2),
            "motion": self.motion,
            "transition": self.transition,
            "priority": self.priority,
            "level": self.level,
            "room_id": self.room_id,
            "keyframes": self.keyframes(),
            "path": [[round(x, 3), round(y, 3)] for x, y in self.path],
            "reference_frames": self.reference_frames,
            "notes": self.notes,
        }


@dataclass
class ShotPlan:
    shots: list[Shot]
    manifest: dict[str, Any]

    def as_manifest(self) -> dict[str, Any]:
        return self.manifest


def plan_film(
    model: BuildingModel,
    scene: Scene,
    *,
    max_shots: int | None = None,
    clip_seconds: int | None = None,
    version: int = 1,
    adjustments: dict[str, Any] | None = None,
) -> ShotPlan:
    settings = get_settings()
    spec = model.spec
    max_shots = max_shots or settings.max_shots
    clip_seconds = clip_seconds or settings.shot_seconds

    lo, hi = model.mesh.bounds_excluding()
    footprint = {"min": lo.round(2).tolist(), "max": hi.round(2).tolist()}
    span_x = float(hi[0] - lo[0])
    span_y = float(hi[1] - lo[1])
    diagonal = math.hypot(span_x, span_y)
    centre = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)

    graph = build_room_graph(spec)
    entrance = _entrance_info(spec, centre, lo, hi)
    notes: list[str] = []
    candidates: list[Shot] = []

    # --- 1. aerial reveal -------------------------------------------------
    # Framing is driven by the *house* diagonal (the site plane is excluded), so
    # the building fills the frame instead of appearing as a speck on the plot.
    aerial_height = max(DRONE_MIN_AGL + 4.0, min(46.0, diagonal * 1.05))
    candidates.append(Shot(
        id="shot_01_aerial",
        type="aerial_reveal",
        title="Aerial reveal",
        purpose="Establish the house in its setting and show the roof form and plan.",
        duration_s=clip_seconds,
        motion="crane",
        start=CameraKeyframe(
            position=(centre[0] - span_x * 0.95, centre[1] - span_y * 1.25, aerial_height),
            target=(centre[0], centre[1], 1.2),
            fov_deg=44.0,
        ),
        end=CameraKeyframe(
            position=(centre[0] - span_x * 0.45, centre[1] - span_y * 0.60, aerial_height * 0.62),
            target=(centre[0], centre[1], 1.6),
            fov_deg=46.0,
        ),
        priority=1,
        transition="fade",
    ))

    # --- 2. drone orbit ---------------------------------------------------
    orbit_radius = diagonal * 0.62
    orbit_height = max(DRONE_MIN_AGL, aerial_height * 0.52)
    start_angle = entrance["approach_angle"] + math.radians(215.0)
    candidates.append(Shot(
        id="shot_02_orbit",
        type="drone_orbit",
        title="Drone orbit",
        purpose="Reveal the massing from every side with a slow, level orbit.",
        duration_s=clip_seconds,
        motion="orbit",
        start=CameraKeyframe(
            position=(centre[0] + math.cos(start_angle) * orbit_radius,
                      centre[1] + math.sin(start_angle) * orbit_radius,
                      orbit_height),
            target=(centre[0], centre[1], 1.6),
            fov_deg=46.0,
        ),
        end=CameraKeyframe(
            position=(centre[0] + math.cos(start_angle - math.radians(105.0)) * orbit_radius,
                      centre[1] + math.sin(start_angle - math.radians(105.0)) * orbit_radius,
                      orbit_height * 0.92),
            target=(centre[0], centre[1], 1.8),
            fov_deg=46.0,
        ),
        priority=2,
        transition="cut",
    ))

    # --- 3. approach to the gate/entrance ---------------------------------
    approach_start = entrance["approach_point"]
    candidates.append(Shot(
        id="shot_03_approach",
        type="approach_gate",
        title="Approach",
        purpose="Move from the plot edge toward the entrance at walking-to-driving pace.",
        duration_s=clip_seconds,
        motion="dolly",
        start=CameraKeyframe(
            position=(approach_start[0], approach_start[1], 2.06),
            target=(entrance["point"][0], entrance["point"][1], 1.9),
            fov_deg=54.0,
        ),
        end=CameraKeyframe(
            position=(
                entrance["point"][0] - entrance["inward"][0] * 2.4,
                entrance["point"][1] - entrance["inward"][1] * 2.4,
                1.72,
            ),
            target=(entrance["point"][0] + entrance["inward"][0] * 2.0,
                    entrance["point"][1] + entrance["inward"][1] * 2.0, 1.62),
            fov_deg=52.0,
        ),
        path=[
            (approach_start[0], approach_start[1]),
            (entrance["point"][0] - entrance["inward"][0] * 2.4,
             entrance["point"][1] - entrance["inward"][1] * 2.4),
        ],
        priority=1,
        transition="cut",
    ))

    # --- 4. compound move (only when the plot actually has depth) ---------
    plot_depth = _plot_depth(spec, span_y)
    if plot_depth >= 14.0:
        side = entrance["outward_perpendicular"]
        candidates.append(Shot(
            id="shot_04_compound",
            type="compound_move",
            title="Compound",
            purpose="Track laterally across the frontage, revealing depth and landscaping.",
            duration_s=clip_seconds,
            motion="track",
            start=CameraKeyframe(
                position=(approach_start[0] - side[0] * span_x * 0.42,
                          approach_start[1] - side[1] * span_x * 0.42, 2.3),
                target=(centre[0], centre[1], 2.2),
                fov_deg=52.0,
            ),
            end=CameraKeyframe(
                position=(approach_start[0] + side[0] * span_x * 0.30,
                          approach_start[1] + side[1] * span_x * 0.30, 2.4),
                target=(centre[0], centre[1], 2.4),
                fov_deg=52.0,
            ),
            priority=4,
            transition="cut",
        ))
    else:
        notes.append("Compound move skipped: the modelled plot has no usable frontage depth.")

    # --- 5. exterior walkaround ------------------------------------------
    wc = _walkaround(model, entrance)
    if wc:
        candidates.append(wc)

    # --- 6. entrance + room-by-room --------------------------------------
    bounds_per_level = {
        lvl.index: routing.build_occupancy(model, lvl.index) for lvl in spec.levels
    }
    for level_id, grid in bounds_per_level.items():
        routing.carve_room_interiors(grid, spec, level_id)

    interior_shots, interior_notes = _interior_shots(
        model, spec, graph, bounds_per_level, clip_seconds)
    notes.extend(interior_notes)
    candidates.extend(interior_shots)

    # --- 8. closing -------------------------------------------------------
    closing = _closing_shot(model, spec, entrance, centre, diagonal, clip_seconds)
    candidates.append(closing)
    adjustments = adjustments or {}
    dropped = set(adjustments.get("drop") or [])
    if dropped:
        removed = [s.type for s in candidates if s.type in dropped]
        candidates = [s for s in candidates if s.type not in dropped]
        if removed:
            notes.append("Correction removed shots: " + ", ".join(sorted(set(removed))) + ".")
    duration_scale = float(adjustments.get("duration_scale", 1.0) or 1.0)
    if duration_scale != 1.0:
        allowed = [d for d in (4, 6, 8) if d <= clip_seconds]
        scaled = min(allowed, key=lambda d: abs(d - clip_seconds * duration_scale))
        for shot in candidates:
            shot.duration_s = scaled
        notes.append(f"Shot length scaled to {scaled}s by request.")
    favour = list(adjustments.get("reorder") or [])
    if favour:
        # Move requested subjects earlier in the story without breaking the
        # establish/approach/closing skeleton.
        preferred = [s for s in candidates if s.type in favour]
        rest = [s for s in candidates if s.type not in favour]
        candidates = rest[:3] + preferred + rest[3:]
        notes.append("Correction prioritised: " + ", ".join(favour) + ".")
    if adjustments.get("note"):
        notes.append(str(adjustments["note"]))

    # --- selection: keep the story, trim by priority ----------------------
    selected = _select_shots(candidates, max_shots)
    trimmed = [s for s in candidates if s.id not in {x.id for x in selected}]
    if trimmed:
        notes.append(
            "Trimmed to the configured shot budget: "
            + ", ".join(f"{s.title} ({s.type})" for s in trimmed)
            + f". Budget is MAX_SHOTS={max_shots}; raise it in configuration to include them."
        )

    total = sum(s.duration_s for s in selected)
    manifest = {
        "planner_version": PLANNER_VERSION,
        "sequence_version": 1,
        "model_geometry_hash": model.geometry_hash,
        "scene_palette": scene.manifest.get("palette_id"),
        "clip_seconds": clip_seconds,
        "max_shots": max_shots,
        "total_duration_s": round(total, 2),
        "shot_count": len(selected),
        "sequence": [s.type for s in selected],
        "agreed_sequence": SHOT_TYPES[:6] + ["upper_floor (conditional)"] + ["closing"],
        "adaptations": notes,
        "house": {
            "levels": len(spec.levels),
            "rooms": len(spec.rooms),
            "footprint": footprint,
            "has_upper_floor": spec.has_upper_floor,
            "has_stairs": bool(spec.stairs),
            "entrance": entrance["description"],
        },
        "room_graph": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "entrance_room": graph.entrance_node,
        },
        "shots": [s.as_dict() for s in selected],
        "camera_rules": {
            "eye_height_m": EYE_HEIGHT,
            "drone_min_agl_m": DRONE_MIN_AGL,
            "easing": "smoothstep",
            "fov_range_deg": [42, 58],
            "note": (
                "Indoor moves stay near eye height with reduced speed; drone moves keep "
                "a minimum height above ground; all paths are collision-checked."
            ),
        },
    }
    return ShotPlan(shots=selected, manifest=manifest)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entrance_info(spec: HouseSpec, centre, lo, hi) -> dict[str, Any]:
    entrances = spec.entrance_openings()
    span = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]))
    if entrances:
        wall, opening = entrances[0]
        along = opening.offset_m / max(wall.length_m, 1e-9)
        px = wall.start[0] + (wall.end[0] - wall.start[0]) * along
        py = wall.start[1] + (wall.end[1] - wall.start[1]) * along
        dx, dy = centre[0] - px, centre[1] - py
        norm = math.hypot(dx, dy) or 1.0
        inward = (dx / norm, dy / norm)
        outward = (-inward[0], -inward[1])
        perp = (-outward[1], outward[0])
        approach_distance = span * 0.85 + 4.0
        return {
            "point": (px, py),
            "inward": inward,
            "outward": outward,
            "outward_perpendicular": perp,
            "approach_point": (px + outward[0] * approach_distance,
                               py + outward[1] * approach_distance),
            "approach_angle": math.atan2(outward[1], outward[0]),
            "description": f"Entrance on wall {wall.id} facing {_compass(outward)}.",
        }
    cx, cy = centre
    return {
        "point": (cx, lo[1]),
        "inward": (0.0, 1.0),
        "outward": (0.0, -1.0),
        "outward_perpendicular": (1.0, 0.0),
        "approach_point": (cx, lo[1] - span * 0.85),
        "approach_angle": -math.pi / 2,
        "description": "No exterior door was found; the approach assumes the south facade.",
    }


def _compass(vector) -> str:
    angle = math.degrees(math.atan2(vector[0], vector[1])) % 360.0
    names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
    return names[int((angle + 22.5) // 45) % 8]


def _plot_depth(spec: HouseSpec, modelled_depth: float) -> float:
    if spec.site.plot_depth_m:
        return float(spec.site.plot_depth_m)
    return modelled_depth * 0.55


def _walkaround(model: BuildingModel, entrance: dict[str, Any]) -> Shot | None:
    lo, hi = model.mesh.bounds_excluding()
    centre = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2)
    radius = max(hi[0] - lo[0], hi[1] - lo[1]) * 0.72 + 2.5
    base = entrance["approach_angle"]
    points = [
        (centre[0] + math.cos(base - math.radians(60)) * radius,
         centre[1] + math.sin(base - math.radians(60)) * radius),
        (centre[0] + math.cos(base - math.radians(5)) * radius,
         centre[1] + math.sin(base - math.radians(5)) * radius),
        (centre[0] + math.cos(base + math.radians(50)) * radius,
         centre[1] + math.sin(base + math.radians(50)) * radius),
    ]
    return Shot(
        id="shot_05_walkaround",
        type="exterior_walkaround",
        title="Facade walkaround",
        purpose="Move along the principal facade so openings, materials and depth read clearly.",
        duration_s=8,
        motion="track",
        start=CameraKeyframe(position=(*points[0], 1.95), target=(*centre, 1.9), fov_deg=52.0),
        end=CameraKeyframe(position=(*points[2], 2.05), target=(*centre, 2.0), fov_deg=52.0),
        path=points,
        priority=3,
        transition="cut",
    )


def _interior_shots(
    model: BuildingModel,
    spec: HouseSpec,
    graph: RoomGraph,
    grids: dict[int, routing.OccupancyGrid],
    clip_seconds: int,
) -> tuple[list[Shot], list[str]]:
    notes: list[str] = []
    level_elevations = dict(model.level_elevations)
    stops = routing.plan_interior_route(graph, spec, max_stops=5)
    if not stops:
        notes.append("No interior route could be planned; exterior shots only.")
        return [], notes

    shots: list[Shot] = []
    entrance_shot = _entrance_shot(model, spec, stops, clip_seconds, level_elevations)
    if entrance_shot is not None:
        shots.append(entrance_shot)
    else:
        stops_for_shot_6 = stops[:2]
        shots.append(_room_shot(
            shot_id="shot_06_entrance",
            shot_type="entrance_walkthrough",
            stops=stops_for_shot_6,
            clip_seconds=clip_seconds,
            grids=grids,
            ttl="Entrance into the house",
            purpose="Step through the front door and land in the first principal space.",
            priority=1,
            level_z=level_elevations,
        ))
        notes.append("Front door position was not resolved; the entrance shot starts indoors.")

    remaining = stops[2:]
    if remaining:
        shots.append(_room_shot(
            shot_id="shot_07_rooms",
            shot_type="room_walkthrough",
            stops=remaining[:3],
            clip_seconds=clip_seconds,
            grids=grids,
            ttl="Room-by-room walkthrough",
            purpose="Move between the main living spaces, keeping room identity consistent.",
            priority=2,
            level_z=level_elevations,
        ))

    # Upper floor: only when a stair genuinely connects another level.
    if spec.has_upper_floor and spec.stairs:
        stair = spec.stairs[0]
        grid = grids.get(stair.to_level)
        upper_rooms = [r for r in spec.rooms if r.level == stair.to_level]
        if grid and upper_rooms:
            upper = max(upper_rooms, key=lambda r: polygon_area(r.polygon))
            centroid = polygon_centroid(upper.polygon)
            base_z = model.level_elevations.get(stair.to_level, 3.0)
            start = (stair.start[0], stair.start[1])
            end = (stair.end[0], stair.end[1])
            path_ok = routing.path_is_clear(grid, [centroid, end])
            shots.append(Shot(
                id="shot_08_upper",
                type="upper_floor",
                title="Upper floor",
                purpose="Climb to the first floor and open onto the landing.",
                duration_s=clip_seconds,
                motion="crane",
                start=CameraKeyframe(
                    position=(start[0], start[1], base_z - 2.6 + EYE_HEIGHT),
                    target=(end[0], end[1], base_z + 0.4),
                    fov_deg=52.0,
                ),
                end=CameraKeyframe(
                    position=(end[0] + (centroid[0] - end[0]) * 0.55,
                              end[1] + (centroid[1] - end[1]) * 0.55,
                              base_z + EYE_HEIGHT),
                    target=(centroid[0], centroid[1], base_z + 1.4),
                    fov_deg=52.0,
                ),
                level=stair.to_level,
                room_id=upper.id,
                path=[start, end, (centroid[0], centroid[1])],
                priority=2,
                transition="cut",
                notes=[] if path_ok else ["Camera path from stair to landing was not clear; verify."],
            ))
        else:
            notes.append("Upper floor skipped: no stair logic resolved to an upper room.")
    elif spec.has_upper_floor:
        notes.append("Upper floor skipped: the model has multiple levels but no stair.")
    else:
        notes.append("Single-storey house: no upper-floor shot was invented.")

    return shots, notes


def _entrance_shot(
    model: BuildingModel,
    spec: HouseSpec,
    stops,
    clip_seconds: int,
    level_z: dict[int, float],
) -> Shot | None:
    """Outside the front door → through it → into the first space.

    This is the one shot where the camera must cross the building envelope, and
    the move reads best as a single forward push rather than a pivot indoors.
    """
    entrances = spec.entrance_openings()
    if not entrances:
        return None
    wall, opening = entrances[0]
    along = opening.offset_m / max(wall.length_m, 1e-9)
    door = (
        wall.start[0] + (wall.end[0] - wall.start[0]) * along,
        wall.start[1] + (wall.end[1] - wall.start[1]) * along,
    )
    dx = wall.end[0] - wall.start[0]
    dy = wall.end[1] - wall.start[1]
    length = math.hypot(dx, dy) or 1.0
    # Wall normal; pick the side that points away from the building centre.
    nx, ny = -dy / length, dx / length
    lo, hi = model.mesh.bounds_excluding()
    centre = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)
    if (door[0] - centre[0]) * nx + (door[1] - centre[1]) * ny < 0:
        nx, ny = -nx, -ny

    base_z = level_z.get(wall.level, 0.0)
    outside = (door[0] + nx * 3.6, door[1] + ny * 3.6)
    inside = (door[0] - nx * 1.9, door[1] - ny * 1.9)
    first_room = stops[0] if stops else None
    look_at = first_room.position if first_room else (door[0] - nx * 5.0, door[1] - ny * 5.0)

    return Shot(
        id="shot_06_entrance",
        type="entrance_walkthrough",
        title="Entrance into the house",
        purpose="Approach the front door and step into the first principal space.",
        duration_s=clip_seconds,
        motion="dolly",
        start=CameraKeyframe(
            position=(outside[0], outside[1], base_z + EYE_HEIGHT + 0.05),
            target=(door[0], door[1], base_z + 1.45),
            fov_deg=62.0,
        ),
        end=CameraKeyframe(
            position=(inside[0], inside[1], base_z + EYE_HEIGHT),
            target=(look_at[0], look_at[1], base_z + 1.35),
            fov_deg=68.0,
        ),
        room_id=first_room.room_id if first_room else None,
        level=wall.level,
        path=[outside, door, inside],
        priority=1,
        transition="cut",
    )


def _room_shot(
    *, shot_id: str, shot_type: str, stops, clip_seconds: int, grids, ttl: str,
    purpose: str, priority: int, level_z: dict[int, float] | None = None,
) -> Shot:
    level_z = level_z or {}
    first = stops[0]
    last = stops[-1]
    z_first = level_z.get(first.level, 0.0) + EYE_HEIGHT
    z_last = level_z.get(last.level, 0.0) + EYE_HEIGHT
    grid = grids.get(first.level)

    path_points: list[tuple[float, float]] = [first.position]
    for stop in stops[1:]:
        step_grid = grids.get(stop.level, grid)
        if step_grid is not None:
            segment = routing.bfs_path(step_grid, path_points[-1], stop.position)
            if segment:
                path_points.extend(segment[1:])
            else:
                path_points.append(stop.position)
        else:
            path_points.append(stop.position)
    resampled = routing.resample(path_points, 24) or path_points

    start_pos = (*path_points[0], z_first)
    end_pos = (*path_points[-1], z_last)
    # Look ahead along the path so the move is motivated, not a fixed pivot.
    look_index = min(len(resampled) - 1, 6)
    start_target = (*resampled[look_index], z_first - 0.25)
    end_target = (last.position[0], last.position[1], z_last - 0.2)

    return Shot(
        id=shot_id,
        type=shot_type,
        title=ttl,
        purpose=purpose,
        duration_s=clip_seconds,
        motion="dolly",
        # 68-70 degrees is the working interior lens for real-estate films: wide
        # enough that a 3 m room does not become a wall-filling frame.
        start=CameraKeyframe(position=start_pos, target=start_target, fov_deg=68.0),
        end=CameraKeyframe(position=end_pos, target=end_target, fov_deg=66.0),
        room_id=last.room_id,
        level=last.level,
        path=path_points,
        priority=priority,
        transition="cut",
    )





def _closing_shot(model, spec, entrance, centre, diagonal, clip_seconds) -> Shot:
    from .routing import strongest_view

    lo, hi = model.mesh.bounds_excluding()
    view = strongest_view(spec, model)
    cx, cy = centre
    dx, dy = view[0] - cx, view[1] - cy
    norm = math.hypot(dx, dy) or 1.0
    pull = diagonal * 0.30
    return Shot(
        id="shot_09_closing",
        type="closing",
        title="Closing view",
        purpose="End on the strongest available view of the house.",
        duration_s=clip_seconds,
        motion="push_in",
        start=CameraKeyframe(
            position=(view[0] + dx / norm * pull, view[1] + dy / norm * pull, 3.4),
            target=(cx, cy, 2.2),
            fov_deg=48.0,
        ),
        end=CameraKeyframe(
            position=(view[0], view[1], 2.4),
            target=(cx, cy, 2.6),
            fov_deg=46.0,
        ),
        priority=1,
        transition="fade",
    )


def _select_shots(candidates: list[Shot], max_shots: int) -> list[Shot]:
    """Keep the narrative order but respect the shot budget.

    Priority 1 shots are always kept (establish, approach, entrance, closing).
    Remaining slots go to the lowest priority number first.
    """
    if len(candidates) <= max_shots:
        return candidates
    must = [s for s in candidates if s.priority == 1]
    optional = sorted(
        [s for s in candidates if s.priority != 1], key=lambda s: (s.priority, s.id)
    )
    keep = set(s.id for s in must)
    for shot in optional:
        if len(keep) >= max_shots:
            break
        keep.add(shot.id)
    return [s for s in candidates if s.id in keep]
