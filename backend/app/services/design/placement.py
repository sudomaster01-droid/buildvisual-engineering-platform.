"""Deterministic furniture placement with clearance and circulation rules.

Blueprint p.46: furniture must be plausible, must not block door openings, must
keep walkable circulation, must not intersect walls, and its placement must be
recordable so a later rectification changes staging without touching geometry.

The algorithm is a scored grid search — deterministic, auditable, and it fails
closed: an item that cannot be placed legally is dropped with a reason, never
forced into a wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..spec.geometry_helpers import point_in_polygon

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1

CLEARANCE_WALL = 0.06          # gap between furniture and wall face
CLEARANCE_DOOR = 1.10          # depth of the zone kept clear in front of a door
CLEARANCE_ITEM = 0.12          # minimum gap between two items
CIRCULATION_WIDTH = 0.80       # reserved walking corridor width
GRID_STEP = 0.20


@dataclass
class DoorZone:
    point: tuple[float, float]
    width: float
    inner_normal: tuple[float, float]


@dataclass
class Placement:
    asset_key: str
    position: tuple[float, float]  # centre in world XY
    rotation_deg: float
    footprint: tuple[float, float]
    z0: float = 0.0
    zone: str = "floor"  # floor | wall | ceiling

    @property
    def rect(self) -> Rect:
        half_w, half_d = self.footprint[0] / 2.0, self.footprint[1] / 2.0
        if round(self.rotation_deg) % 180 == 90:
            half_w, half_d = half_d, half_w
        return (
            self.position[0] - half_w, self.position[1] - half_d,
            self.position[0] + half_w, self.position[1] + half_d,
        )


@dataclass
class RoomLayout:
    room_id: str
    room_type: str
    placements: list[Placement] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    blocked: list[Rect] = field(default_factory=list)
    occupied: list[Rect] = field(default_factory=list)

    def is_free(self, rect: Rect, ignore: tuple[int, ...] = ()) -> bool:
        for b in self.blocked:
            if _overlaps(rect, b, 0.0):
                return False
        for index, occ in enumerate(self.occupied):
            if index in ignore:
                continue
            if _overlaps(rect, occ, CLEARANCE_ITEM):
                return False
        return True


def _overlaps(a: Rect, b: Rect, margin: float) -> bool:
    return not (
        a[2] + margin <= b[0] or b[2] + margin <= a[0]
        or a[3] + margin <= b[1] or b[3] + margin <= a[1]
    )


def _expand(rect: Rect, margin: float) -> Rect:
    return (rect[0] - margin, rect[1] - margin, rect[2] + margin, rect[3] + margin)


def room_bounds(polygon: list[tuple[float, float]]) -> Rect:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def door_blocked_zones(
    polygon: list[tuple[float, float]],
    doors: list[tuple[tuple[float, float], float]],
) -> list[Rect]:
    """Keep a clear approach zone inside the room in front of every door."""
    from ..spec.geometry_helpers import polygon_centroid

    cx, cy = polygon_centroid(polygon)
    zones: list[Rect] = []
    for point, width in doors:
        dx, dy = cx - point[0], cy - point[1]
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = dx / norm, dy / norm
        depth = CLEARANCE_DOOR
        reach = max(width, 0.9)
        x0 = min(point[0], point[0] + nx * depth) - reach / 2
        x1 = max(point[0], point[0] + nx * depth) + reach / 2
        y0 = min(point[1], point[1] + ny * depth) - reach / 2
        y1 = max(point[1], point[1] + ny * depth) + reach / 2
        zones.append((x0, y0, x1, y1))
    return zones


def circulation_corridor(
    polygon: list[tuple[float, float]],
    doors: list[tuple[tuple[float, float], float]],
    centroid: tuple[float, float] | None = None,
) -> list[Rect]:
    """Reserve a walkable strip from each door toward the room centre."""
    from ..spec.geometry_helpers import polygon_centroid

    if centroid is None:
        cx, cy = polygon_centroid(polygon)
    else:
        cx, cy = centroid
    corridors: list[Rect] = []
    for point, _width in doors:
        half = CIRCULATION_WIDTH / 2
        corridors.append((
            min(point[0], cx) - half, min(point[1], cy) - half,
            max(point[0], cx) + half, max(point[1], cy) + half,
        ))
    return corridors


def place_items(
    *,
    room_id: str,
    room_type: str,
    polygon: list[tuple[float, float]],
    doors: list[tuple[tuple[float, float], float]],
    plan: list[tuple[str, tuple[float, float], str]],  # (asset key, size, zone)
    ceiling_height: float,
) -> RoomLayout:
    """Place the requested items in priority order, skipping illegal positions."""
    layout = RoomLayout(room_id=room_id, room_type=room_type)
    layout.blocked.extend(door_blocked_zones(polygon, doors))
    layout.blocked.extend(circulation_corridor(polygon, doors))

    x0, y0, x1, y1 = room_bounds(polygon)
    for asset_key, size, zone in plan:
        width, depth = size
        best: tuple[float, Placement] | None = None
        for rotation in (0.0, 90.0, 180.0, 270.0):
            half_w, half_d = (width / 2, depth / 2) if rotation % 180 == 0 else (depth / 2, width / 2)
            # Wall-hugging items scan the perimeter first; centre items scan inward.
            xs = _axis_candidates(x0, x1, half_w, zone)
            ys = _axis_candidates(y0, y1, half_d, zone)
            for cx in xs:
                for cy in ys:
                    rect = (cx - half_w, cy - half_d, cx + half_w, cy + half_d)
                    if not _inside_room(rect, polygon):
                        continue
                    if not layout.is_free(rect):
                        continue
                    score = _score(rect, rotation, polygon, zone, x0, y0, x1, y1)
                    if best is None or score > best[0]:
                        best = (score, Placement(
                            asset_key=asset_key,
                            position=(round(cx, 3), round(cy, 3)),
                            rotation_deg=rotation,
                            footprint=(width, depth),
                            z0=0.0,
                            zone="ceiling" if zone == "ceiling" else "floor",
                        ))
        if best is None:
            layout.skipped.append({
                "asset_key": asset_key,
                "reason": "no legal position with the required clearances",
            })
            continue
        layout.placements.append(best[1])
        layout.occupied.append(best[1].rect)
    return layout


def _axis_candidates(lo: float, hi: float, half: float, zone: str) -> list[float]:
    start = lo + half + CLEARANCE_WALL
    end = hi - half - CLEARANCE_WALL
    if end <= start:
        return []
    steps = int((end - start) / GRID_STEP) + 1
    values = [round(start + i * GRID_STEP, 3) for i in range(steps + 1)]
    if zone == "wall":
        # bias scanning toward the perimeter (first and last rows already are)
        values.sort()
    return values


def _inside_room(rect: Rect, polygon: list[tuple[float, float]]) -> bool:
    corners = [
        (rect[0], rect[1]), (rect[2], rect[1]), (rect[2], rect[3]), (rect[0], rect[3]),
        ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2),
    ]
    return all(point_in_polygon(c, polygon) for c in corners)


def _score(
    rect: Rect, rotation: float, polygon, zone: str,
    rx0: float, ry0: float, rx1: float, ry1: float,
) -> float:
    """Higher is better. Rewards wall adjacency, punishes door proximity."""
    score = 0.0
    if zone == "wall":
        gaps = [
            abs(rect[0] - rx0), abs(rx1 - rect[2]),
            abs(rect[1] - ry0), abs(ry1 - rect[3]),
        ]
        score -= min(gaps) * 3.0
        # Facing the room centre is preferred over facing a corner.
        cx, cy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
        ix = (rect[0] + rect[2]) / 2
        iy = (rect[1] + rect[3]) / 2
        score -= 0.02 * (abs(ix - cx) + abs(iy - cy))
    else:
        centre_rect = ((rx0 + rx1) / 2, (ry0 + ry1) / 2)
        ix = (rect[0] + rect[2]) / 2
        iy = (rect[1] + rect[3]) / 2
        score -= math.hypot(ix - centre_rect[0], iy - centre_rect[1]) * 2.0
    # Prefer axis-aligned rotations for tidiness.
    if rotation % 90 == 0:
        score += 0.5
    return score
