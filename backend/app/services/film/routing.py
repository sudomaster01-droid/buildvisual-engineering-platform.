"""Camera path planning and collision validation.

Blueprint p.51/p.53: rooms are graph nodes, passable openings are edges, and a
camera path must be validated against the model before references are rendered —
"do not place a camera through a wall".

The occupancy grid here is built from the wall segments of the locked model, so
a route that does not fit the house cannot silently become a shot.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..modeling.geometry import BuildingModel
from ..modeling.validate import RoomGraph
from ..spec.geometry_helpers import point_in_polygon, polygon_centroid
from ..spec.schema import HouseSpec, Room, RoomType

GRID_RESOLUTION = 0.15
CAMERA_RADIUS = 0.38      # keeps the lens off walls
DOOR_RADIUS = 0.30

Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


@dataclass
class OccupancyGrid:
    """2-D obstruction grid for one floor level."""

    origin: tuple[float, float]
    resolution: float
    blocked: np.ndarray

    def world_to_cell(self, point: Point2) -> tuple[int, int]:
        col = int((point[0] - self.origin[0]) / self.resolution)
        row = int((point[1] - self.origin[1]) / self.resolution)
        return row, col

    def cell_to_world(self, row: int, col: int) -> Point2:
        return (
            self.origin[0] + (col + 0.5) * self.resolution,
            self.origin[1] + (row + 0.5) * self.resolution,
        )

    def is_blocked(self, point: Point2) -> bool:
        row, col = self.world_to_cell(point)
        if row < 0 or col < 0 or row >= self.blocked.shape[0] or col >= self.blocked.shape[1]:
            return True
        return bool(self.blocked[row, col])


def build_occupancy(model: BuildingModel, level: int, margin: float = CAMERA_RADIUS) -> OccupancyGrid:
    spec = model.spec
    lo, hi = model.mesh.bounds
    resolution = GRID_RESOLUTION
    origin = (float(lo[0]) - 1.0, float(lo[1]) - 1.0)
    width = int((hi[0] - lo[0] + 2.0) / resolution) + 1
    height = int((hi[1] - lo[1] + 2.0) / resolution) + 1
    blocked = np.zeros((height, width), dtype=bool)

    def mark(wall, radius: float, skip_openings: bool) -> None:
        ax, ay = wall.start
        bx, by = wall.end
        length = math.hypot(bx - ax, by - ay)
        if length < 1e-6:
            return
        ux, uy = (bx - ax) / length, (by - ay) / length
        # openings we may pass through
        openings = sorted(wall.openings, key=lambda o: o.offset_m)
        segments: list[tuple[float, float]] = []
        cursor = 0.0
        if skip_openings:
            for opening in openings:
                if opening.kind.value == "window":
                    continue
                if opening.width_m < 2 * radius + 0.1:
                    continue
                start = opening.offset_m - opening.width_m / 2 + (radius * 0.8)
                end = opening.offset_m + opening.width_m / 2 - (radius * 0.8)
                if start > cursor:
                    segments.append((cursor, start))
                cursor = max(cursor, end)
        segments.append((cursor, length))

        for start, end in segments:
            steps = max(2, int((end - start) / (resolution * 0.6)))
            for i in range(steps + 1):
                d = start + (end - start) * i / steps
                cx, cy = ax + ux * d, ay + uy * d
                _stamp_capsule(blocked, origin, resolution, cx, cy, radius)

    for wall in spec.walls:
        if wall.level != level:
            continue
        radius = wall.thickness_m / 2 + margin
        mark(wall, radius, skip_openings=True)

    return OccupancyGrid(origin=origin, resolution=resolution, blocked=blocked)


def _stamp_capsule(blocked: np.ndarray, origin, resolution: float, cx: float, cy: float, radius: float) -> None:
    col = int((cx - origin[0]) / resolution)
    row = int((cy - origin[1]) / resolution)
    reach = int(math.ceil(radius / resolution))
    rows = slice(max(0, row - reach), min(blocked.shape[0], row + reach + 1))
    cols = slice(max(0, col - reach), min(blocked.shape[1], col + reach + 1))
    blocked[rows, cols] = True


def carve_room_interiors(grid: OccupancyGrid, spec: HouseSpec, level: int) -> None:
    """Walls are stamped as thick capsules; re-open the inside of each room."""
    for room in spec.rooms:
        if room.level != level:
            continue
        xs = [p[0] for p in room.polygon]
        ys = [p[1] for p in room.polygon]
        x0, x1 = min(xs) + 0.45, max(xs) - 0.45
        y0, y1 = min(ys) + 0.45, max(ys) - 0.45
        if x1 <= x0 or y1 <= y0:
            continue
        r0, c0 = grid.world_to_cell((x0, y0))
        r1, c1 = grid.world_to_cell((x1, y1))
        for row in range(max(0, r0), min(grid.blocked.shape[0], r1 + 1)):
            for col in range(max(0, c0), min(grid.blocked.shape[1], c1 + 1)):
                wx, wy = grid.cell_to_world(row, col)
                if point_in_polygon((wx, wy), room.polygon):
                    # keep a thin skin along the walls
                    if _distance_to_boundary((wx, wy), room.polygon) > CAMERA_RADIUS + 0.08:
                        grid.blocked[row, col] = False


def _distance_to_boundary(point, polygon) -> float:
    from ..spec.geometry_helpers import distance_point_to_segment

    poly = list(polygon)
    n = len(poly)
    return min(
        distance_point_to_segment(point, poly[i], poly[(i + 1) % n]) for i in range(n)
    )


def path_is_clear(grid: OccupancyGrid, points: list[Point2]) -> bool:
    for i in range(len(points) - 1):
        if not segment_is_clear(grid, points[i], points[i + 1]):
            return False
    return True


def segment_is_clear(grid: OccupancyGrid, a: Point2, b: Point2) -> bool:
    distance = math.hypot(b[0] - a[0], b[1] - a[1])
    steps = max(2, int(distance / (grid.resolution * 0.5)))
    for i in range(steps + 1):
        t = i / steps
        point = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        if grid.is_blocked(point):
            return False
    return True


def bfs_path(grid: OccupancyGrid, start: Point2, goal: Point2) -> list[Point2]:
    """Shortest clear path on the occupancy grid (4-connected, deterministic)."""
    start_cell = grid.world_to_cell(start)
    goal_cell = grid.world_to_cell(goal)
    if grid.is_blocked(start):
        start_cell = _nearest_free(grid, start_cell)
    if grid.is_blocked(goal):
        goal_cell = _nearest_free(grid, goal_cell)
    if start_cell is None or goal_cell is None:
        return []
    queue = deque([start_cell])
    came: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
    while queue:
        cell = queue.popleft()
        if cell == goal_cell:
            break
        row, col = cell
        for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + drow, col + dcol
            if not (0 <= nr < grid.blocked.shape[0] and 0 <= nc < grid.blocked.shape[1]):
                continue
            if grid.blocked[nr, nc]:
                continue
            if (nr, nc) in came:
                continue
            came[(nr, nc)] = cell
            queue.append((nr, nc))
    if goal_cell not in came:
        return []
    cells = [goal_cell]
    while cells[-1] != start_cell:
        previous = came[cells[-1]]
        if previous is None:
            break
        cells.append(previous)
    cells.reverse()
    points = [grid.cell_to_world(r, c) for r, c in cells]
    return _simplify(points)


def _nearest_free(grid: OccupancyGrid, cell: tuple[int, int], max_radius: int = 40):
    row, col = cell
    if not grid.blocked[row, col] if (
        0 <= row < grid.blocked.shape[0] and 0 <= col < grid.blocked.shape[1]
    ) else False:
        return cell
    for radius in range(1, max_radius):
        for drow in range(-radius, radius + 1):
            for dcol in range(-radius, radius + 1):
                if max(abs(drow), abs(dcol)) != radius:
                    continue
                nr, nc = row + drow, col + dcol
                if 0 <= nr < grid.blocked.shape[0] and 0 <= nc < grid.blocked.shape[1]:
                    if not grid.blocked[nr, nc]:
                        return (nr, nc)
    return None


def _simplify(points: list[Point2], tolerance: float = 0.35) -> list[Point2]:
    """Drop collinear/interior points so interpolated camera moves stay smooth."""
    if len(points) <= 2:
        return points
    out = [points[0]]
    for point in points[1:-1]:
        if math.hypot(point[0] - out[-1][0], point[1] - out[-1][1]) >= tolerance:
            out.append(point)
    out.append(points[-1])
    return out


def resample(points: list[Point2], count: int) -> list[Point2]:
    """Evenly resample a polyline to ``count`` points (arc-length parameterised)."""
    if not points:
        return []
    if len(points) == 1:
        return [points[0]] * count
    lengths = [0.0]
    for i in range(1, len(points)):
        lengths.append(lengths[-1] + math.hypot(
            points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]))
    total = lengths[-1]
    if total < 1e-6:
        return [points[0]] * count
    out = []
    for i in range(count):
        target = total * i / max(1, count - 1)
        for j in range(1, len(lengths)):
            if lengths[j] >= target - 1e-9:
                span = lengths[j] - lengths[j - 1]
                t = 0.0 if span < 1e-9 else (target - lengths[j - 1]) / span
                out.append((
                    points[j - 1][0] + (points[j][0] - points[j - 1][0]) * t,
                    points[j - 1][1] + (points[j][1] - points[j - 1][1]) * t,
                ))
                break
        else:
            out.append(points[-1])
    return out


# ---------------------------------------------------------------------------
# Room route planning
# ---------------------------------------------------------------------------


@dataclass
class RouteStop:
    room_id: str
    room_name: str
    room_type: str
    level: int
    position: Point2
    entering_via: str | None = None
    path: list[Point2] = field(default_factory=list)


def plan_interior_route(graph: RoomGraph, spec: HouseSpec, *, max_stops: int = 6) -> list[RouteStop]:
    """Greedy coverage route over the room graph starting at the entrance.

    Rooms that cannot be reached keep their edge out of the route (and the
    validator has already warned about them) — the camera is never teleported
    through a wall.
    """
    if not graph.nodes:
        return []
    start = graph.entrance_node or next(iter(graph.nodes))
    priority = {
        RoomType.LIVING: 0, RoomType.DINING: 1, RoomType.KITCHEN: 2,
        RoomType.HALL: 3, RoomType.BEDROOM: 4, RoomType.STUDY: 5,
        RoomType.BATHROOM: 8, RoomType.UTILITY: 9, RoomType.GARAGE: 10,
    }
    visited: list[str] = []
    current = start
    stops: list[RouteStop] = []
    while current and len(stops) < max_stops:
        node = graph.nodes[current]
        stops.append(RouteStop(
            room_id=node.id, room_name=node.name, room_type=node.type,
            level=node.level, position=node.centroid,
        ))
        visited.append(current)
        candidates = [
            (nid, edge) for nid, edge in graph.neighbours(current) if nid not in visited
        ]
        if not candidates:
            # allow revisiting a corridor room to reach an unvisited branch
            reachable_unvisited = [
                nid for nid in graph.nodes
                if nid not in visited and graph.shortest_path(current, nid)
            ]
            if not reachable_unvisited:
                break
            target = min(
                reachable_unvisited,
                key=lambda nid: (
                    priority.get(_room_type(spec, nid), 7),
                    _distance(graph.nodes[current].centroid, graph.nodes[nid].centroid),
                ),
            )
            path = graph.shortest_path(current, target)
            for step in path[1:]:
                visited.append(step)
            current = target
            continue
        nxt = min(
            candidates,
            key=lambda item: (
                priority.get(_room_type(spec, item[0]), 7),
                _distance(graph.nodes[current].centroid, graph.nodes[item[0]].centroid),
            ),
        )[0]
        current = nxt
    return stops


def _room_type(spec: HouseSpec, room_id: str) -> RoomType:
    room = spec.room_by_id(room_id)
    return room.type if room else RoomType.OTHER


def _distance(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def strongest_view(spec: HouseSpec, model: BuildingModel) -> Point3:
    """Pick the exterior viewpoint with the longest facade and best entrance view."""
    lo, hi = model.mesh.bounds
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    entrances = spec.entrance_openings()
    if entrances:
        wall, opening = entrances[0]
        along = opening.offset_m / max(wall.length_m, 1e-9)
        ex = wall.start[0] + (wall.end[0] - wall.start[0]) * along
        ey = wall.start[1] + (wall.end[1] - wall.start[1]) * along
        dx, dy = ex - cx, ey - cy
        norm = math.hypot(dx, dy) or 1.0
        span = max(hi[0] - lo[0], hi[1] - lo[1])
        return (ex + dx / norm * span * 0.95, ey + dy / norm * span * 0.95, 2.1)
    return (cx, lo[1] - (hi[1] - lo[1]) * 0.8, 2.1)
