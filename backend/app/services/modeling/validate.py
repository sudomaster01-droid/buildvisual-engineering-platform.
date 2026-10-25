"""Structural validation and room-graph construction.

Blueprint p.42/p.51: geometry and connectivity must be checked *before* any
expensive generation, and a camera must never be routed through a wall. This
module produces a machine-readable report; critical findings raise so the
pipeline stops before spending money.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ...errors import GeometryError
from ..spec.geometry_helpers import (
    bbox,
    bbox_overlap,
    distance_point_to_segment,
    point_in_polygon,
    polygon_area,
    polygon_centroid,
)
from ..spec.schema import HouseSpec, OpeningKind, RoomType

MIN_ROOM_AREA_M2 = 1.5
MAX_FOOTPRINT_M2 = 2500.0
MIN_CEILING_M = 2.0
MAX_CEILING_M = 5.0


@dataclass
class Finding:
    code: str
    message: str
    severity: str = "warning"  # info | warning | critical
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def critical(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "critical"]

    @property
    def ok(self) -> bool:
        return not self.critical

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "metrics": self.metrics,
            "findings": [f.as_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Room graph
# ---------------------------------------------------------------------------


@dataclass
class RoomNode:
    id: str
    name: str
    type: str
    level: int
    centroid: tuple[float, float]
    area_m2: float


@dataclass
class RoomEdge:
    a: str
    b: str
    kind: str  # door | opening | stair
    point: tuple[float, float]
    width_m: float
    same_level: bool = True


@dataclass
class RoomGraph:
    nodes: dict[str, RoomNode] = field(default_factory=dict)
    edges: list[RoomEdge] = field(default_factory=list)
    entrance_node: str | None = None
    entrance_point: tuple[float, float] | None = None

    def neighbours(self, node_id: str) -> list[tuple[str, RoomEdge]]:
        out = []
        for edge in self.edges:
            if edge.a == node_id:
                out.append((edge.b, edge))
            elif edge.b == node_id:
                out.append((edge.a, edge))
        return out

    def shortest_path(self, start: str, goal: str) -> list[str]:
        """Dijkstra over passable edges; returns [] when unreachable."""
        import heapq

        if start not in self.nodes or goal not in self.nodes:
            return []
        dist = {start: 0.0}
        prev: dict[str, str] = {}
        heap = [(0.0, start)]
        while heap:
            d, node = heapq.heappop(heap)
            if node == goal:
                break
            if d > dist.get(node, math.inf):
                continue
            for nxt, edge in self.neighbours(node):
                cost = self._edge_cost(node, nxt, edge)
                nd = d + cost
                if nd < dist.get(nxt, math.inf):
                    dist[nxt] = nd
                    prev[nxt] = node
                    heapq.heappush(heap, (nd, nxt))
        if goal not in dist:
            return []
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        return list(reversed(path))

    @staticmethod
    def _edge_cost(a: str, b: str, edge: RoomEdge) -> float:
        if edge.kind == "stair":
            return 3.0  # stairs cost more: prefer walking over climbing mid-shot
        return 1.0 + max(0.0, 1.2 - edge.width_m) * 0.5


def build_room_graph(spec: HouseSpec) -> RoomGraph:
    graph = RoomGraph()
    for room in spec.rooms:
        cx, cy = polygon_centroid(room.polygon)
        graph.nodes[room.id] = RoomNode(
            id=room.id, name=room.name, type=room.type.value, level=room.level,
            centroid=(round(cx, 3), round(cy, 3)),
            area_m2=round(polygon_area(room.polygon), 2),
        )

    for wall in spec.walls:
        for opening in wall.openings:
            if opening.kind is OpeningKind.WINDOW:
                continue
            along = (
                wall.start[0] + (wall.end[0] - wall.start[0]) * (opening.offset_m / max(wall.length_m, 1e-6)),
                wall.start[1] + (wall.end[1] - wall.start[1]) * (opening.offset_m / max(wall.length_m, 1e-6)),
            )
            connects = [c for c in opening.connects if c in graph.nodes]
            if len(connects) == 2:
                graph.edges.append(RoomEdge(
                    a=connects[0], b=connects[1], kind=opening.kind.value,
                    point=(round(along[0], 3), round(along[1], 3)),
                    width_m=opening.width_m,
                ))
            elif opening.is_entrance and len(connects) == 1:
                graph.entrance_node = connects[0]
                graph.entrance_point = (round(along[0], 3), round(along[1], 3))
            elif len(connects) == 1:
                graph.entrance_node = graph.entrance_node or connects[0]
                graph.entrance_point = graph.entrance_point or (round(along[0], 3), round(along[1], 3))

    # Stairs connect the rooms that contain their ends.
    for stair in spec.stairs:
        upper_level = stair.to_level
        lower_room = _room_containing(spec, stair.start, stair.from_level)
        upper_room = _room_containing(spec, stair.end, upper_level)
        if lower_room and upper_room and lower_room.id != upper_room.id:
            graph.edges.append(RoomEdge(
                a=lower_room.id, b=upper_room.id, kind="stair",
                point=(round(stair.start[0], 3), round(stair.start[1], 3)),
                width_m=stair.width_m, same_level=False,
            ))

    if graph.entrance_node is None and graph.nodes:
        # No exterior door: start from the room whose centroid is nearest a shell wall.
        graph.entrance_node = next(iter(graph.nodes))
    return graph


def _room_containing(spec: HouseSpec, point, level: int):
    for room in spec.rooms:
        if room.level == level and point_in_polygon(point, room.polygon):
            return room
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_spec(spec: HouseSpec) -> ValidationReport:
    report = ValidationReport()
    f = report.findings

    if not spec.rooms:
        raise GeometryError("The house has no rooms; there is nothing to model.")

    # level sanity
    indices = sorted(lvl.index for lvl in spec.levels)
    if indices != list(range(len(indices))):
        f.append(Finding("level_index_gap", "Floor levels are not numbered consecutively.",
                         "critical"))
    for level in spec.levels:
        if not (MIN_CEILING_M <= level.ceiling_height_m <= MAX_CEILING_M):
            f.append(Finding(
                "ceiling_out_of_range",
                f"{level.name} ceiling height {level.ceiling_height_m} m is outside "
                f"{MIN_CEILING_M}–{MAX_CEILING_M} m.",
                "critical", {"level": level.index},
            ))

    # rooms
    total_area = 0.0
    for room in spec.rooms:
        area = polygon_area(room.polygon)
        total_area += area
        if area < MIN_ROOM_AREA_M2:
            f.append(Finding(
                "room_too_small",
                f"Room '{room.name}' is only {area:.2f} m².",
                "warning", {"room_id": room.id, "area_m2": round(area, 2)},
            ))
        if not room.is_factual:
            f.append(Finding(
                "room_inferred",
                f"Room '{room.name}' is inferred, not taken from a drawing or statement.",
                "info", {"room_id": room.id, "source": room.source.value},
            ))

    if total_area > MAX_FOOTPRINT_M2:
        f.append(Finding(
            "footprint_too_large",
            f"Total modelled floor area {total_area:.0f} m² exceeds the supported maximum.",
            "critical",
        ))
    if total_area < 20:
        f.append(Finding("footprint_tiny", "Total floor area below 20 m².", "critical"))

    # room overlap (approximate: bounding boxes that overlap heavily and identical
    # centroids inside each other indicate duplicate geometry)
    boxes = {r.id: (bbox(r.polygon), r) for r in spec.rooms}
    ids = list(boxes)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_id, b_id = ids[i], ids[j]
            (ab, a_room) = boxes[a_id]
            (bb, b_room) = boxes[b_id]
            if a_room.level != b_room.level:
                continue
            if not bbox_overlap(ab, bb):
                continue
            ca = polygon_centroid(a_room.polygon)
            cb = polygon_centroid(b_room.polygon)
            if point_in_polygon(ca, b_room.polygon) and point_in_polygon(cb, a_room.polygon):
                f.append(Finding(
                    "rooms_overlap",
                    f"Rooms '{a_room.name}' and '{b_room.name}' overlap on the same floor.",
                    "critical", {"rooms": [a_room.id, b_room.id]},
                ))

    # walls + openings
    for wall in spec.walls:
        if wall.interior and not any(o.kind is not OpeningKind.WINDOW for o in wall.openings):
            f.append(Finding(
                "partition_without_opening",
                f"Interior wall {wall.id} has no door or opening; the space will be sealed.",
                "warning", {"wall_id": wall.id},
            ))
        for opening in wall.openings:
            centre = (
                wall.start[0] + (wall.end[0] - wall.start[0]) * opening.offset_m / max(wall.length_m, 1e-9),
                wall.start[1] + (wall.end[1] - wall.start[1]) * opening.offset_m / max(wall.length_m, 1e-9),
            )
            if opening.offset_m - opening.width_m / 2 < -0.05 or (
                opening.offset_m + opening.width_m / 2 > wall.length_m + 0.05
            ):
                f.append(Finding(
                    "opening_outside_wall",
                    f"Opening {opening.id} extends beyond its host wall.",
                    "critical", {"opening_id": opening.id, "wall_id": wall.id},
                ))
            if opening.sill_m + opening.height_m > min(wall.height_m, 6.0) + 0.05:
                f.append(Finding(
                    "opening_too_tall",
                    f"Opening {opening.id} is taller than its wall allows.",
                    "critical", {"opening_id": opening.id},
                ))
            for connected in opening.connects:
                room = spec.room_by_id(connected)
                if room is None:
                    f.append(Finding(
                        "opening_unknown_room",
                        f"Opening {opening.id} references unknown room '{connected}'.",
                        "critical", {"opening_id": opening.id},
                    ))
                elif not point_in_polygon(centre, room.polygon) and _distance_to_room(centre, room) > 0.75:
                    f.append(Finding(
                        "opening_not_at_room",
                        f"Opening {opening.id} is not on the boundary of room '{connected}'.",
                        "warning",
                        {"opening_id": opening.id, "room_id": connected},
                    ))

    # stairs
    for stair in spec.stairs:
        level_ids = {lvl.index for lvl in spec.levels}
        if stair.from_level not in level_ids or stair.to_level not in level_ids:
            f.append(Finding(
                "stair_unknown_level", f"Stair {stair.id} references a missing floor.",
                "critical", {"stair_id": stair.id},
            ))
        lower = _room_containing(spec, stair.start, stair.from_level)
        if lower is None:
            f.append(Finding(
                "stair_off_plan", f"Stair {stair.id} does not begin inside a room.",
                "warning", {"stair_id": stair.id},
            ))
    if spec.has_upper_floor and not spec.stairs:
        f.append(Finding(
            "upper_floor_unreachable",
            "The house has an upper floor but no stair connects it.",
            "critical",
        ))

    # graph reachability
    graph = build_room_graph(spec)
    if graph.entrance_node:
        reachable = {graph.entrance_node}
        frontier = [graph.entrance_node]
        while frontier:
            current = frontier.pop()
            for nxt, _edge in graph.neighbours(current):
                if nxt not in reachable:
                    reachable.add(nxt)
                    frontier.append(nxt)
        unreachable = [n for n, node in graph.nodes.items() if n not in reachable]
        if unreachable:
            names = [graph.nodes[n].name for n in unreachable]
            f.append(Finding(
                "rooms_unreachable",
                "These rooms cannot be reached from the entrance: " + ", ".join(names),
                "warning", {"room_ids": unreachable},
            ))
        report.metrics["reachable_rooms"] = len(reachable)
        report.metrics["total_rooms"] = len(graph.nodes)

    report.metrics.update({
        "floor_area_m2": round(total_area, 2),
        "rooms": len(spec.rooms),
        "levels": len(spec.levels),
        "walls": len(spec.walls),
        "openings": sum(len(w.openings) for w in spec.walls),
        "stairs": len(spec.stairs),
        "factual_fields": spec.factual_field_count(),
        "graph_edges": len(graph.edges),
    })
    _ = RoomType  # imported for typing readability
    return report


def _distance_to_room(point, room) -> float:
    poly = list(room.polygon)
    return min(
        distance_point_to_segment(point, poly[i], poly[(i + 1) % len(poly)])
        for i in range(len(poly))
    )


def assert_valid(spec: HouseSpec) -> ValidationReport:
    report = validate_spec(spec)
    if not report.ok:
        messages = "; ".join(f.message for f in report.critical)
        raise GeometryError(f"Structural checks failed: {messages}")
    return report
