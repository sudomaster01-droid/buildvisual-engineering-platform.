"""Plan-view geometry helpers shared by validation, design and modelling."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

Point2 = tuple[float, float]


def polygon_area(polygon: Iterable[Point2]) -> float:
    pts = np.asarray(list(polygon), dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def polygon_centroid(polygon: Iterable[Point2]) -> Point2:
    pts = np.asarray(list(polygon), dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return (0.0, 0.0)
    area = polygon_area(pts)
    if area < 1e-9:
        m = pts.mean(axis=0)
        return (float(m[0]), float(m[1]))
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    cx = float(np.sum((x + xn) * cross) / (6.0 * area))
    cy = float(np.sum((y + yn) * cross) / (6.0 * area))
    return (cx, cy)


def point_in_polygon(point: Point2, polygon: Iterable[Point2]) -> bool:
    pts = np.asarray(list(polygon), dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return False
    x, y = point
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
    return inside


def distance_point_to_segment(p: Point2, a: Point2, b: Point2) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def bbox(points: Iterable[Point2]) -> tuple[float, float, float, float]:
    pts = np.asarray(list(points), dtype=np.float64).reshape(-1, 2)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def bbox_overlap(a: tuple[float, float, float, float], b, tol: float = 1e-6) -> bool:
    return not (
        a[2] - tol <= b[0] or b[2] - tol <= a[0] or a[3] - tol <= b[1] or b[3] - tol <= a[1]
    )


def rect_polygon(x0: float, y0: float, x1: float, y1: float) -> list[Point2]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def inset_polygon(polygon: list[Point2], margin: float) -> list[Point2]:
    """Conservative inward offset for a convex-ish plan polygon (centroid scaling)."""
    cx, cy = polygon_centroid(polygon)
    pts = np.asarray(polygon, dtype=np.float64)
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        norm = math.hypot(dx, dy) or 1.0
        scale = max(0.0, norm - margin) / norm
        out.append((cx + dx * scale, cy + dy * scale))
    return out


def angle_deg(a: Point2, b: Point2) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_point(a: Point2, b: Point2, t: float) -> Point2:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t))


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def normalise_angle_deg(angle: float) -> float:
    return angle % 360.0
