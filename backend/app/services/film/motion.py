"""Camera motion sampling.

A shot stores start and end keyframes plus a motion profile. Sampling with
smoothstep easing (blueprint p.53: "smooth acceleration/deceleration rather than
abrupt starts") gives a professional feel and keeps the motion physically
plausible: an orbit interpolates around the subject rather than through it.
"""

from __future__ import annotations

import math

from ..modeling.raster import Camera
from .shotplan import Shot

Point3 = tuple[float, float, float]


def smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ease_in_out(t: float) -> float:
    """Sine easing — marginally gentler than smoothstep at the ends."""
    t = max(0.0, min(1.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _lerp3(a: Point3, b: Point3, t: float) -> Point3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def sample_camera(shot: Shot, t: float) -> Camera:
    """Camera state at normalised time ``t`` in [0, 1]."""
    t = max(0.0, min(1.0, t))
    eased = ease_in_out(t)
    start = shot.start
    end = shot.end

    if shot.motion == "orbit":
        centre = start.target
        radius_start = math.hypot(
            start.position[0] - centre[0], start.position[1] - centre[1]
        )
        radius_end = math.hypot(
            end.position[0] - centre[0], end.position[1] - centre[1]
        )
        angle_start = math.atan2(start.position[1] - centre[1], start.position[0] - centre[0])
        angle_end = math.atan2(end.position[1] - centre[1], end.position[0] - centre[0])
        delta = (angle_end - angle_start + math.pi) % (2 * math.pi) - math.pi
        angle = angle_start + delta * eased
        radius = radius_start + (radius_end - radius_start) * eased
        position = (
            centre[0] + math.cos(angle) * radius,
            centre[1] + math.sin(angle) * radius,
            start.position[2] + (end.position[2] - start.position[2]) * eased,
        )
        target = _lerp3(start.target, end.target, eased)
    elif shot.motion == "crane":
        position = _lerp3(start.position, end.position, eased)
        # A crane move follows a slight arc rather than a straight line.
        lift = math.sin(math.pi * eased) * 0.04 * (
            math.hypot(end.position[0] - start.position[0], end.position[1] - start.position[1])
        )
        position = (position[0], position[1], position[2] + lift)
        target = _lerp3(start.target, end.target, eased)
    else:
        position = _lerp3(start.position, end.position, t)
        target = _lerp3(start.target, end.target, t)

    fov = start.fov_deg + (end.fov_deg - start.fov_deg) * eased
    return Camera(position=position, target=target, fov_deg=fov)


def camera_positions(shot: Shot, samples: int) -> list[Point3]:
    return [sample_camera(shot, i / max(1, samples - 1)).position for i in range(samples)]
