"""Solar geometry.

Blueprint p.49: if location and time are known, compute the sun from them;
otherwise use a consistent default and do not claim it is the real sun.

Implements the NOAA solar-position approximation (declination + equation of
time). Accuracy is a fraction of a degree — far beyond what is needed to pick a
plausible sun angle, and it is only used when the engineer actually supplied a
location.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class SunPosition:
    altitude_deg: float
    azimuth_deg: float  # clockwise from north
    source: str  # 'computed' | 'default'
    description: str

    @property
    def direction(self) -> tuple[float, float, float]:
        """Unit vector pointing from the scene toward the sun (+x east, +y north, +z up)."""
        alt = math.radians(self.altitude_deg)
        az = math.radians(self.azimuth_deg)
        return (
            round(math.sin(az) * math.cos(alt), 4),
            round(math.cos(az) * math.cos(alt), 4),
            round(max(math.sin(alt), 0.08), 4),
        )

    @property
    def warmth(self) -> float:
        """0 = cool/high sun, 1 = warm/low sun (used for colour grading)."""
        return round(max(0.0, min(1.0, 1.0 - (self.altitude_deg / 65.0))), 3)


DEFAULT_SUN = SunPosition(
    altitude_deg=38.0,
    azimuth_deg=135.0,
    source="default",
    description=(
        "Mid-morning sun from the south-east. No site coordinates were supplied, "
        "so this is a presentation default, not the real sun at the property."
    ),
)


def solar_position(
    latitude: float | None,
    longitude: float | None,
    local_time: str | None,
) -> SunPosition:
    if latitude is None or longitude is None:
        return DEFAULT_SUN
    when = _parse_time(local_time)
    if when is None:
        return DEFAULT_SUN
    try:
        alt, az = _noaa(latitude, longitude, when)
    except (ValueError, ZeroDivisionError):  # pragma: no cover - defensive
        return DEFAULT_SUN
    return SunPosition(
        altitude_deg=round(alt, 2),
        azimuth_deg=round(az, 2),
        source="computed",
        description=(
            f"Sun computed for {latitude:.3f}, {longitude:.3f} at "
            f"{when.strftime('%Y-%m-%d %H:%M UTC')}."
        ),
    )


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    for parser in (
        datetime.fromisoformat,
        lambda v: datetime.strptime(v, "%Y-%m-%d %H:%M"),
        lambda v: datetime.strptime(v, "%Y-%m-%d"),
    ):
        try:
            parsed = parser(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _noaa(latitude: float, longitude: float, when: datetime) -> tuple[float, float]:
    when = when.astimezone(timezone.utc)
    day_of_year = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60.0 + when.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour - 12.0) / 24.0)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    time_offset = eqtime + 4.0 * longitude
    true_solar_minutes = hour * 60.0 + time_offset
    hour_angle = math.radians(true_solar_minutes / 4.0 - 180.0)
    lat_rad = math.radians(latitude)

    cos_zenith = (
        math.sin(lat_rad) * math.sin(decl)
        + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle)
    )
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.acos(cos_zenith)
    altitude = 90.0 - math.degrees(zenith)

    # Azimuth measured clockwise from north.
    sin_az = math.sin(hour_angle)
    cos_az = (
        math.cos(hour_angle) * math.sin(lat_rad) - math.tan(decl) * math.cos(lat_rad)
    )
    azimuth = (math.degrees(math.atan2(sin_az, cos_az)) + 180.0) % 360.0
    return altitude, azimuth
