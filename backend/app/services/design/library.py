"""Asset and material libraries (data, not logic).

The library JSON is versioned and shipped with the code; nothing here is
downloaded at runtime and no third-party mesh licence is involved because every
item is a parametric assembly of primitives authored in-repo. The manifest
records ``asset_source`` and ``license`` for each placed item so that swapping in
CC0 mesh assets later is a data change with a full audit trail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

LIBRARY_DIR = Path(__file__).parent / "library"
ASSET_LICENSE = "BuildVisual in-repo parametric asset (CC0-equivalent: authored for this project)"


class LibraryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Part:
    name: str
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    mat: str


@dataclass(frozen=True)
class Asset:
    key: str
    display_name: str
    rooms: tuple[str, ...]
    width: float
    depth: float
    height: float
    wall_affinity: str
    material_slots: dict[str, str]
    parts: tuple[Part, ...]
    flat: bool = False
    is_landscape: bool = False

    @property
    def footprint(self) -> tuple[float, float]:
        return (self.width, self.depth)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "footprint_m": [round(self.width, 2), round(self.depth, 2)],
            "height_m": round(self.height, 2),
            "license": ASSET_LICENSE,
            "source": "procedural:buildvisual",
        }


def _parse_parts(raw: list[dict], origin_x: float, origin_y: float) -> tuple[Part, ...]:
    parts: list[Part] = []
    for item in raw:
        x0, y0, z0, x1, y1, z1 = (float(v) for v in item["box"])
        parts.append(Part(
            name=str(item["name"]),
            x0=x0 - origin_x, y0=y0 - origin_y, z0=z0,
            x1=x1 - origin_x, y1=y1 - origin_y, z1=z1,
            mat=str(item["mat"]),
        ))
    return tuple(parts)


def _bounds(parts: tuple[Part, ...]) -> tuple[float, float, float, float]:
    x0 = min(p.x0 for p in parts)
    x1 = max(p.x1 for p in parts)
    y0 = min(p.y0 for p in parts)
    y1 = max(p.y1 for p in parts)
    return x0, y0, x1, y1


@lru_cache(maxsize=1)
def load_materials() -> dict[str, Any]:
    path = LIBRARY_DIR / "materials.json"
    data = json.loads(path.read_text())
    if "materials" not in data or "palettes" not in data:
        raise LibraryError("materials.json is missing required sections")
    return data


@lru_cache(maxsize=1)
def load_assets() -> dict[str, Asset]:
    path = LIBRARY_DIR / "furniture.json"
    data = json.loads(path.read_text())
    assets: dict[str, Asset] = {}
    for bucket, is_landscape in (("items", False), ("landscape", True)):
        for key, raw in data.get(bucket, {}).items():
            parts_raw = raw["parts"]
            origin_x = (min(float(p["box"][0]) for p in parts_raw)
                        + max(float(p["box"][3]) for p in parts_raw)) / 2.0
            origin_y = (min(float(p["box"][1]) for p in parts_raw)
                        + max(float(p["box"][4]) for p in parts_raw)) / 2.0
            parts = _parse_parts(parts_raw, origin_x, origin_y)
            x0, y0, x1, y1 = _bounds(parts)
            assets[key] = Asset(
                key=key,
                display_name=str(raw["display_name"]),
                rooms=tuple(raw.get("rooms", [])),
                width=x1 - x0,
                depth=y1 - y0,
                height=max(p.z1 for p in parts),
                wall_affinity=str(raw.get("wall_affinity", "any")),
                material_slots=dict(raw.get("material_slots", {})),
                parts=parts,
                flat=bool(raw.get("flat", False)),
                is_landscape=is_landscape,
            )
    if not assets:
        raise LibraryError("no assets could be loaded")
    return assets


def palette(palette_id: str | None, fallback: str = "warm_natural") -> tuple[str, dict[str, Any]]:
    palettes = load_materials()["palettes"]
    key = palette_id if palette_id in palettes else fallback
    return key, palettes[key]


def material_spec(key: str) -> dict[str, Any]:
    materials = load_materials()["materials"]
    raw = materials.get(key)
    if raw is None:
        raise LibraryError(f"unknown material key: {key}")
    return raw


def resolve_slot(slot: str, slots: dict[str, str], pal: dict[str, Any], default: str) -> str:
    """Map an asset material role to a concrete palette material."""
    material_key = slots.get(slot)
    if material_key in ("furniture_wood", "furniture_soft", "furniture_dark",
                        "metalware", "accent"):
        return pal.get(material_key, default)
    if material_key:
        return material_key
    return pal.get("furniture_wood", default)


def assets_for_room(room_type: str) -> list[Asset]:
    return [
        a for a in load_assets().values()
        if not a.is_landscape and room_type in a.rooms
    ]


def landscape_assets() -> dict[str, Asset]:
    return {k: a for k, a in load_assets().items() if a.is_landscape}


def library_version() -> dict[str, str]:
    materials = load_materials()
    furniture = json.loads((LIBRARY_DIR / "furniture.json").read_text())
    return {
        "materials": str(materials.get("library_version", "unknown")),
        "furniture": str(furniture.get("library_version", "unknown")),
    }
