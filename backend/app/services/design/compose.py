"""Automatic design completion: the styling layer.

Turns a locked structural model plus the engineer's stated preferences into a
scene manifest: materials, furniture placements, lighting and landscaping. The
blueprint (p.43/p.44) requires this layer to be *separate* from geometry — it may
read the model but must never change it. That is enforced structurally here: the
manifest contains no geometry mutation, and the model version hash is carried in
read-only.

Every choice is deterministic for a given (model version, attempt seed) pair, so
"Make it different" produces a reproducible alternative rather than random noise,
and a correction can be replayed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from ..modeling.geometry import BuildingModel
from ..modeling.mesh import Material, Mesh, ObjectRef, oriented_box
from ..spec.schema import HouseSpec, Room, RoomType, SourceType
from . import library
from .placement import RoomLayout, place_items
from .solar import SunPosition, solar_position

DESIGN_VERSION = "1.0.0"

#: Room program: (asset_key, role, priority). Applied in order; anything that
#: does not fit legally is skipped with a recorded reason.
ROOM_PROGRAM: dict[RoomType, list[tuple[str, str, str]]] = {
    RoomType.LIVING: [
        ("sofa_three", "anchor", "wall"),
        ("rug_large", "soft", "center"),
        ("coffee_table", "soft", "center"),
        ("tv_unit", "media", "wall"),
        ("armchair", "extra", "any"),
        ("plant_large", "life", "any"),
        ("floor_lamp", "light", "any"),
        ("art_canvas", "life", "wall"),
    ],
    RoomType.DINING: [
        ("dining_table", "anchor", "center"),
        ("dining_chair", "seating", "center"),
        ("pendant", "light", "ceiling"),
        ("plant_large", "life", "any"),
        ("art_canvas", "life", "wall"),
    ],
    RoomType.KITCHEN: [
        ("kitchen_run", "anchor", "wall"),
        ("island", "island", "center"),
        ("pendant", "light", "ceiling"),
    ],
    RoomType.BEDROOM: [
        ("bed_double", "anchor", "wall"),
        ("nightstand", "bedside", "wall"),
        ("wardrobe", "storage", "wall"),
        ("rug_large", "soft", "center"),
        ("art_canvas", "life", "wall"),
        ("plant_large", "life", "any"),
    ],
    RoomType.BATHROOM: [
        ("bath_run", "anchor", "wall"),
        ("bath_tub", "bath", "wall"),
    ],
    RoomType.HALL: [
        ("art_canvas", "life", "wall"),
        ("plant_large", "life", "any"),
        ("floor_lamp", "light", "any"),
    ],
    RoomType.STUDY: [
        ("desk", "anchor", "wall"),
        ("armchair", "seating", "any"),
        ("bookshelf", "storage", "wall"),
        ("plant_large", "life", "any"),
    ],
    RoomType.UTILITY: [("kitchen_run", "anchor", "wall")],
    RoomType.BALCONY: [("armchair", "seating", "any"), ("plant_large", "life", "any")],
    RoomType.GARAGE: [],
    RoomType.STAIR: [],
    RoomType.OTHER: [("plant_large", "life", "any")],
}

#: Minimum room area before an asset is considered at all.
MIN_AREA_BY_ASSET = {
    "sofa_three": 8.5,
    "dining_table": 6.0,
    "island": 9.0,
    "bed_double": 7.5,
    "wardrobe": 5.0,
    "bath_tub": 3.6,
    "kitchen_run": 4.0,
    "bookshelf": 3.5,
    "tv_unit": 7.0,
    "rug_large": 8.0,
}

PALETTE_ORDER = ["warm_natural", "cool_modern", "crisp_estate", "mediterranean"]


@dataclass
class Scene:
    """The design layer's complete output for one model version."""

    manifest: dict[str, Any]
    materials: dict[str, Material] = field(default_factory=dict)
    design_mesh: Mesh = field(default_factory=Mesh)
    sun: SunPosition | None = None


def _seeded_rng(seed_material: str):
    """Deterministic integer stream from a string (no wall clock, no OS entropy)."""
    digest = hashlib.sha256(seed_material.encode()).digest()
    state = int.from_bytes(digest[:8], "big")

    def nxt(bound: int) -> int:
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 33) % max(1, bound)

    return nxt


def choose_palette(
    spec: HouseSpec, seed_material: str, attempt: int = 0,
    palette_override: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Palette from an explicit correction, then the engineer's intent, else deterministic."""
    if palette_override:
        return library.palette(palette_override)
    if spec.design_intent.palette:
        return library.palette(spec.design_intent.palette)
    rng = _seeded_rng(f"palette:{seed_material}")
    index = rng(len(PALETTE_ORDER))
    index = (index + attempt) % len(PALETTE_ORDER)
    return library.palette(PALETTE_ORDER[index])


def _door_points(spec: HouseSpec, room: Room) -> list[tuple[tuple[float, float], float]]:
    from ..spec.geometry_helpers import polygon_centroid, point_in_polygon

    cx, cy = polygon_centroid(room.polygon)
    points: list[tuple[tuple[float, float], float]] = []
    for wall in spec.walls:
        if wall.level != room.level:
            continue
        for opening in wall.openings:
            if opening.id.startswith("dummy"):
                continue
            along = opening.offset_m / max(wall.length_m, 1e-9)
            px = wall.start[0] + (wall.end[0] - wall.start[0]) * along
            py = wall.start[1] + (wall.end[1] - wall.start[1]) * along
            touches = room.id in opening.connects
            if not touches:
                # exterior openings that sit on this room's boundary count too
                probe_in = point_in_polygon(
                    (px + (cx - px) * 0.05, py + (cy - py) * 0.05), room.polygon
                )
                touches = probe_in and opening.kind.value != "window"
            if touches and opening.kind.value != "window":
                points.append(((round(px, 3), round(py, 3)), opening.width_m))
    return points


def _material_for_slot(slot_material_key: str, palette: dict[str, Any]) -> Material:
    key = palette.get(slot_material_key) or slot_material_key
    raw = library.material_spec(key)
    return Material(
        key=key,
        name=str(raw["name"]),
        color=tuple(float(c) for c in raw["color"]),
        roughness=float(raw.get("roughness", 0.7)),
        metallic=float(raw.get("metallic", 0.0)),
        kind=str(raw.get("kind", "generic")),
        source="procedural:buildvisual",
    )


def compose_scene(
    model: BuildingModel,
    *,
    attempt: int = 0,
    seed_material: str | None = None,
    palette_override: str | None = None,
    lighting_overrides: dict[str, Any] | None = None,
    staging: dict[str, Any] | None = None,
) -> Scene:
    spec = model.spec
    seed_material = seed_material or f"{spec.title}:{model.geometry_hash[:12]}"
    palette_id, palette = choose_palette(spec, seed_material, attempt, palette_override)
    staging = staging or {}
    density = staging.get("density")
    target_rooms = set(staging.get("rooms") or [])
    rng = _seeded_rng(f"design:{seed_material}:{attempt}")

    assets = library.load_assets()
    rooms_manifest: list[dict[str, Any]] = []
    placed_objects: list[dict[str, Any]] = []
    layouts: list[RoomLayout] = []
    skipped_total: list[dict[str, Any]] = []

    for room in spec.rooms:
        if room.type in (RoomType.STAIR, RoomType.GARAGE):
            continue
        program = ROOM_PROGRAM.get(room.type, ROOM_PROGRAM[RoomType.OTHER])
        plan: list[tuple[str, tuple[float, float], str]] = []
        area = _room_area(room)
        for asset_key, _role, zone in program:
            asset = assets.get(asset_key)
            if asset is None:
                continue
            if area < MIN_AREA_BY_ASSET.get(asset_key, 2.0):
                continue
            if asset_key == "dining_chair":
                count = 6 if area >= 16 else 4
                for _ in range(count):
                    plan.append((asset_key, asset.footprint, "any"))
                continue
            if asset_key == "nightstand":
                for _ in range(2):
                    plan.append((asset_key, asset.footprint, "wall"))
                continue
            if asset_key == "bed_double" and area < 9.5:
                plan.append(("bed_single", assets["bed_single"].footprint, "wall"))
                continue
            if asset_key in ("art_canvas", "floor_lamp", "plant_large", "armchair"):
                if density == "sparse" and rng(2) != 0:
                    continue
                if density != "fuller" and rng(3) == 0:
                    continue  # light, deterministic sparseness
            if density == "fuller" and asset_key == "armchair" and area >= 12:
                plan.append((asset_key, asset.footprint, "any"))
            plan.append((asset_key, asset.footprint, "wall" if zone == "wall" else zone))

        doors = _door_points(spec, room)
        layout = place_items(
            room_id=room.id,
            room_type=room.type.value,
            polygon=[(float(p[0]), float(p[1])) for p in room.polygon],
            doors=doors,
            plan=plan,
            ceiling_height=room.ceiling_height_m,
        )
        layouts.append(layout)
        skipped_total.extend({"room_id": room.id, **s} for s in layout.skipped)

        level_elevation = model.level_elevations.get(room.level, 0.0)
        for placement in layout.placements:
            asset = assets[placement.asset_key]
            materials = {
                role: library.resolve_slot(role, asset.material_slots, palette, "oak_wide")
                for role in {p.mat for p in asset.parts}
            }
            z = level_elevation
            if placement.zone == "ceiling":
                z = level_elevation + room.ceiling_height_m - asset.height
            entry = {
                "asset_key": placement.asset_key,
                "room_id": room.id,
                "room_type": room.type.value,
                "position": [placement.position[0], placement.position[1], round(z, 3)],
                "rotation_deg": placement.rotation_deg,
                "footprint_m": [round(placement.footprint[0], 3), round(placement.footprint[1], 3)],
                "height_m": round(asset.height, 3),
                **asset.as_dict(),
                "materials": materials,
                "provenance": SourceType.INFERRED.value,
            }
            placed_objects.append(entry)
        rooms_manifest.append({
            "id": room.id,
            "name": room.name,
            "type": room.type.value,
            "level": room.level,
            "area_m2": round(area, 2),
            "surface_materials": {
                "floor": palette.get(
                    "floor_wet" if room.type is RoomType.BATHROOM else "floor_living"
                ),
                "wall": palette.get("wall_interior"),
                "ceiling": palette.get("ceiling"),
            },
            "items": [p.asset_key for p in layout.placements],
            "skipped": layout.skipped,
        })

    landscape, landscape_objects = _compose_landscape(
        model, palette, palette_id, seed_material, attempt
    )
    sun = solar_position(
        spec.site.latitude, spec.site.longitude, spec.site.local_time
    )
    lighting = _lighting_preset(palette, sun, spec, lighting_overrides)

    manifest = {
        "design_version": DESIGN_VERSION,
        "library_version": library.library_version(),
        "palette_id": palette_id,
        "palette_display_name": palette.get("display_name", palette_id),
        "seed": seed_material,
        "attempt": attempt,
        "model_version_hash": model.geometry_hash,
        "structural_geometry_modified": False,
        "rooms": rooms_manifest,
        "objects": placed_objects,
        "landscape": landscape,
        "landscape_objects": landscape_objects,
        "lighting": lighting,
        "skipped_items": skipped_total,
        "asset_license": library.ASSET_LICENSE,
    }
    scene = Scene(manifest=manifest, sun=sun)
    wanted: set[str] = set()
    for value in palette.values():
        if isinstance(value, str) and _valid_material(value):
            wanted.add(value)
    for item in placed_objects:
        wanted.update(item["materials"].values())
    for item in landscape_objects:
        wanted.update(m for m in item.get("materials", {}).values())
    wanted.add(palette.get("lawn", "lawn"))
    wanted.add(palette.get("paving", "paving"))
    scene.materials = {
        key: _material_for_slot(key, palette) for key in wanted if _valid_material(key)
    }
    scene.design_mesh = build_design_mesh(scene, model)
    return scene


def _valid_material(key: str) -> bool:
    try:
        library.material_spec(key)
        return True
    except library.LibraryError:
        return False


def _room_area(room: Room) -> float:
    from ..spec.geometry_helpers import polygon_area

    return polygon_area(room.polygon)


def _lighting_preset(
    palette: dict[str, Any], sun: SunPosition, spec: HouseSpec,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    overrides = overrides or {}
    interior_k = int(overrides.get("temperature_k") or palette.get("lighting_temp_k", 3000))
    exposure_delta = float(overrides.get("exposure_delta", 0.0))
    contrast_delta = float(overrides.get("contrast_delta", 0.0))
    preset = {
        "sun": {
            "direction": list(sun.direction),
            "altitude_deg": sun.altitude_deg,
            "azimuth_deg": sun.azimuth_deg,
            "source": sun.source,
            "description": sun.description,
        },
        "exterior": {
            "exposure": round(0.95 + 0.10 * (1.0 - sun.warmth) + exposure_delta, 3),
            "ambient": 0.44,
            "sky_temperature_k": int(9500 - 3000 * sun.warmth),
            "time_of_day_label": _time_label(sun),
        },
        "interior": {
            "temperature_k": interior_k,
            "exposure": round(1.02 + exposure_delta, 3),
            "ambient": 0.52,
            "practical_lights": True,
        },
        "grading": {
            "contrast": round(1.04 + contrast_delta, 3),
            "saturation": 1.03,
            "warmth": round(
                0.35 + 0.3 * sun.warmth + (0.1 if overrides.get("temperature_bias") == "warm" else 0.0)
                - (0.08 if overrides.get("temperature_bias") == "cool" else 0.0),
                3,
            ),
        },
        "requested_overrides": overrides,
        "claims": (
            "Sun position computed from the supplied site coordinates."
            if sun.source == "computed"
            else "Presentation lighting default; no site coordinates were supplied."
        ),
        "site_context_supplied": bool(spec.site.context_note),
    }
    return preset


def _time_label(sun: SunPosition) -> str:
    if sun.source != "computed":
        return "mid-morning"
    alt = sun.altitude_deg
    if alt < 5:
        return "golden hour"
    if alt < 20:
        return "early morning" if sun.azimuth_deg < 180 else "late afternoon"
    if alt < 50:
        return "mid-morning" if sun.azimuth_deg < 180 else "afternoon"
    return "midday"


# ---------------------------------------------------------------------------
# Landscaping
# ---------------------------------------------------------------------------


def _compose_landscape(
    model: BuildingModel,
    palette: dict[str, Any],
    palette_id: str,
    seed_material: str,
    attempt: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Ground plane treatment, approach path, planting. Never blocks the camera."""
    spec = model.spec
    assets = library.landscape_assets()
    rng = _seeded_rng(f"landscape:{seed_material}:{attempt}")

    lo, hi = model.mesh.bounds_excluding()
    width = float(hi[0] - lo[0])
    depth = float(hi[1] - lo[1])
    pad = max(26.0, (width + depth) * 0.55)
    x0, y0 = lo[0] - pad, lo[1] - pad
    x1, y1 = hi[0] + pad, hi[1] + pad

    entrances = spec.entrance_openings()
    entry_point = None
    entry_angle = 270.0
    if entrances:
        wall, opening = entrances[0]
        along = opening.offset_m / max(wall.length_m, 1e-9)
        entry_point = (
            wall.start[0] + (wall.end[0] - wall.start[0]) * along,
            wall.start[1] + (wall.end[1] - wall.start[1]) * along,
        )
        import math as _math

        entry_angle = _math.degrees(_math.atan2(
            wall.end[1] - wall.start[1], wall.end[0] - wall.start[0]
        )) % 360.0

    if entry_point is None:
        entry_point = ((lo[0] + hi[0]) / 2, lo[1] - 2.0)

    objects: list[dict[str, Any]] = []
    drive_len = max(6.0, float(entry_point[1] - y0))
    drive_width = 3.4
    objects.append({
        "asset_key": "driveway",
        "kind": "paving",
        "position": [round(entry_point[0], 3), round(entry_point[1] - drive_len / 2 + 0.3, 3), 0.0],
        "elevation_m": 0.0,
        "size": [round(drive_width, 2), round(drive_len, 2), 0.04],
        "material": palette.get("paving", "paving"),
        "provenance": SourceType.SYSTEM_DEFAULT.value,
    })
    objects.append({
        "asset_key": "entrance_landing",
        "kind": "paving",
        "position": [round(entry_point[0], 3), round(entry_point[1] - 1.1, 3), 0.0],
        "size": [round(opening_width(entrances) + 1.6, 2), 2.2, 0.05],
        "material": palette.get("paving", "paving"),
        "provenance": SourceType.SYSTEM_DEFAULT.value,
    })

    # Planting: trees along the boundary, explicitly outside the approach corridor.
    # A generous clear cone in front of the entrance keeps trees out of the
    # approach shots; the camera plan starts outside the plot looking in.
    corridor_half = (drive_width / 2) + 5.6
    margin = 1.6
    tree_count = max(4, min(12, int((x1 - x0 + y1 - y0) / 7.0)))
    for i in range(tree_count):
        for attempt_index in range(12):
            px = x0 + margin + rng(int(max(1.0, (x1 - x0 - 2 * margin)) * 10)) / 10.0
            py = y0 + margin + rng(int(max(1.0, (y1 - y0 - 2 * margin)) * 10)) / 10.0
            if abs(px - entry_point[0]) < corridor_half and py < entry_point[1] + 1.0:
                continue
            if _inside(px, py, (lo[0] - 1.5, lo[1] - 1.5, hi[0] + 1.5, hi[1] + 1.5)):
                continue
            if any(
                abs(px - o["position"][0]) < 3.0 and abs(py - o["position"][1]) < 3.0
                for o in objects if o["kind"] == "tree"
            ):
                continue
            key = "tree_slim" if rng(3) == 0 else "tree_broadleaf"
            asset = assets[key]
            objects.append({
                "asset_key": key,
                "kind": "tree",
                "position": [round(px, 3), round(py, 3), 0.0],
                "rotation_deg": float(rng(4) * 90),
                "materials": {
                    role: library.resolve_slot(role, asset.material_slots, palette, "lawn")
                    for role in {p.mat for p in asset.parts}
                },
                "provenance": SourceType.SYSTEM_DEFAULT.value,
            })
            break

    # Shrubs hugging the facade but clear of the entrance.
    for i in range(6):
        offset = (i - 2.5) * 2.2
        px = entry_point[0] + offset
        py = entry_point[1] - 0.75
        if abs(offset) < 1.8:
            continue
        if not (lo[0] - 0.9 <= px <= hi[0] + 0.9):
            continue
        asset = assets["shrub"]
        objects.append({
            "asset_key": "shrub",
            "kind": "shrub",
            "position": [round(px, 3), round(py, 3), 0.0],
            "rotation_deg": 0.0,
            "materials": {
                role: library.resolve_slot(role, asset.material_slots, palette, "lawn")
                for role in {p.mat for p in asset.parts}
            },
            "provenance": SourceType.SYSTEM_DEFAULT.value,
        })

    landscape = {
        "palette_id": palette_id,
        "ground_material": palette.get("lawn", "lawn"),
        "site_extent_m": {"width": round(x1 - x0, 2), "depth": round(y1 - y0, 2)},
        "approach": {
            "entrance_point": [round(entry_point[0], 3), round(entry_point[1], 3)],
            "entrance_facing_deg": round(entry_angle, 2),
            "driveway_width_m": drive_width,
        },
        "claims": (
            "The site surroundings are a presentation interpretation, not a survey "
            "of the real plot."
        ),
        "counts": {
            "trees": sum(1 for o in objects if o["kind"] == "tree"),
            "shrubs": sum(1 for o in objects if o["kind"] == "shrub"),
        },
    }
    return landscape, objects


def opening_width(entrances) -> float:
    return float(entrances[0][1].width_m) if entrances else 1.0


def _inside(x: float, y: float, rect) -> bool:
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


# ---------------------------------------------------------------------------
# Styled mesh
# ---------------------------------------------------------------------------


def build_design_mesh(scene: Scene, model: BuildingModel) -> Mesh:
    """Furniture + landscaping as a separate mesh that never mutates the model."""
    import math

    mesh = Mesh()
    material_by_key = dict(scene.materials)

    def material(key: str) -> Material:
        if key not in material_by_key:
            raw = library.material_spec(key)
            material_by_key[key] = Material(
                key=key, name=str(raw["name"]),
                color=tuple(float(c) for c in raw["color"]),
                roughness=float(raw.get("roughness", 0.7)),
                metallic=float(raw.get("metallic", 0.0)),
                kind=str(raw.get("kind", "generic")),
            )
        return material_by_key[key]

    assets = library.load_assets()

    for index, item in enumerate(scene.manifest["objects"]):
        asset = assets.get(item["asset_key"])
        if asset is None:
            continue
        angle = math.radians(float(item.get("rotation_deg", 0.0)))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        ox, oy, oz = item["position"]
        for part in asset.parts:
            key = item["materials"].get(part.mat, "oak_wide")
            # Rotate the part's local box corners, then take the axis-aligned box
            # of the rotated corners (renderer works from triangles, so use a
            # rotated footprint with the original vertical extent).
            corners = [
                (part.x0, part.y0), (part.x1, part.y0), (part.x1, part.y1), (part.x0, part.y1),
            ]
            rotated = [(ox + cx * cos_a - cy * sin_a, oy + cx * sin_a + cy * cos_a) for cx, cy in corners]
            xs = [p[0] for p in rotated]
            ys = [p[1] for p in rotated]
            verts = [
                (min(xs), min(ys), oz + part.z0), (max(xs), min(ys), oz + part.z0),
                (max(xs), max(ys), oz + part.z0), (min(xs), max(ys), oz + part.z0),
                (min(xs), min(ys), oz + part.z1), (max(xs), min(ys), oz + part.z1),
                (max(xs), max(ys), oz + part.z1), (min(xs), max(ys), oz + part.z1),
            ]
            import numpy as _np

            from ..modeling.mesh import _BOX_FACES

            mesh.add(
                _np.array(verts, dtype=float), _BOX_FACES.copy(),
                ObjectRef(
                    id=f"design:{item['room_id'] if 'room_id' in item else item['kind']}:{item['asset_key']}:{index}:{part.name}",
                    name=asset.display_name,
                    kind="furniture" if "room_id" in item else "landscape",
                    room_id=item.get("room_id"),
                    level=0,
                    material=key,
                    provenance=item.get("provenance", "inferred"),
                ),
                material(key),
            )

    for index, item in enumerate(scene.manifest.get("landscape_objects", [])):
        if item.get("kind") in ("tree", "shrub"):
            asset = library.load_assets().get(item["asset_key"])
            if asset is None:
                continue
            ox, oy, oz = item["position"]
            angle = math.radians(float(item.get("rotation_deg", 0.0)))
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for part in asset.parts:
                corners = [
                    (part.x0, part.y0), (part.x1, part.y0),
                    (part.x1, part.y1), (part.x0, part.y1),
                ]
                rotated = [(ox + cx * cos_a - cy * sin_a, oy + cx * sin_a + cy * cos_a) for cx, cy in corners]
                xs = [p[0] for p in rotated]
                ys = [p[1] for p in rotated]
                verts = [
                    (min(xs), min(ys), oz + part.z0), (max(xs), min(ys), oz + part.z0),
                    (max(xs), max(ys), oz + part.z0), (min(xs), max(ys), oz + part.z0),
                    (min(xs), min(ys), oz + part.z1), (max(xs), min(ys), oz + part.z1),
                    (max(xs), max(ys), oz + part.z1), (min(xs), max(ys), oz + part.z1),
                ]
                import numpy as _np

                from ..modeling.mesh import _BOX_FACES

                key = item["materials"].get(part.mat, "leaf_deep")
                if key.startswith("leaf"):
                    # Subdivide foliage cubes so the silhouette reads as planting
                    # rather than a block. Deterministic: driven by the part index.
                    for sub_verts in _subdivide_foliage(verts, part.name, index):
                        mesh.add(
                            _np.array(sub_verts, dtype=float), _BOX_FACES.copy(),
                            ObjectRef(
                                id=f"landscape:{item['asset_key']}:{index}:{part.name}",
                                name=asset.display_name, kind="landscape", level=0,
                                material=key, provenance=item.get("provenance", "default"),
                            ),
                            material(key),
                        )
                    continue
                mesh.add(
                    _np.array(verts, dtype=float), _BOX_FACES.copy(),
                    ObjectRef(
                        id=f"landscape:{item['asset_key']}:{index}:{part.name}",
                        name=asset.display_name, kind="landscape", level=0,
                        material=key, provenance=item.get("provenance", "default"),
                    ),
                    material(key),
                )
        elif item.get("kind") == "paving":
            import numpy as _np

            from ..modeling.mesh import box

            px, py, pz = item["position"]
            sx, sy, sz = item["size"]
            verts, faces = box(px - sx / 2, py - sy / 2, pz, px + sx / 2, py + sy / 2, pz + sz)
            key = item.get("material", "paving")
            mesh.add(
                verts, faces,
                ObjectRef(id=f"landscape:{item['asset_key']}:{index}",
                          name=item["asset_key"], kind="landscape", level=0,
                          material=key, provenance=item.get("provenance", "default")),
                material(key),
            )

    # Vertical glazing detail is already in the structural mesh; nothing else to add.
    _ = oriented_box
    return mesh


def _subdivide_foliage(verts, part_name: str, index: int) -> list[list[tuple[float, float, float]]]:
    """Split one foliage box into a jittered 2x2x2 cluster of smaller boxes."""
    import hashlib

    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    x0, x1, y0, y1, z0, z1 = min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)
    mx, my, mz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    if part_name.startswith("crown_top"):
        cells = [
            (x0, mx, y0, my, mz, z1), (mx, x1, y0, my, mz, z1),
            (x0, mx, my, y1, mz, z1), (mx, x1, my, y1, mz, z1),
        ]
    else:
        cells = [
            (x0, mx, y0, my, z0, mz), (mx, x1, y0, my, z0, mz),
            (x0, mx, my, y1, z0, mz), (mx, x1, my, y1, mz, mz),
            (x0, mx, y0, my, mz, z1), (mx, x1, y0, my, mz, z1),
            (x0, mx, my, y1, mz, z1), (mx, x1, my, y1, mz, z1),
        ]
    out = []
    for cell_index, (ax, bx, ay, by, az, bz) in enumerate(cells):
        digest = hashlib.sha256(f"{part_name}:{index}:{cell_index}".encode()).digest()
        jitter = [(digest[i] / 255.0 - 0.5) * 0.22 for i in range(3)]
        shrink = 0.12
        ax, bx = ax - shrink + jitter[0], bx + shrink + jitter[0]
        ay, by = ay - shrink + jitter[1], by + shrink + jitter[1]
        az, bz = az - shrink + jitter[2], bz + shrink + jitter[2]
        out.append([
            (ax, ay, az), (bx, ay, az), (bx, by, az), (ax, by, az),
            (ax, ay, bz), (bx, ay, bz), (bx, by, bz), (ax, by, bz),
        ])
    return out


def styled_scene_mesh(model: BuildingModel, scene: Scene) -> Mesh:
    """Structural + design combined, with palette materials applied to surfaces."""
    combined = Mesh()
    combined.merge(model.mesh)
    combined.merge(scene.design_mesh)
    return combined
