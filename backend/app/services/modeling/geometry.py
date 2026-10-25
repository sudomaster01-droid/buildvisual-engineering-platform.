"""Deterministic geometry builder: HouseSpec → triangle mesh.

Blueprint p.40: "Given the same schema version and builder version, the builder
should produce equivalent geometry." There is no randomness here at all; visual
variation lives in the styling layer. Every mesh element records the spec id it
came from so a later correction can be traced.

What this builder does **not** do: it does not claim to reconstruct arbitrary
CAD drawings. It constructs a building from a validated, typed plan. Reading
plan linework out of a raster drawing is the interpreter's job, and its output
is explicitly labelled with per-field confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...errors import GeometryError
from ..spec.geometry_helpers import bbox as bounds_of
from ..spec.geometry_helpers import polygon_area
from ..spec.schema import (
    HouseSpec,
    Level,
    OpeningKind,
    RoofType,
    Room,
    SourceType,
    Wall,
)
from .mesh import (
    Material,
    Mesh,
    ObjectRef,
    extrude_polygon,
    oriented_box,
    quad_slope,
    triangulate_polygon,
)

BUILDER_VERSION = "1.0.0"

#: Slab thickness and other structural constants live here, not scattered.
SLAB_THICKNESS = 0.22


# ---------------------------------------------------------------------------
# Material catalogue (structural defaults; the design layer may override via
# the scene manifest, which never rewrites geometry).
# ---------------------------------------------------------------------------

STRUCTURAL_MATERIALS: dict[str, Material] = {
    "wall_exterior": Material("wall_exterior", "Exterior render", (0.86, 0.84, 0.80), 0.85, kind="wall"),
    "wall_interior": Material("wall_interior", "Interior plaster", (0.93, 0.92, 0.90), 0.9, kind="wall"),
    "slab": Material("slab", "Floor slab", (0.55, 0.53, 0.50), 0.85, kind="floor"),
    "roof_tile": Material("roof_tile", "Roof tile", (0.42, 0.25, 0.20), 0.75, kind="roof"),
    "roof_metal": Material("roof_metal", "Standing-seam roof", (0.42, 0.44, 0.46), 0.35, metallic=0.6, kind="roof"),
    "glass": Material("glass", "Glazing", (0.62, 0.74, 0.80), 0.08, metallic=0.1, kind="glass"),
    "door_leaf": Material("door_leaf", "Timber door", (0.35, 0.24, 0.16), 0.55, kind="wood"),
    "stair": Material("stair", "Stair tread", (0.68, 0.62, 0.55), 0.7, kind="wood"),
    "ground": Material("ground", "Ground", (0.47, 0.55, 0.36), 0.95, kind="ground"),
}


@dataclass
class BuildingModel:
    """The locked structural artifact produced by this builder."""

    mesh: Mesh
    spec: HouseSpec
    builder_version: str = BUILDER_VERSION
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    level_elevations: dict[int, float] = field(default_factory=dict)
    room_areas: dict[str, float] = field(default_factory=dict)

    @property
    def geometry_hash(self) -> str:
        return self.mesh.geometry_hash()

    def manifest(self) -> dict[str, Any]:
        lo, hi = self.mesh.bounds
        return {
            "builder_version": self.builder_version,
            "schema_version": self.spec.schema_version,
            "geometry_hash": self.geometry_hash,
            "units": "m",
            "coordinate_system": "+x east, +y north, +z up; origin at footprint SW corner",
            "bounds": {"min": lo.round(4).tolist(), "max": hi.round(4).tolist()},
            "counts": {
                "objects": len(self.mesh.objects),
                "vertices": int(len(self.mesh.vertices)),
                "triangles": int(len(self.mesh.faces)),
            },
            "metrics": self.metrics,
            "warnings": self.warnings,
            "levels": [
                {"index": lvl.index, "name": lvl.name, "elevation_m": lvl.elevation_m,
                 "ceiling_height_m": lvl.ceiling_height_m}
                for lvl in self.spec.levels
            ],
            "rooms": [
                {"id": r.id, "name": r.name, "type": r.type.value, "level": r.level,
                 "area_m2": round(self.room_areas.get(r.id, 0.0), 2),
                 "source": r.source.value, "confidence": r.confidence}
                for r in self.spec.rooms
            ],
            "objects": [
                {"id": o.id, "name": o.name, "kind": o.kind, "level": o.level,
                 "room_id": o.room_id, "material": o.material,
                 "provenance": o.provenance}
                for o in self.mesh.objects
            ],
            "materials": [
                {"key": m.key, "name": m.name, "color": list(m.color),
                 "roughness": m.roughness, "metallic": m.metallic, "kind": m.kind,
                 "source": m.source}
                for m in self.mesh.materials
            ],
        }


class GeometryBuilder:
    def __init__(self, spec: HouseSpec) -> None:
        self.spec = spec
        self.mesh = Mesh()
        self.warnings: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self.room_areas: dict[str, float] = {}
        self.level_elevations = {lvl.index: lvl.elevation_m for lvl in spec.levels}

    # -- entry point -----------------------------------------------------
    def build(self) -> BuildingModel:
        if not self.spec.rooms:
            raise GeometryError("The house definition contains no rooms.")
        self._build_slabs()
        self._build_walls()
        self._build_stairs()
        self._build_roof()
        self._build_ground()
        self._metrics()
        return BuildingModel(
            mesh=self.mesh,
            spec=self.spec,
            metrics=self.metrics,
            warnings=self.warnings,
            level_elevations=self.level_elevations,
            room_areas=self.room_areas,
        )

    # -- helpers ---------------------------------------------------------
    def _level(self, index: int) -> Level:
        for lvl in self.spec.levels:
            if lvl.index == index:
                return lvl
        return Level(index=index)

    def _warn(self, code: str, message: str, **detail: Any) -> None:
        self.warnings.append({"code": code, "message": message, **detail})

    def _material(self, key: str, room: Room | None = None) -> Material:
        base = STRUCTURAL_MATERIALS.get(key) or STRUCTURAL_MATERIALS["slab"]
        if room is not None and key in ("wall_interior", "slab"):
            # Slight per-room tint keeps interiors readable without breaking determinism.
            tint = _deterministic_tint(room.id, amplitude=0.035)
            color = tuple(min(1.0, max(0.0, c + t)) for c, t in zip(base.color, tint))
            return Material(
                key=base.key, name=base.name, color=color, roughness=base.roughness,
                metallic=base.metallic, kind=base.kind, source=base.source,
            )
        return base

    # -- slabs -----------------------------------------------------------
    def _build_slabs(self) -> None:
        for room in self.spec.rooms:
            elevation = self._level(room.level).elevation_m
            self.room_areas[room.id] = polygon_area(room.polygon)
            poly = np.asarray(room.polygon, dtype=np.float64)
            verts, faces = extrude_polygon(poly, elevation - SLAB_THICKNESS, elevation)
            self.mesh.add(
                verts,
                faces,
                ObjectRef(
                    id=f"slab:{room.id}",
                    name=f"{room.name} floor",
                    kind="slab",
                    room_id=room.id,
                    level=room.level,
                    material="slab",
                    provenance=room.source.value,
                ),
                self._material("slab", room),
            )
            if room.level == max(lvl.index for lvl in self.spec.levels):
                # Cap the top storey so the roof void is not visible from outside.
                ceil_verts, ceil_faces = extrude_polygon(
                    poly, elevation + room.ceiling_height_m,
                    elevation + room.ceiling_height_m + 0.14,
                )
                self.mesh.add(
                    ceil_verts, ceil_faces,
                    ObjectRef(
                        id=f"ceiling:{room.id}", name=f"{room.name} ceiling",
                        kind="slab", room_id=room.id, level=room.level,
                        material="wall_interior", provenance=room.source.value,
                    ),
                    self._material("wall_interior", room),
                )

    # -- walls -----------------------------------------------------------
    def _build_walls(self) -> None:
        for wall in self.spec.walls:
            self._build_wall(wall)

    def _build_wall(self, wall: Wall) -> None:
        level = self._level(wall.level)
        z0 = level.elevation_m
        z1 = level.elevation_m + min(wall.height_m, level.ceiling_height_m + 0.0)
        length = wall.length_m
        angle = wall.angle_rad
        ux, uy = math.cos(angle), math.sin(angle)
        self._validate_openings(wall)

        intervals: list[tuple[float, float]] = []
        cursor = 0.0
        for opening in sorted(wall.openings, key=lambda o: o.offset_m):
            start = max(0.0, opening.offset_m - opening.width_m / 2.0)
            end = min(length, opening.offset_m + opening.width_m / 2.0)
            if start - cursor > 0.02:
                intervals.append((cursor, start))
            cursor = max(cursor, end)
        if length - cursor > 0.02:
            intervals.append((cursor, length))
        if not intervals and not wall.openings:
            intervals = [(0.0, length)]

        material_key = "wall_interior" if wall.interior else "wall_exterior"
        room = self.spec.room_by_id(wall.openings[0].connects[0]) if (
            wall.openings and wall.openings[0].connects
        ) else None
        material = self._material(material_key, room)

        for index, (a, b) in enumerate(intervals):
            seg_len = b - a
            if seg_len <= 0.02:
                continue
            centre = (wall.start[0] + ux * (a + seg_len / 2.0),
                      wall.start[1] + uy * (a + seg_len / 2.0))
            verts, faces = oriented_box(centre, seg_len, wall.thickness_m, z0, z1, angle)
            self.mesh.add(
                verts,
                faces,
                ObjectRef(
                    id=f"{wall.id}:seg{index}",
                    name=f"Wall {wall.id}",
                    kind="wall",
                    level=wall.level,
                    material=material_key,
                    provenance=wall.source.value,
                ),
                material,
            )

        # lintels, sills and glazing/door leaves
        for opening in wall.openings:
            centre = (wall.start[0] + ux * opening.offset_m, wall.start[1] + uy * opening.offset_m)
            open_z0 = z0 + opening.sill_m
            open_z1 = open_z0 + opening.height_m
            if opening.sill_m > 0.02:  # sill / spandrel below a window
                verts, faces = oriented_box(
                    centre, opening.width_m, wall.thickness_m, z0, open_z0, angle
                )
                self.mesh.add(
                    verts, faces,
                    ObjectRef(id=f"{opening.id}:sill", name=f"Wall below {opening.id}",
                              kind="wall", level=wall.level, material=material_key,
                              provenance=wall.source.value),
                    material,
                )
            if open_z1 < z1 - 0.02:  # lintel above
                verts, faces = oriented_box(
                    centre, opening.width_m, wall.thickness_m, open_z1, z1, angle
                )
                self.mesh.add(
                    verts, faces,
                    ObjectRef(id=f"{opening.id}:lintel", name=f"Wall above {opening.id}",
                              kind="wall", level=wall.level, material=material_key,
                              provenance=wall.source.value),
                    material,
                )
            if opening.kind is OpeningKind.WINDOW:
                verts, faces = oriented_box(
                    centre, opening.width_m * 0.96, 0.04, open_z0, open_z1, angle
                )
                self.mesh.add(
                    verts, faces,
                    ObjectRef(id=f"{opening.id}:glazing", name="Window",
                              kind="fixture", level=wall.level, material="glass",
                              provenance=wall.source.value),
                    STRUCTURAL_MATERIALS["glass"],
                )
            elif opening.kind is OpeningKind.DOOR and not opening.is_entrance:
                # Leave interior doors open so the camera route has clear sightlines.
                continue

    def _validate_openings(self, wall: Wall) -> None:
        length = wall.length_m
        cursor = -1.0
        for opening in sorted(wall.openings, key=lambda o: o.offset_m):
            start = opening.offset_m - opening.width_m / 2.0
            end = opening.offset_m + opening.width_m / 2.0
            if end - start > length + 1e-6:
                raise GeometryError(
                    f"Opening {opening.id} is wider than its host wall {wall.id}."
                )
            if start < -0.05 or end > length + 0.05:
                raise GeometryError(
                    f"Opening {opening.id} extends outside its host wall {wall.id}."
                )
            if start < cursor:
                self._warn(
                    "openings_overlap",
                    f"Openings on wall {wall.id} overlap; geometry may be ambiguous.",
                    wall_id=wall.id,
                )
            cursor = end

    # -- stairs ----------------------------------------------------------
    def _build_stairs(self) -> None:
        for stair in self.spec.stairs:
            z0 = self._level(stair.from_level).elevation_m
            z1 = self._level(stair.to_level).elevation_m
            rise = z1 - z0
            if rise <= 0.2:
                self._warn("stair_degenerate", f"Stair {stair.id} has no usable rise.",
                           stair_id=stair.id)
                continue
            steps = max(3, int(round(rise / 0.18)))
            dx = stair.end[0] - stair.start[0]
            dy = stair.end[1] - stair.start[1]
            run_len = math.hypot(dx, dy)
            if run_len < 0.5:
                raise GeometryError(f"Stair {stair.id} has no run; a climbable stair needs length.")
            angle = math.atan2(dy, dx)
            for i in range(steps):
                t = (i + 0.5) / steps
                cx = stair.start[0] + dx * t
                cy = stair.start[1] + dy * t
                tread_z0 = z0
                tread_z1 = z0 + rise * (i + 1) / steps
                verts, faces = oriented_box(
                    (cx, cy), run_len / steps * 1.02, stair.width_m, tread_z0, tread_z1, angle
                )
                self.mesh.add(
                    verts, faces,
                    ObjectRef(id=f"{stair.id}:step{i}", name=f"Stair {stair.id}",
                              kind="stair", level=stair.from_level, material="stair",
                              provenance=stair.source.value),
                    STRUCTURAL_MATERIALS["stair"],
                )

    # -- roof ------------------------------------------------------------
    def _footprint(self) -> tuple[float, float, float, float]:
        top_level = max(lvl.index for lvl in self.spec.levels)
        rooms = [r for r in self.spec.rooms if r.level == top_level] or self.spec.rooms
        points = [p for r in rooms for p in r.polygon]
        if not points:
            raise GeometryError("No room footprints available to place a roof.")
        return bounds_of(points)

    def _build_roof(self) -> None:
        top_level = max(lvl.index for lvl in self.spec.levels)
        top = self._level(top_level)
        # The roof soffit sits just above the ceiling slab: coplanar faces would
        # z-fight and stripe the ceiling when the camera is indoors.
        eave_z = top.elevation_m + top.ceiling_height_m + 0.16
        x0, y0, x1, y1 = self._footprint()
        overhang = self.spec.roof.overhang_m
        x0 -= overhang
        y0 -= overhang
        x1 += overhang
        y1 += overhang

        roof = self.spec.roof
        roof_type = roof.type
        if roof_type is RoofType.UNKNOWN:
            roof_type = RoofType.GABLE
            self._warn(
                "roof_type_assumed",
                "Roof type was not supplied; a low-pitch gable roof was assumed.",
            )
        pitch = math.radians(max(roof.pitch_deg, 10.0) if roof_type is not RoofType.FLAT else 0.0)
        material = STRUCTURAL_MATERIALS.get(roof.material_key) or STRUCTURAL_MATERIALS["roof_tile"]
        obj = ObjectRef(id="roof", name="Roof", kind="roof", level=top_level,
                        material=roof.material_key, provenance=roof.source.value)

        if roof_type is RoofType.FLAT:
            verts, faces = extrude_polygon(
                np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=np.float64),
                eave_z, eave_z + 0.18,
            )
            self.mesh.add(verts, faces, obj, material)
            return

        width = x1 - x0
        depth = y1 - y0
        if roof_type is RoofType.SHED:
            rise = math.tan(pitch) * depth
            verts = np.array([
                [x0, y0, eave_z], [x1, y0, eave_z],
                [x1, y1, eave_z + rise], [x0, y1, eave_z + rise],
            ])
            faces = np.array([[0, 1, 2], [0, 2, 3]])
            self.mesh.add(verts, faces, obj, material)
            self._roof_end_caps(x0, y0, x1, y1, eave_z, [(y0, eave_z), (y1, eave_z + rise)],
                                material, obj)
            return

        if roof_type is RoofType.HIP:
            if width >= depth:
                ridge_half = max(0.0, (width - depth) / 2.0)
                rx0, rx1 = x0 + ridge_half, x1 - ridge_half
                ry = (y0 + y1) / 2.0
                rise = math.tan(pitch) * (depth / 2.0)
                ridge_z = eave_z + rise
                verts = np.array([
                    [x0, y0, eave_z], [x1, y0, eave_z], [x1, y1, eave_z], [x0, y1, eave_z],
                    [rx0, ry, ridge_z], [rx1, ry, ridge_z],
                ])
                faces = np.array([
                    [0, 1, 5], [0, 5, 4],   # south slope
                    [2, 3, 4], [2, 4, 5],   # north slope
                    [1, 2, 5],              # east hip
                    [3, 0, 4],              # west hip
                ])
            else:
                ridge_half = max(0.0, (depth - width) / 2.0)
                ry0, ry1 = y0 + ridge_half, y1 - ridge_half
                rx = (x0 + x1) / 2.0
                rise = math.tan(pitch) * (width / 2.0)
                ridge_z = eave_z + rise
                verts = np.array([
                    [x0, y0, eave_z], [x1, y0, eave_z], [x1, y1, eave_z], [x0, y1, eave_z],
                    [rx, ry0, ridge_z], [rx, ry1, ridge_z],
                ])
                faces = np.array([
                    [1, 2, 5], [1, 5, 4],
                    [3, 0, 4], [3, 4, 5],
                    [0, 1, 4],
                    [2, 3, 5],
                ])
            self.mesh.add(verts, faces, obj, material)
            return

        # GABLE (default for gable + fallback)
        if width >= depth:
            ridge_z = eave_z + math.tan(pitch) * (depth / 2.0)
            ry = (y0 + y1) / 2.0
            verts = np.array([
                [x0, y0, eave_z], [x1, y0, eave_z], [x1, y1, eave_z], [x0, y1, eave_z],
                [x0, ry, ridge_z], [x1, ry, ridge_z],
            ])
            faces = np.array([[0, 1, 5], [0, 5, 4], [2, 3, 4], [2, 4, 5]])
            gable_ends = [
                np.array([[x0, y0, eave_z], [x0, ry, ridge_z], [x0, y1, eave_z]]),
                np.array([[x1, y1, eave_z], [x1, ry, ridge_z], [x1, y0, eave_z]]),
            ]
        else:
            ridge_z = eave_z + math.tan(pitch) * (width / 2.0)
            rx = (x0 + x1) / 2.0
            verts = np.array([
                [x0, y0, eave_z], [x1, y0, eave_z], [x1, y1, eave_z], [x0, y1, eave_z],
                [rx, y0, ridge_z], [rx, y1, ridge_z],
            ])
            faces = np.array([[1, 2, 5], [1, 5, 4], [3, 0, 4], [3, 4, 5]])
            gable_ends = [
                np.array([[x0, y0, eave_z], [x0, y1, eave_z], [rx, y1, ridge_z], [rx, y0, ridge_z]]),
                np.array([[x1, y1, eave_z], [x1, y0, eave_z], [rx, y0, ridge_z], [rx, y1, ridge_z]]),
            ]
        self.mesh.add(verts, faces, obj, material)
        for i, cap in enumerate(gable_ends):
            cap_faces = (
                np.array([[0, 1, 2]]) if len(cap) == 3
                else np.array([[0, 1, 2], [0, 2, 3]])
            )
            self.mesh.add(
                cap, cap_faces,
                ObjectRef(id=f"roof:gable{i}", name="Gable end", kind="roof",
                          level=top_level, material=roof.material_key,
                          provenance=roof.source.value),
                material,
            )

    def _roof_end_caps(self, x0, y0, x1, y1, eave_z, profile, material, obj) -> None:
        for sign, x in (("w", x0), ("e", x1)):
            (ay, az), (by, bz) = profile
            cap = np.array([
                [x, ay, eave_z], [x, by, eave_z], [x, by, bz], [x, ay, az],
            ])
            self.mesh.add(cap, np.array([[0, 1, 2], [0, 2, 3]]),
                          ObjectRef(id=f"roof:cap{sign}", name="Roof end", kind="roof",
                                    material=obj.material, provenance=obj.provenance),
                          material)

    # -- site ------------------------------------------------------------
    def _build_ground(self) -> None:
        """Site plane, tessellated into a grid.

        A single huge quad would be dropped by the renderer's near-plane test
        (any vertex behind the camera culls the triangle), leaving holes in
        frames. A grid of modest cells clips cleanly and lets distance haze fade
        the far edge into the sky.
        """
        x0, y0, x1, y1 = self._footprint()
        pad = 120.0
        cells = 24
        gx0, gy0 = x0 - pad, y0 - pad
        gx1, gy1 = x1 + pad, y1 + pad
        step_x = (gx1 - gx0) / cells
        step_y = (gy1 - gy0) / cells

        verts: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        for row in range(cells):
            for col in range(cells):
                ax = gx0 + col * step_x
                ay = gy0 + row * step_y
                bx = ax + step_x
                by = ay + step_y
                base = len(verts)
                # z = 0 keeps planting and paths sitting exactly on the surface.
                verts.extend([
                    (ax, ay, 0.0), (bx, ay, 0.0), (bx, by, 0.0), (ax, by, 0.0),
                ])
                faces.append((base, base + 1, base + 2))
                faces.append((base, base + 2, base + 3))

        self.mesh.add(
            np.array(verts, dtype=np.float64),
            np.array(faces, dtype=np.int32),
            ObjectRef(id="site_ground", name="Site ground", kind="landscape",
                      material="ground", provenance=SourceType.SYSTEM_DEFAULT.value),
            STRUCTURAL_MATERIALS["ground"],
        )

    # -- metrics ---------------------------------------------------------
    def _metrics(self) -> None:
        total_wall = sum(w.length_m * w.height_m for w in self.spec.walls)
        opening_area = sum(
            o.width_m * o.height_m for w in self.spec.walls for o in w.openings
        )
        lo, hi = self.mesh.bounds
        self.metrics = {
            "floor_area_m2": round(sum(self.room_areas.values()), 2),
            "gross_volume_m3": round(
                sum(
                    self.room_areas[r.id] * self._level(r.level).ceiling_height_m
                    for r in self.spec.rooms
                    if r.id in self.room_areas
                ), 2
            ),
            "wall_area_m2": round(total_wall, 2),
            "opening_area_m2": round(opening_area, 2),
            "window_wall_ratio": round(
                opening_area / total_wall, 3) if total_wall else 0.0,
            "footprint_m2": round(polygon_area([
                (lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])
            ]), 2),
            "levels": len(self.spec.levels),
            "triangles": int(len(self.mesh.faces)),
            "vertices": int(len(self.mesh.vertices)),
            "objects": len(self.mesh.objects),
            "extent_m": [round(float(hi[i] - lo[i]), 2) for i in range(3)],
        }


def _deterministic_tint(seed: str, amplitude: float = 0.03) -> tuple[float, float, float]:
    """A tiny, stable colour offset derived from an id (no RNG, no wall clock)."""
    import hashlib

    digest = hashlib.sha256(seed.encode()).digest()
    return tuple(
        ((digest[i] / 255.0) - 0.5) * 2.0 * amplitude for i in range(3)
    )  # type: ignore[return-value]


def build_model(spec: HouseSpec) -> BuildingModel:
    """Public entry point used by the pipeline."""
    return GeometryBuilder(spec).build()


def model_to_glb_bytes(model: BuildingModel) -> bytes:
    from .exporter import mesh_to_glb

    return mesh_to_glb(model.mesh, metadata={
        "generator": "BuildVisual",
        "builder_version": model.builder_version,
        "geometry_hash": model.geometry_hash,
    })
