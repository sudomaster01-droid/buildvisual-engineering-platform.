"""The typed house specification — the contract between interpretation and geometry.

Blueprint p.34: every numeric dimension needs explicit units and a valid range;
ids are unique; openings reference a host wall; stairs connect known levels;
roof type is an enum or ``unknown``. Blueprint p.8/p.36: every fact carries its
source, confidence and criticality so that inferred details can never be
presented as measured facts.

Coordinates are metres, ``+x`` east, ``+y`` north, ``+z`` up, origin at the
building's south-west footprint corner.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"

Point2 = tuple[float, float]


class SourceType(StrEnum):
    """Provenance of a value. Ordering matters for merge priority."""

    DRAWING_DIMENSION = "drawing_dimension"  # measured annotation on a drawing
    DRAWING_LABEL = "drawing_label"          # room name / text on a drawing
    USER_STATED = "user_stated"              # stated by the engineer in text/voice
    DERIVED = "derived"                      # computed deterministically from others
    SYSTEM_DEFAULT = "system_default"        # chosen by the platform
    INFERRED = "inferred"                    # guessed by a model


_SOURCE_RANK = {
    SourceType.DRAWING_DIMENSION: 5,
    SourceType.USER_STATED: 4,
    SourceType.DRAWING_LABEL: 3,
    SourceType.DERIVED: 3,
    SourceType.INFERRED: 2,
    SourceType.SYSTEM_DEFAULT: 1,
}


class RoomType(StrEnum):
    LIVING = "living"
    DINING = "dining"
    KITCHEN = "kitchen"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    HALL = "hall"
    STUDY = "study"
    UTILITY = "utility"
    GARAGE = "garage"
    STAIR = "stair"
    BALCONY = "balcony"
    OTHER = "other"


class RoofType(StrEnum):
    GABLE = "gable"
    HIP = "hip"
    FLAT = "flat"
    SHED = "shed"
    UNKNOWN = "unknown"


class OpeningKind(StrEnum):
    DOOR = "door"
    WINDOW = "window"
    OPENING = "opening"  # cased opening / archway, no leaf


class Provenanced(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: SourceType = SourceType.USER_STATED
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    @property
    def is_factual(self) -> bool:
        return self.source in (
            SourceType.DRAWING_DIMENSION,
            SourceType.DRAWING_LABEL,
            SourceType.USER_STATED,
        )


class Opening(Provenanced):
    id: str
    kind: OpeningKind = OpeningKind.DOOR
    #: distance from the wall's start point to the opening's centre, metres
    offset_m: float = Field(ge=0.0, le=200.0)
    width_m: float = Field(gt=0.0, le=20.0)
    height_m: float = Field(gt=0.0, le=8.0)
    sill_m: float = Field(default=0.0, ge=0.0, le=5.0)
    #: rooms joined by this opening (empty = exterior door)
    connects: list[str] = Field(default_factory=list)
    is_entrance: bool = False

    @model_validator(mode="after")
    def _sill_below_top(self) -> "Opening":
        if self.sill_m + self.height_m > 6.5:
            raise ValueError(f"opening {self.id}: sill+height exceeds plausible wall height")
        if self.kind is OpeningKind.WINDOW and self.sill_m <= 0:
            self.sill_m = 0.9
        return self


class Wall(Provenanced):
    id: str
    level: int = 0
    start: Point2
    end: Point2
    thickness_m: float = Field(default=0.15, gt=0.02, le=1.0)
    height_m: float = Field(default=2.7, gt=1.8, le=6.0)
    interior: bool = False
    openings: list[Opening] = Field(default_factory=list)

    @property
    def length_m(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return float((dx * dx + dy * dy) ** 0.5)

    @property
    def angle_rad(self) -> float:
        import math

        return math.atan2(self.end[1] - self.start[1], self.end[0] - self.start[0])

    @field_validator("start", "end")
    @classmethod
    def _finite_point(cls, v: Point2) -> Point2:
        if len(v) != 2:
            raise ValueError("point must have exactly two coordinates")
        x, y = float(v[0]), float(v[1])
        if abs(x) > 500 or abs(y) > 500:
            raise ValueError("coordinate outside the plausible site extent (±500 m)")
        return (x, y)

    @model_validator(mode="after")
    def _degenerate(self) -> "Wall":
        if self.length_m < 0.2:
            raise ValueError(f"wall {self.id} is shorter than 0.2 m")
        return self


class Room(Provenanced):
    id: str
    name: str
    type: RoomType = RoomType.OTHER
    level: int = 0
    polygon: list[Point2]
    ceiling_height_m: float = Field(default=2.7, gt=1.8, le=6.0)

    @field_validator("polygon")
    @classmethod
    def _polygon(cls, v: list[Point2]) -> list[Point2]:
        if len(v) < 3:
            raise ValueError("room polygon needs at least three points")
        return [(float(p[0]), float(p[1])) for p in v]

    @model_validator(mode="after")
    def _area(self) -> "Room":
        from .geometry_helpers import polygon_area

        if polygon_area(self.polygon) < 0.8:
            raise ValueError(f"room {self.id} has an implausibly small area")
        return self


class Stair(Provenanced):
    id: str
    from_level: int
    to_level: int
    start: Point2
    end: Point2
    width_m: float = Field(default=1.0, gt=0.5, le=3.0)
    #: open wells make a camera path unsafe; recorded explicitly
    has_landing: bool = True

    @model_validator(mode="after")
    def _levels(self) -> "Stair":
        if self.to_level <= self.from_level:
            raise ValueError(f"stair {self.id}: to_level must exceed from_level")
        return self


class Level(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0, le=6)
    name: str = "Ground floor"
    elevation_m: float = Field(default=0.0, ge=-10.0, le=60.0)
    ceiling_height_m: float = Field(default=2.7, gt=1.8, le=6.0)


class Roof(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: RoofType = RoofType.UNKNOWN
    pitch_deg: float = Field(default=25.0, ge=0.0, le=70.0)
    overhang_m: float = Field(default=0.6, ge=0.0, le=2.5)
    material_key: str = "roof_tile"
    source: SourceType = SourceType.SYSTEM_DEFAULT
    confidence: float = 0.5


class Site(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plot_width_m: float | None = Field(default=None, ge=5.0, le=400.0)
    plot_depth_m: float | None = Field(default=None, ge=5.0, le=400.0)
    #: building rotation on the plot, degrees clockwise from north-up
    orientation_deg: float = Field(default=0.0, ge=-360.0, le=360.0)
    ground_material: str = "lawn"
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    #: ISO local time used for the sun angle, if supplied
    local_time: str | None = None
    context_note: str | None = None


class DesignIntent(BaseModel):
    """Explicit preferences the engineer expressed. Empty by default."""

    model_config = ConfigDict(extra="forbid")

    style: str | None = None
    palette: str | None = None
    materials: dict[str, str] = Field(default_factory=dict)
    atmosphere: str | None = None
    must_include: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)


class Unresolved(BaseModel):
    """A critical ambiguity that blocks (or endangers) model construction."""

    model_config = ConfigDict(extra="forbid")

    field: str
    description: str
    options: list[str] = Field(default_factory=list)
    critical: bool = True
    #: remaining entropy in bits over the plausible options
    entropy_bits: float = 0.0
    chosen: str | None = None


class HouseSpec(BaseModel):
    """Validated, versioned, provenance-carrying house definition."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    title: str = "House"
    unit: Literal["m"] = "m"

    levels: list[Level] = Field(default_factory=lambda: [Level(index=0)])
    rooms: list[Room] = Field(default_factory=list)
    walls: list[Wall] = Field(default_factory=list)
    stairs: list[Stair] = Field(default_factory=list)
    roof: Roof = Field(default_factory=Roof)
    site: Site = Field(default_factory=Site)
    design_intent: DesignIntent = Field(default_factory=DesignIntent)

    stated_requirements: list[str] = Field(default_factory=list)
    inferred_fields: list[str] = Field(default_factory=list)
    unresolved: list[Unresolved] = Field(default_factory=list)
    #: field path -> confidence, for critical fields only
    field_confidence: dict[str, float] = Field(default_factory=dict)
    #: interpretation provenance: which model, which source artifacts
    provenance: dict[str, Any] = Field(default_factory=dict)

    # -- integrity -------------------------------------------------------
    @model_validator(mode="after")
    def _unique_ids(self) -> "HouseSpec":
        for label, values in (
            ("room", [r.id for r in self.rooms]),
            ("wall", [w.id for w in self.walls]),
            ("stair", [s.id for s in self.stairs]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} ids")
        return self

    @model_validator(mode="after")
    def _levels_exist(self) -> "HouseSpec":
        level_ids = {lvl.index for lvl in self.levels}
        for coll, name in ((self.rooms, "room"), (self.walls, "wall")):
            for item in coll:
                if item.level not in level_ids:
                    raise ValueError(f"{name} {item.id} references unknown level {item.level}")
        return self

    # -- derived ---------------------------------------------------------
    @property
    def floor_area_m2(self) -> float:
        from .geometry_helpers import polygon_area

        return round(sum(polygon_area(r.polygon) for r in self.rooms), 2)

    @property
    def storey_count(self) -> int:
        return max(1, len(self.levels))

    @property
    def has_upper_floor(self) -> bool:
        return len(self.levels) > 1

    def room_by_id(self, room_id: str) -> Room | None:
        return next((r for r in self.rooms if r.id == room_id), None)

    def entrance_openings(self) -> list[tuple[Wall, Opening]]:
        found = [(w, o) for w in self.walls for o in w.openings if o.is_entrance]
        if found:
            return found
        exterior_doors = [
            (w, o)
            for w in self.walls
            for o in w.openings
            if o.kind is OpeningKind.DOOR and not o.connects
        ]
        return exterior_doors[:1]

    def factual_field_count(self) -> int:
        count = 0
        for wall in self.walls:
            if wall.is_factual:
                count += 1
            count += sum(1 for o in wall.openings if o.is_factual)
        count += sum(1 for r in self.rooms if r.is_factual)
        return count

    def summary(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "levels": len(self.levels),
            "rooms": len(self.rooms),
            "room_types": sorted({r.type.value for r in self.rooms}),
            "floor_area_m2": self.floor_area_m2,
            "walls": len(self.walls),
            "openings": sum(len(w.openings) for w in self.walls),
            "stairs": len(self.stairs),
            "roof": self.roof.type.value,
            "inferred_fields": len(self.inferred_fields),
            "unresolved": len(self.unresolved),
        }


def merge_confidence(spec: HouseSpec) -> dict[str, float]:
    """Field-level confidence map for the critical structural fields."""
    conf: dict[str, float] = {}
    for level in spec.levels:
        conf[f"levels[{level.index}].elevation_m"] = 0.95 if level.elevation_m else 0.9
    for room in spec.rooms:
        conf[f"rooms.{room.id}.polygon"] = room.confidence
        conf[f"rooms.{room.id}.type"] = room.confidence
    for wall in spec.walls:
        conf[f"walls.{wall.id}.geometry"] = wall.confidence
        conf[f"walls.{wall.id}.height_m"] = wall.confidence
    for stair in spec.stairs:
        conf[f"stairs.{stair.id}"] = stair.confidence
    conf["roof.type"] = spec.roof.confidence
    return {k: round(v, 3) for k, v in conf.items()}
