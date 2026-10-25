"""Small, dependency-free triangle-mesh utilities.

Geometry is built by explicitly appending primitives rather than by
constructing a scene graph, because every mesh element must carry a stable
``object_id`` and ``material`` for segmentation masks and QA comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    color: tuple[float, float, float]
    roughness: float = 0.7
    metallic: float = 0.0
    kind: str = "generic"  # wall | floor | roof | glass | wood | metal | fabric | foliage | ground | water
    #: provenance for the material catalogue entry
    source: str = "procedural:buildvisual"


@dataclass
class ObjectRef:
    id: str
    name: str
    kind: str  # wall | slab | roof | stair | furniture | landscape | fixture
    room_id: str | None = None
    level: int = 0
    material: str = ""
    #: 'explicit' (from the engineer/drawing) or 'inferred' (design completion)
    provenance: str = "explicit"


@dataclass
class Mesh:
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))
    face_material: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    face_object: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    materials: list[Material] = field(default_factory=list)
    objects: list[ObjectRef] = field(default_factory=list)
    _material_index: dict[str, int] = field(default_factory=dict)
    _object_index: dict[str, int] = field(default_factory=dict)

    # -- queries ---------------------------------------------------------
    def object_kind(self, index: int) -> str:
        if 0 <= index < len(self.objects):
            return getattr(self.objects[index], "kind", "") or ""
        return ""

    def bounds_excluding(self, kinds: tuple[str, ...] = ("landscape",)) -> tuple[np.ndarray, np.ndarray]:
        """Bounds of the building itself, ignoring site/landscape geometry.

        Camera framing must use the house, not the plot: including a 50 m ground
        plane makes every shot pull back until the building is a speck.
        """
        if len(self.faces) == 0:
            return np.zeros(3), np.zeros(3)
        keep = np.array(
            [self.object_kind(int(idx)) not in kinds for idx in self.face_object],
            dtype=bool,
        )
        if not keep.any():
            return self.bounds
        faces = self.faces[keep]
        verts = self.vertices[np.unique(faces)]
        return verts.min(axis=0), verts.max(axis=0)

    # -- registries ------------------------------------------------------
    def material_index(self, material: Material) -> int:
        if material.key not in self._material_index:
            self._material_index[material.key] = len(self.materials)
            self.materials.append(material)
        return self._material_index[material.key]

    def object_index(self, obj: ObjectRef) -> int:
        if obj.id not in self._object_index:
            self._object_index[obj.id] = len(self.objects)
            self.objects.append(obj)
        else:  # keep material/provenance consistent on repeated use
            self.objects[self._object_index[obj.id]] = obj
        return self._object_index[obj.id]

    # -- mutation --------------------------------------------------------
    def add(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        obj: ObjectRef,
        material: Material,
    ) -> None:
        vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        if len(faces) == 0:
            return
        base = len(self.vertices)
        self.vertices = (
            vertices if len(self.vertices) == 0 else np.vstack([self.vertices, vertices])
        )
        self.faces = faces if len(self.faces) == 0 else np.vstack([self.faces, faces + base])
        mat_i = self.material_index(material)
        obj_i = self.object_index(obj)
        self.face_material = np.concatenate(
            [self.face_material, np.full(len(faces), mat_i, dtype=np.int32)]
        )
        self.face_object = np.concatenate(
            [self.face_object, np.full(len(faces), obj_i, dtype=np.int32)]
        )

    def merge(self, other: "Mesh") -> None:
        """Append another mesh, remapping its registries."""
        base = len(self.vertices)
        self.vertices = (
            other.vertices if len(self.vertices) == 0
            else np.vstack([self.vertices, other.vertices])
        )
        self.faces = (
            other.faces if len(self.faces) == 0 else np.vstack([self.faces, other.faces + base])
        )
        mat_offset = len(self.materials)
        obj_offset = len(self.objects)
        for mat in other.materials:
            if mat.key not in self._material_index:
                self._material_index[mat.key] = len(self.materials)
                self.materials.append(mat)
        for obj in other.objects:
            if obj.id not in self._object_index:
                self._object_index[obj.id] = len(self.objects)
                self.objects.append(obj)
        # Remap per-face indices through the (possibly shared) registries.
        remapped_mat = np.array(
            [other.materials[i].key for i in other.face_material], dtype=object
        )
        self.face_material = np.concatenate([
            self.face_material,
            np.array([self._material_index[k] for k in remapped_mat], dtype=np.int32),
        ])
        remapped_obj = np.array([other.objects[i].id for i in other.face_object], dtype=object)
        self.face_object = np.concatenate([
            self.face_object,
            np.array([self._object_index[k] for k in remapped_obj], dtype=np.int32),
        ])
        del mat_offset, obj_offset

    # -- queries ---------------------------------------------------------
    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.vertices) == 0:
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def triangle_array(self) -> np.ndarray:
        """(M, 3, 3) array of triangle vertices."""
        return self.vertices[self.faces]

    def object_bounds(self, object_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        idx = self._object_index.get(object_id)
        if idx is None:
            return None
        faces = np.nonzero(self.face_object == idx)[0]
        if len(faces) == 0:
            return None
        verts = self.vertices[self.faces[faces].ravel()]
        return verts.min(axis=0), verts.max(axis=0)

    def geometry_hash(self, precision: int = 4) -> str:
        """Deterministic content hash of the geometry (blueprint p.43)."""
        import hashlib

        digest = hashlib.sha256()
        if len(self.vertices):
            rounded = np.round(self.vertices, precision)
            digest.update(rounded.astype(np.float64).tobytes())
            digest.update(self.faces.astype(np.int32).tobytes())
            digest.update(self.face_material.astype(np.int32).tobytes())
        digest.update("|".join(o.id for o in self.objects).encode())
        digest.update("|".join(m.key for m in self.materials).encode())
        return digest.hexdigest()


# ---------------------------------------------------------------------------
# Primitive builders (returned as (vertices, faces) index triplets)
# ---------------------------------------------------------------------------

_BOX_FACES = np.array(
    [
        [0, 2, 1], [0, 3, 2],  # bottom (z-)
        [4, 5, 6], [4, 6, 7],  # top (z+)
        [0, 1, 5], [0, 5, 4],  # y-
        [1, 2, 6], [1, 6, 5],  # x+
        [2, 3, 7], [2, 7, 6],  # y+
        [3, 0, 4], [3, 4, 7],  # x-
    ],
    dtype=np.int32,
)


def box(x0: float, y0: float, z0: float, x1: float, y1: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box with outward-facing (counter-clockwise) triangles."""
    xs = (min(x0, x1), max(x0, x1))
    ys = (min(y0, y1), max(y0, y1))
    zs = (min(z0, z1), max(z0, z1))
    verts = np.array(
        [
            [xs[0], ys[0], zs[0]], [xs[1], ys[0], zs[0]],
            [xs[1], ys[1], zs[0]], [xs[0], ys[1], zs[0]],
            [xs[0], ys[0], zs[1]], [xs[1], ys[0], zs[1]],
            [xs[1], ys[1], zs[1]], [xs[0], ys[1], zs[1]],
        ],
        dtype=np.float64,
    )
    return verts, _BOX_FACES.copy()


def oriented_box(
    center_xy: tuple[float, float],
    length: float,
    thickness: float,
    z0: float,
    z1: float,
    angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Box of given length/thickness rotated about Z, centred on ``center_xy``."""
    half_l, half_t = length / 2.0, thickness / 2.0
    local = np.array(
        [
            [-half_l, -half_t], [half_l, -half_t], [half_l, half_t], [-half_l, half_t],
        ]
    )
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    world = local @ rot.T + np.array(center_xy)
    verts = np.vstack([
        np.column_stack([world, np.full(4, min(z0, z1))]),
        np.column_stack([world, np.full(4, max(z0, z1))]),
    ])
    return verts, _BOX_FACES.copy()


def triangulate_polygon(polygon: np.ndarray) -> np.ndarray:
    """Ear-clipping triangulation for a simple 2-D polygon (CCW or CW)."""
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int32)
    if n == 3:
        return np.array([[0, 1, 2]], dtype=np.int32)

    area = 0.5 * float(
        np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    )
    indices = list(range(n))
    if area < 0:  # normalise to counter-clockwise
        indices.reverse()

    triangles: list[list[int]] = []
    guard = 0
    while len(indices) > 3 and guard < 4 * n * n:
        guard += 1
        ear_found = False
        for i in range(len(indices)):
            i0 = indices[(i - 1) % len(indices)]
            i1 = indices[i]
            i2 = indices[(i + 1) % len(indices)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-12:  # reflex or degenerate ear
                continue
            if any(
                _point_in_triangle(pts[j], a, b, c)
                for j in indices
                if j not in (i0, i1, i2)
            ):
                continue
            triangles.append([i0, i1, i2])
            indices.pop(i)
            ear_found = True
            break
        if not ear_found:  # fallback: fan triangulation of the remainder
            break
    if len(indices) == 3:
        triangles.append([indices[0], indices[1], indices[2]])
    elif len(indices) > 3:
        for i in range(1, len(indices) - 1):
            triangles.append([indices[0], indices[i], indices[i + 1]])
    return np.array(triangles, dtype=np.int32).reshape(-1, 3)


def _point_in_triangle(p, a, b, c) -> bool:
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    has_neg = (d1 < -1e-12) or (d2 < -1e-12) or (d3 < -1e-12)
    has_pos = (d1 > 1e-12) or (d2 > 1e-12) or (d3 > 1e-12)
    return not (has_neg and has_pos)


def extrude_polygon(
    polygon: np.ndarray, z0: float, z1: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a 2-D polygon between two heights, capping both ends."""
    poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    n = len(poly)
    tris2d = triangulate_polygon(poly)
    bottom = np.column_stack([poly, np.full(n, min(z0, z1))])
    top = np.column_stack([poly, np.full(n, max(z0, z1))])
    verts = np.vstack([bottom, top])

    faces: list[list[int]] = []
    for tri in tris2d:  # bottom cap, reversed winding
        faces.append([int(tri[2]), int(tri[1]), int(tri[0])])
    for tri in tris2d:  # top cap
        faces.append([int(tri[0]) + n, int(tri[1]) + n, int(tri[2]) + n])
    for i in range(n):  # side walls
        j = (i + 1) % n
        faces.append([i, j, j + n])
        faces.append([i, j + n, i + n])
    return verts, np.array(faces, dtype=np.int32).reshape(-1, 3)


def quad_slope(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    d: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Two triangles for a planar quad given in counter-clockwise order."""
    verts = np.array([a, b, c, d], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, faces


def polygon_area(polygon: np.ndarray) -> float:
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    return abs(
        0.5
        * float(
            np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
        )
    )


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros(2)
    area = polygon_area(pts)
    if area < 1e-9:
        return pts.mean(axis=0)
    xs, ys = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    cross = xs * yn - xn * ys
    cx = float(np.sum((xs + xn) * cross) / (6 * (area if area else 1)))
    cy = float(np.sum((ys + yn) * cross) / (6 * (area if area else 1)))
    return np.array([cx, cy])


def rotate_about_z(points: np.ndarray, angle_rad: float, origin=(0.0, 0.0)) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).copy()
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return (pts - np.array(origin)) @ rot.T + np.array(origin)
