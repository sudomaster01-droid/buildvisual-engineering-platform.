"""A small, deterministic software renderer (NumPy z-buffer).

Why this exists: the hybrid pipeline needs *model-derived* reference images
(shaded view, metric depth, edge map, object segmentation) on a machine with no
GPU, no OpenGL context and no desktop 3D application. Blueprint p.57 is explicit
that these artifacts are guidance for a video model — not a hard geometric
lock — and that claim is preserved here: this module renders references, it does
not control the external video provider.

Everything is CPU-only, dependency-free beyond NumPy, and takes an explicit
sun direction so lighting is reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mesh import Mesh

FAR_DEPTH = 1e4


@dataclass(frozen=True)
class Camera:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_deg: float = 55.0
    near: float = 0.05
    far: float = 500.0

    @property
    def forward(self) -> np.ndarray:
        f = np.array(self.target, dtype=np.float64) - np.array(self.position, dtype=np.float64)
        n = np.linalg.norm(f)
        return f / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = self.forward
        up_hint = np.array(self.up, dtype=np.float64)
        right = np.cross(forward, up_hint)
        if np.linalg.norm(right) < 1e-6:  # looking straight up/down
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        return right, true_up, forward

    def as_dict(self) -> dict:
        return {
            "position": [round(v, 3) for v in self.position],
            "target": [round(v, 3) for v in self.target],
            "fov_deg": self.fov_deg,
        }


@dataclass
class RenderResult:
    color: np.ndarray      # (H, W, 3) uint8
    depth: np.ndarray      # (H, W) float32, metres, 0 where no geometry
    object_id: np.ndarray  # (H, W) int32, -1 where no geometry
    material_id: np.ndarray

    def edge_map(self, depth_threshold: float = 0.12) -> np.ndarray:
        """Binary structural edges from depth and object discontinuities."""
        depth = self.depth
        obj = self.object_id
        valid = depth > 0
        edges = np.zeros(depth.shape, dtype=bool)
        if depth.shape[0] < 2 or depth.shape[1] < 2:
            return edges
        dz_x = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
        dz_y = np.abs(np.diff(depth, axis=0, prepend=depth[:1, :]))
        near_mask = np.minimum(depth, np.roll(depth, 1, axis=1))
        rel = np.maximum(dz_x, dz_y) / np.maximum(near_mask, 0.1)
        edges |= rel > depth_threshold
        obj_change = np.zeros(depth.shape, dtype=bool)
        obj_change[:, 1:] = obj[:, 1:] != obj[:, :-1]
        obj_change[1:, :] |= obj[1:, :] != obj[:-1, :]
        # only count object changes that also involve real geometry
        edges |= obj_change & valid & (depth < FAR_DEPTH * 0.5)
        return edges

    def segmentation(self) -> np.ndarray:
        """Deterministic colour-coded object ids for QA comparison."""
        ids = self.object_id
        out = np.zeros((*ids.shape, 3), dtype=np.uint8)
        palette = np.array([
            [230, 60, 60], [60, 200, 120], [70, 130, 240], [240, 200, 60],
            [200, 90, 230], [60, 220, 220], [250, 140, 60], [150, 220, 90],
        ], dtype=np.uint8)
        positive = ids >= 0
        out[positive] = palette[ids[positive] % len(palette)]
        return out


def render(
    mesh: Mesh,
    camera: Camera,
    width: int,
    height: int,
    *,
    sun_direction: tuple[float, float, float] = (-0.45, 0.55, 0.70),
    ambient: float = 0.50,
    sky_top: tuple[float, float, float] = (0.42, 0.55, 0.72),
    sky_horizon: tuple[float, float, float] = (0.80, 0.84, 0.88),
    exposure: float = 1.0,
    fog_density: float | None = None,
) -> RenderResult:
    width, height = int(max(16, width)), int(max(16, height))
    color_buf = _sky_gradient(width, height, sky_top, sky_horizon)
    depth_buf = np.full((height, width), 0.0, dtype=np.float32)
    object_buf = np.full((height, width), -1, dtype=np.int32)
    material_buf = np.full((height, width), -1, dtype=np.int32)

    if len(mesh.faces) == 0:
        return RenderResult(
            (np.clip(color_buf, 0, 1) * 255).astype(np.uint8),
            depth_buf, object_buf, material_buf,
        )

    if fog_density is None:
        # Scale fog to the scene so a 10 m house and a 60 m plot both read clearly.
        extent = float(np.linalg.norm(np.subtract(*[_hb(mesh) for _ in (0,)]))) if False else 0.0
        lo, hi = (mesh.bounds if len(mesh.vertices) else (np.zeros(3), np.zeros(3)))
        diag = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))) or 40.0
        fog_density = 2.2 / max(25.0, diag)

    right, true_up, forward = camera.basis()
    eye = np.array(camera.position, dtype=np.float64)
    tris = mesh.vertices[mesh.faces]                        # (M, 3, 3)
    view = (tris - eye) @ np.stack([right, true_up, forward], axis=0).T
    # view space: x right, y up, z along forward (positive = in front)

    z = view[:, :, 2]
    in_front = np.all(z > camera.near, axis=1) & np.all(z < camera.far, axis=1)

    focal = 1.0 / math.tan(math.radians(camera.fov_deg) / 2.0)
    aspect = width / height
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc_x = (view[:, :, 0] / z) * focal / aspect
        ndc_y = (view[:, :, 1] / z) * focal
    # Vertices at or behind the near plane produce NaN in the projection; the
    # masks below discard them, so silence the (expected) NumPy warnings.
    with np.errstate(invalid="ignore", divide="ignore"):
        sx = (ndc_x * 0.5 + 0.5) * width
        sy = (1.0 - (ndc_y * 0.5 + 0.5)) * height

    # Cheap whole-triangle rejects.
    pad = 2.0
    tri_offscreen = (
        (np.nanmax(sx, axis=1) < -pad) | (np.nanmin(sx, axis=1) > width + pad)
        | (np.nanmax(sy, axis=1) < -pad) | (np.nanmin(sy, axis=1) > height + pad)
    )
    with np.errstate(invalid="ignore"):
        area = (
            (sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0])
            - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0])
        )
    degenerate = np.abs(area) < 1e-9
    candidates = np.nonzero(in_front & ~tri_offscreen & ~degenerate)[0]

    if len(candidates) == 0:
        return RenderResult(
            (np.clip(color_buf, 0, 1) * 255).astype(np.uint8),
            depth_buf, object_buf, material_buf,
        )

    sun = np.array(sun_direction, dtype=np.float64)
    sun_norm = np.linalg.norm(sun)
    sun = sun / sun_norm if sun_norm else sun
    # world-space face normals for shading
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(e1, e2)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.where(nlen == 0, 1.0, nlen))
    # Two-sided shading: face winding is not reliable for extruded slabs and
    # roof planes, so orient each normal towards the camera before lighting it.
    centroids = tris.mean(axis=1)
    facing = np.einsum("ij,ij->i", normals, centroids - eye) > 0
    normals[facing] *= -1.0

    mat_index = mesh.face_material
    colours = np.array([m.color for m in mesh.materials], dtype=np.float64) if mesh.materials else np.zeros((1, 3))
    roughness = np.array([m.roughness for m in mesh.materials], dtype=np.float64) if mesh.materials else np.ones(1)
    kinds = [m.kind for m in mesh.materials] or ["generic"]

    # Painter's order is unnecessary with a z-buffer, but drawing front-to-back
    # reduces overdraw. Sort by centroid depth.
    centroid_z = z[candidates].mean(axis=1)
    order = candidates[np.argsort(centroid_z)]

    # Triangles smaller than a couple of pixels are splatted at their centre:
    # a full per-pixel scan for them costs more than it can possibly show.
    with np.errstate(invalid="ignore"):
        widths = np.nanmax(sx, axis=1) - np.nanmin(sx, axis=1)
        heights = np.nanmax(sy, axis=1) - np.nanmin(sy, axis=1)
    tiny = (widths < 2.2) | (heights < 2.2)

    for ti in order:
        if tiny[ti]:
            _splat_triangle(
                ti, sx[ti], sy[ti], z[ti], normals[ti], mat_index[ti],
                colours, roughness, kinds[mat_index[ti] % len(kinds)], sun, ambient,
                exposure, camera, color_buf, depth_buf, object_buf, material_buf,
                mesh.face_object[ti], width, height, fog_density,
            )
            continue
        _raster_triangle(
            ti, sx[ti], sy[ti], z[ti], normals[ti], mat_index[ti],
            colours, roughness, kinds[mat_index[ti] % len(kinds)], sun, ambient, exposure,
            camera, color_buf, depth_buf, object_buf, material_buf,
            mesh.face_object[ti], width, height, fog_density,
        )

    return RenderResult(
        (np.clip(color_buf, 0, 1) * 255 + 0.5).astype(np.uint8),
        depth_buf, object_buf, material_buf,
    )


def _raster_triangle(
    ti, sx, sy, zv, normal, mat_i, colours, roughness, kind,
    sun, ambient, exposure, camera,
    color_buf, depth_buf, object_buf, material_buf,
    object_index, width, height, fog_density,
) -> None:
    x0 = max(int(math.floor(min(sx[0], sx[1], sx[2]))), 0)
    x1 = min(int(math.ceil(max(sx[0], sx[1], sx[2]))) + 1, width)
    y0 = max(int(math.floor(min(sy[0], sy[1], sy[2]))), 0)
    y1 = min(int(math.ceil(max(sy[0], sy[1], sy[2]))) + 1, height)
    if x1 <= x0 or y1 <= y0:
        return

    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    px, py = np.meshgrid(xs, ys)

    ax, ay = sx[0], sy[0]
    bx, by = sx[1], sy[1]
    cx, cy = sx[2], sy[2]

    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < 1e-12:
        return
    w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
    w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
    if not inside.any():
        return

    # Perspective-correct depth via linear interpolation of 1/z.
    inv_z = w0 * (1.0 / zv[0]) + w1 * (1.0 / zv[1]) + w2 * (1.0 / zv[2])
    depth = np.where(inv_z > 1e-9, 1.0 / np.maximum(inv_z, 1e-9), FAR_DEPTH)

    depth_patch = depth_buf[y0:y1, x0:x1]
    closer = inside & ((depth_patch <= 0.0) | (depth < depth_patch))
    if not closer.any():
        return

    base = colours[mat_i % len(colours)]
    n = normal
    # Two-sided shading: flip the normal toward the viewer.
    to_view = np.array(camera.position) - np.array(camera.target)
    if np.dot(n, -to_view) < 0:
        n = -n
    lambert = max(0.0, float(np.dot(n, sun)))
    hemi = 0.5 + 0.5 * float(n[2])
    rough = float(roughness[mat_i % len(roughness)])
    diffuse = ambient * (0.55 + 0.45 * hemi) + (1.0 - ambient) * lambert
    shade = base * (diffuse * exposure)
    if kind == "glass":
        # Fake a sky reflection so glazing reads as glazing.
        reflect = 0.55 * (0.5 + 0.5 * hemi)
        shade = shade * (1.0 - reflect) + np.array([0.62, 0.74, 0.86]) * reflect
    elif kind == "ground":
        shade = shade * 0.92
    # specular highlight for smooth surfaces
    spec = (1.0 - rough) ** 4 * max(0.0, lambert) ** 8 * 0.35
    shade = np.clip(shade + spec, 0.0, 1.5)

    # Exponential distance fog keeps depth readable in references.
    fog = 1.0 - np.exp(-depth * fog_density)
    fog_rgb = np.array([0.78, 0.82, 0.86])
    patch = shade[None, None, :] * (1.0 - fog[..., None]) + fog_rgb * fog[..., None]

    colour_patch = color_buf[y0:y1, x0:x1]
    colour_patch[closer] = patch[closer]
    depth_patch[closer] = depth[closer]
    object_buf[y0:y1, x0:x1][closer] = object_index
    material_buf[y0:y1, x0:x1][closer] = mat_i


def _splat_triangle(
    ti, sx, sy, zv, normal, mat_i, colours, roughness, kind,
    sun, ambient, exposure, camera, color_buf, depth_buf, object_buf, material_buf,
    object_index, width, height, fog_density,
) -> None:
    """Single-sample rasterisation for sub-pixel triangles."""
    px = int(round(float(np.mean(sx))))
    py = int(round(float(np.mean(sy))))
    if px < 0 or py < 0 or px >= width or py >= height:
        return
    depth = float(np.mean(zv))
    if depth <= 0:
        return
    current = depth_buf[py, px]
    if current > 0 and depth >= current:
        return
    base = colours[mat_i % len(colours)]
    n = normal
    to_view = np.array(camera.position) - np.array(camera.target)
    if np.dot(n, -to_view) < 0:
        n = -n
    lambert = max(0.0, float(np.dot(n, sun)))
    hemi = 0.5 + 0.5 * float(n[2])
    diffuse = ambient * (0.55 + 0.45 * hemi) + (1.0 - ambient) * lambert
    shade = np.clip(base * (diffuse * exposure), 0.0, 1.5)
    fog = 1.0 - math.exp(-depth * fog_density)
    color_buf[py, px] = shade * (1.0 - fog) + np.array([0.78, 0.82, 0.86]) * fog
    depth_buf[py, px] = depth
    object_buf[py, px] = object_index
    material_buf[py, px] = mat_i


def _sky_gradient(width: int, height: int, top, horizon) -> np.ndarray:
    t = np.linspace(0.0, 1.0, height)[:, None, None]
    top = np.array(top, dtype=np.float64)
    horizon = np.array(horizon, dtype=np.float64)
    gradient = top * (1.0 - t) + horizon * t
    return np.repeat(gradient, width, axis=1).astype(np.float64)


def depth_to_png_bytes(result: RenderResult, invert: bool = False) -> bytes:
    """Metric depth preview: near = bright, far = dark (or inverted)."""
    depth = result.depth
    valid = depth > 0
    if not valid.any():
        norm = np.zeros_like(depth)
    else:
        near = np.percentile(depth[valid], 2)
        far = np.percentile(depth[valid], 98)
        span = max(far - near, 1e-4)
        norm = np.clip((depth - near) / span, 0.0, 1.0)
        norm[~valid] = 0.0
    if not invert:
        norm = 1.0 - norm
    norm[~valid] = 0.0
    return _png_from_array((norm * 255).astype(np.uint8))


def edge_to_png_bytes(result: RenderResult) -> bytes:
    edges = result.edge_map()
    return _png_from_array((edges * 255).astype(np.uint8))


def colour_to_png_bytes(result: RenderResult) -> bytes:
    return _png_from_array(result.color)


def segmentation_to_png_bytes(result: RenderResult) -> bytes:
    return _png_from_array(result.segmentation())


def _png_from_array(array: np.ndarray) -> bytes:
    from io import BytesIO

    from PIL import Image

    if array.ndim == 2:
        image = Image.fromarray(array, mode="L")
    else:
        image = Image.fromarray(array, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def camera_from_plan(position, target, fov_deg=55.0, up=(0.0, 0.0, 1.0)) -> Camera:
    return Camera(
        position=tuple(float(v) for v in position),
        target=tuple(float(v) for v in target),
        fov_deg=float(fov_deg),
        up=tuple(float(v) for v in up),
    )
