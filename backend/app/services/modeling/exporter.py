"""Minimal glTF 2.0 binary (GLB) writer.

The locked model must be a real, inspectable artifact that a client (Blender,
three.js, an architect) can open — not an internal pickle. This writes a valid
GLB with one primitive per material, flat-shaded normals, and the model
metadata embedded in ``extras``. No third-party glTF dependency is needed.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

from .mesh import Mesh

_COMPONENT_FLOAT = 5126
_COMPONENT_UINT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_MODE_TRIANGLES = 4
_TARGET = _ARRAY_BUFFER


def _pad(data: bytes, multiple: int = 4) -> bytes:
    remainder = len(data) % multiple
    return data if remainder == 0 else data + b"\x00" * (multiple - remainder)


def mesh_to_glb(mesh: Mesh, metadata: dict[str, Any] | None = None) -> bytes:
    if len(mesh.faces) == 0:
        raise ValueError("cannot export an empty mesh")

    triangles = mesh.vertices[mesh.faces]                        # (M, 3, 3)
    # Group faces by material with a stable sort so each material becomes one
    # contiguous primitive, then flat-shade by expanding to unique vertices.
    order = np.argsort(mesh.face_material, kind="stable")
    triangles = triangles[order]
    face_material = mesh.face_material[order]

    edge1 = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(edge1, axis=1, keepdims=True)
    unit = np.divide(edge1, np.where(lengths == 0.0, 1.0, lengths))

    binary = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    primitives: list[dict] = []

    def add_view(payload: bytes, target: int | None = None) -> int:
        binary.extend(_pad(payload, 4))
        view: dict = {
            "buffer": 0,
            "byteOffset": len(binary) - len(_pad(payload, 4)),
            "byteLength": len(payload),
        }
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    materials_present = sorted({int(m) for m in face_material})
    for mat_index in materials_present:
        mask = face_material == mat_index
        count = int(mask.sum())
        if count == 0:
            continue
        face_positions = triangles[mask].reshape(-1, 3).astype(np.float32)
        face_normals = np.repeat(unit[mask], 3, axis=0).astype(np.float32)
        indices = np.arange(count * 3, dtype=np.uint32)

        pos_view = add_view(face_positions.tobytes(), _TARGET)
        nrm_view = add_view(face_normals.tobytes(), _TARGET)
        idx_view = add_view(indices.tobytes(), _ELEMENT_ARRAY_BUFFER)

        base = len(accessors)
        accessors.extend([
            {
                "bufferView": pos_view,
                "componentType": _COMPONENT_FLOAT,
                "count": int(len(face_positions)),
                "type": "VEC3",
                "min": face_positions.min(axis=0).tolist(),
                "max": face_positions.max(axis=0).tolist(),
            },
            {
                "bufferView": nrm_view,
                "componentType": _COMPONENT_FLOAT,
                "count": int(len(face_normals)),
                "type": "VEC3",
            },
            {
                "bufferView": idx_view,
                "componentType": _COMPONENT_UINT,
                "count": int(len(indices)),
                "type": "SCALAR",
            },
        ])
        primitives.append({
            "attributes": {"POSITION": base, "NORMAL": base + 1},
            "indices": base + 2,
            "material": mat_index,
            "mode": _MODE_TRIANGLES,
        })

    gltf_materials = []
    for material in mesh.materials:
        entry: dict[str, Any] = {
            "name": material.name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [*material.color, 1.0],
                "metallicFactor": float(material.metallic),
                "roughnessFactor": float(material.roughness),
            },
        }
        if material.kind == "glass":
            entry["alphaMode"] = "BLEND"
            entry["pbrMetallicRoughness"]["baseColorFactor"][3] = 0.45
        gltf_materials.append(entry)

    gltf: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": (metadata or {}).get("generator", "BuildVisual"),
        },
        "scene": 0,
        "scenes": [{"nodes": [0], "name": "House"}],
        "nodes": [{"mesh": 0, "name": (metadata or {}).get("name", "Locked model")}],
        "meshes": [{
            "name": "House",
            "primitives": primitives,
            "extras": {
                "objects": [
                    {"id": o.id, "name": o.name, "kind": o.kind, "level": o.level,
                     "room_id": o.room_id, "material": o.material,
                     "provenance": o.provenance}
                    for o in mesh.objects
                ],
            },
        }],
        "materials": gltf_materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(_pad(bytes(binary)))}],
        "extras": metadata or {},
    }

    json_chunk = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), 4)
    bin_chunk = _pad(bytes(binary), 4)

    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    header = struct.pack("<III", 0x46546C67, 2, total)
    json_header = struct.pack("<II", len(json_chunk), 0x4E4F534A)
    bin_header = struct.pack("<II", len(bin_chunk), 0x004E4942)
    return header + json_header + json_chunk + bin_header + bin_chunk
