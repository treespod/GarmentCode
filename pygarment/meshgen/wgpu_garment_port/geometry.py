"""OBJ and cloth-model assembly helpers for the GarmentCode simulation path."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class BodyContactQuery:
    """Reusable CPU mesh query for per-frame static-body contact anchors."""

    def __init__(
        self,
        body_vertices: np.ndarray,
        body_faces: np.ndarray,
        *,
        particle_radius: float = 0.1,
        body_thickness: float = 0.25,
        margin: float = 0.2,
        enable_reference_drag: bool = False,
        face_filter_masks: np.ndarray | None = None,
        particle_filter_ids: np.ndarray | None = None,
        body_face_labels: list[str] | None = None,
        particle_reference_labels: list[str] | None = None,
        reference_drag_pairs: set[tuple[str, str]] | None = None,
    ) -> None:
        import trimesh

        vertices = np.asarray(body_vertices, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(body_faces, dtype=np.int32).reshape(-1, 3)
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if not self.mesh.is_watertight:
            raise ValueError("Body mesh must be watertight for signed contact queries")
        self.proximity = trimesh.proximity.ProximityQuery(self.mesh)
        self.particle_radius = float(particle_radius)
        self.body_thickness = float(body_thickness)
        self.margin = float(margin)
        self.enable_reference_drag = bool(enable_reference_drag)
        self.face_filter_masks = (
            np.empty((0, len(faces)), dtype=bool)
            if face_filter_masks is None
            else np.asarray(face_filter_masks, dtype=bool).reshape(-1, len(faces))
        )
        self.particle_filter_ids = (
            None
            if particle_filter_ids is None
            else np.asarray(particle_filter_ids, dtype=np.int32).reshape(-1)
        )
        self.body_face_labels = body_face_labels
        self.particle_reference_labels = particle_reference_labels
        self.reference_drag_pairs = reference_drag_pairs or set()
        self.filtered_proximity: dict[int, tuple[Any, np.ndarray]] = {}
        for filter_id, excluded_faces in enumerate(self.face_filter_masks):
            allowed_face_ids = np.flatnonzero(~excluded_faces)
            if len(allowed_face_ids):
                filtered_mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces[allowed_face_ids],
                    process=False,
                )
                self.filtered_proximity[filter_id] = (
                    trimesh.proximity.ProximityQuery(filtered_mesh),
                    allowed_face_ids,
                )

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Return [particle, 3, vec4] contact point/normal/closest-point records."""
        p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if self.particle_filter_ids is not None and len(self.particle_filter_ids) != len(p):
            raise ValueError("particle_filter_ids must have one entry per position")
        closest = np.empty_like(p)
        distance = np.empty(len(p), dtype=np.float64)
        face_id = np.empty(len(p), dtype=np.int64)
        filter_ids = (
            np.full(len(p), -1, dtype=np.int32)
            if self.particle_filter_ids is None
            else self.particle_filter_ids
        )
        for filter_id in np.unique(filter_ids):
            particle_ids = np.flatnonzero(filter_ids == filter_id)
            if filter_id < 0:
                query = self.proximity
                original_face_ids = None
            else:
                if int(filter_id) not in self.filtered_proximity:
                    raise ValueError(
                        f"body collision filter {filter_id} excludes every body face"
                    )
                query, original_face_ids = self.filtered_proximity[int(filter_id)]
            group_closest, group_distance, group_face_ids = query.on_surface(p[particle_ids])
            closest[particle_ids] = group_closest
            distance[particle_ids] = group_distance
            if original_face_ids is None:
                face_id[particle_ids] = group_face_ids
            else:
                face_id[particle_ids] = original_face_ids[group_face_ids]
        # The body mesh is watertight and winding-consistent. Its closest-face
        # normal therefore gives the local signed side without running a costly
        # all-rays point-in-mesh query on every frame.
        face_normals = self.mesh.face_normals[face_id]
        delta = p - closest
        side_dot = np.einsum("ij,ij->i", delta, face_normals)
        sign = np.where(side_dot < 0.0, -1.0, 1.0)
        normals = np.zeros_like(delta)
        nonzero = distance > 1.0e-10
        normals[nonzero] = delta[nonzero] / distance[nonzero, None] * sign[nonzero, None]
        if np.any(~nonzero):
            normals[~nonzero] = face_normals[~nonzero]
        norm_len = np.linalg.norm(normals, axis=1)
        good = norm_len > 1.0e-10
        normals[good] /= norm_len[good, None]
        normals[~good] = np.array([0.0, 1.0, 0.0])

        effective_d = distance * sign - self.body_thickness
        contact_active = (effective_d > -self.margin) & (
            effective_d < self.margin + self.particle_radius
        )
        drag_active = (
            effective_d < -self.margin
            if self.enable_reference_drag
            else np.zeros(len(p), dtype=bool)
        )
        if drag_active.any() and self.reference_drag_pairs:
            if (
                self.body_face_labels is None
                or self.particle_reference_labels is None
                or len(self.body_face_labels) != len(self.mesh.faces)
                or len(self.particle_reference_labels) != len(p)
            ):
                raise ValueError(
                    "reference-drag labels must cover body faces and cloth particles"
                )
            drag_active &= np.fromiter(
                (
                    (
                        self.particle_reference_labels[particle_id],
                        (
                            "legs"
                            if self.particle_reference_labels[particle_id] == "legs"
                            and self.body_face_labels[int(face_id[particle_id])]
                            in {"left_leg", "right_leg"}
                            else self.body_face_labels[int(face_id[particle_id])]
                        ),
                    )
                    in self.reference_drag_pairs
                    for particle_id in range(len(p))
                ),
                dtype=bool,
                count=len(p),
            )

        data = np.zeros((len(p), 3, 4), dtype=np.float32)
        # Warp computes body_pos = x - n * (signed_distance - thickness),
        # which equals closest + n * thickness for an exact closest query.
        data[:, 0, :3] = closest + normals * self.body_thickness
        data[:, 0, 3] = contact_active.astype(np.float32)
        data[:, 1, :3] = normals
        data[:, 1, 3] = drag_active.astype(np.float32)
        data[:, 2, :3] = closest
        return data


def build_body_contact_data(
    positions: np.ndarray,
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    *,
    particle_radius: float = 0.1,
    body_thickness: float = 0.25,
    margin: float = 0.2,
    enable_reference_drag: bool = False,
) -> np.ndarray:
    """One-shot helper; use :class:`BodyContactQuery` for repeated frames."""
    query = BodyContactQuery(
        body_vertices,
        body_faces,
        particle_radius=particle_radius,
        body_thickness=body_thickness,
        margin=margin,
        enable_reference_drag=enable_reference_drag,
    )
    return query(positions)


@dataclass
class ClothModelData:
    positions: np.ndarray
    faces: np.ndarray
    masses: np.ndarray
    inverse_masses: np.ndarray
    spring_indices: np.ndarray
    spring_rest_lengths: np.ndarray
    bend_indices: np.ndarray
    segmentation: list[str]
    stitch_vertices: np.ndarray
    triangle_areas: np.ndarray


def read_obj_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vertex positions and polygon faces from an OBJ.

    The GarmentCode test serializes a triangle mesh. Polygon faces are accepted
    here as well and fan-triangulated so the helper remains useful for edited
    inputs.
    """
    vertices: list[list[float]] = []
    triangles: list[tuple[int, int, int]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed OBJ vertex in {path}: {line.rstrip()}")
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                refs = line.split()[1:]
                if len(refs) < 3:
                    continue
                face: list[int] = []
                for ref in refs:
                    raw = int(ref.split("/", 1)[0])
                    face.append(raw - 1 if raw > 0 else len(vertices) + raw)
                for j in range(1, len(face) - 1):
                    triangles.append((face[0], face[j], face[j + 1]))
    if not vertices:
        raise ValueError(f"No vertices found in OBJ: {path}")
    if not triangles:
        raise ValueError(f"No faces found in OBJ: {path}")
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(triangles, dtype=np.int32)
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError(f"OBJ face index out of range in {path}")
    return v, f


def write_obj_with_positions(
    source_obj: str | Path,
    output_obj: str | Path,
    vertices: np.ndarray,
) -> None:
    """Copy an OBJ while replacing only its ``v`` records."""
    source_obj = Path(source_obj)
    output_obj = Path(output_obj)
    vertices = np.asarray(vertices).reshape(-1, 3)
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    out_index = 0
    with source_obj.open("r", encoding="utf-8", errors="replace") as src, output_obj.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            if line.startswith("v "):
                if out_index >= len(vertices):
                    raise ValueError("OBJ has more vertex records than the supplied simulation state")
                x, y, z = vertices[out_index]
                dst.write(f"v {float(x):.9g} {float(y):.9g} {float(z):.9g}\n")
                out_index += 1
            else:
                dst.write(line)
    if out_index != len(vertices):
        raise ValueError(
            f"OBJ vertex count ({out_index}) does not match simulated count ({len(vertices)})"
        )


def read_vertex_segmentation(path: str | Path, expected_vertices: int) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for row in f:
            labels.append(row.rstrip("\r\n").split(",", 1)[0])
    if labels and len(labels) != expected_vertices:
        raise ValueError(
            f"Segmentation has {len(labels)} rows for {expected_vertices} mesh vertices"
        )
    stitch = np.asarray([i for i, label in enumerate(labels) if label.startswith("stitch")], dtype=np.int32)
    return labels, stitch


def _read_original_lengths(path: str | Path | None) -> dict[tuple[int, int], float] | None:
    if path is None or not Path(path).exists():
        return None
    with Path(path).open("rb") as f:
        raw = pickle.load(f)
    result: dict[tuple[int, int], float] = {}
    for key, value in raw.items():
        i, j = (int(key[0]), int(key[1]))
        result[(min(i, j), max(i, j))] = float(value)
    return result


@dataclass
class _EdgeInfo:
    v0: int
    v1: int
    o0: int
    o1: int
    f0: int
    f1: int


def mesh_interior_edges(faces: np.ndarray) -> np.ndarray:
    """Return the 4-vertex interior-edge records used by Warp's MeshAdjacency.

    Each row is (opposite vertex in first face, opposite vertex in second face,
    first directed edge vertex, second directed edge vertex). The insertion and
    orientation rules intentionally mirror ``warp.utils.MeshAdjacency``.
    """
    edges: dict[tuple[int, int], _EdgeInfo] = {}
    for face_id, tri in enumerate(np.asarray(faces, dtype=np.int64)):
        i, j, k = (int(x) for x in tri)
        directed = ((i, j, k), (j, k, i), (k, i, j))
        for v0, v1, opposite in directed:
            key = (min(v0, v1), max(v0, v1))
            if key not in edges:
                edges[key] = _EdgeInfo(v0, v1, opposite, -1, face_id, -1)
            else:
                info = edges[key]
                if info.f1 == -1:
                    info.o1 = opposite
                    info.f1 = face_id
    return np.asarray(
        [(e.o0, e.o1, e.v0, e.v1) for e in edges.values() if e.f0 != -1 and e.f1 != -1],
        dtype=np.int32,
    ).reshape(-1, 4)


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a)))


def _stitch_area_from_original_lengths(
    tri: tuple[int, int, int],
    original_lengths: dict[tuple[int, int], float],
) -> float:
    ind1, ind2, ind3 = sorted(int(x) for x in tri)
    try:
        # These three lookups and their ordering match add_triangles_stitching
        # in the source Warp fork (the first value is passed as side a).
        a = original_lengths[(ind1, ind2)]
        b = original_lengths[(ind2, ind3)]
        c = original_lengths[(ind1, ind3)]
    except KeyError:
        return 0.0
    if min(a, b, c) <= 0.0 or a + b <= c or a + c <= b or b + c <= a:
        return 0.0
    cosine = np.clip((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0)
    sine = math_sqrt_nonnegative(1.0 - cosine * cosine)
    return 0.5 * b * c * sine


def math_sqrt_nonnegative(x: float) -> float:
    return float(np.sqrt(max(float(x), 0.0)))


def build_cloth_model_data(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    segmentation_path: str | Path | None = None,
    original_lengths_path: str | Path | None = None,
    fabric_density: float = 1.0,
    particle_mass_seed: float = 0.1,
) -> ClothModelData:
    """Build masses, springs and bending constraints for the tested garment.

    The particle mass seed preserves the source fork's current call
    ``add_particle(p, vel, radius)`` (the third positional argument is mass).
    Triangle-area mass contributions and original-seam lengths are added in the
    same way as ``ModelBuilder.add_cloth_mesh_sewing_spring``.
    """
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if segmentation_path is not None and Path(segmentation_path).exists():
        labels, stitch_vertices = read_vertex_segmentation(segmentation_path, len(v))
    else:
        labels = [""] * len(v)
        stitch_vertices = np.empty(0, dtype=np.int32)
    original_lengths = _read_original_lengths(original_lengths_path)
    stitch_set = set(int(i) for i in stitch_vertices)

    areas = np.zeros(len(f), dtype=np.float64)
    for face_id, tri in enumerate(f):
        i, j, k = (int(x) for x in tri)
        if stitch_set and (i in stitch_set or j in stitch_set or k in stitch_set):
            area = _stitch_area_from_original_lengths((i, j, k), original_lengths or {})
            if area <= 0.0:
                area = _triangle_area(v[i], v[j], v[k])
        else:
            area = _triangle_area(v[i], v[j], v[k])
        areas[face_id] = area

    # Match Warp's initialization: add_cloth_mesh_sewing_spring passes radius
    # positionally as the particle's initial mass, then accumulates triangle mass.
    masses = np.full(len(v), float(particle_mass_seed), dtype=np.float64)
    for tri, area in zip(f, areas):
        share = float(fabric_density) * float(area) / 3.0
        masses[tri] += share
    inverse_masses = np.divide(
        1.0, masses, out=np.zeros_like(masses), where=masses > 0.0
    )

    interior = mesh_interior_edges(f)
    spring_set: set[tuple[int, int]] = set()
    for o0, o1, v0, v1 in interior:
        i, j, k, l = int(o0), int(o1), int(v0), int(v1)
        for a, b in ((i, k), (i, l), (k, l), (j, k), (j, l)):
            spring_set.add((min(a, b), max(a, b)))
    spring_indices = np.asarray(sorted(spring_set), dtype=np.int32).reshape(-1, 2)

    rest_lengths = np.zeros(len(spring_indices), dtype=np.float64)
    for s, (i, j) in enumerate(spring_indices):
        pair = (int(i), int(j))
        if original_lengths is not None and pair in original_lengths:
            rest_lengths[s] = original_lengths[pair]
        else:
            rest_lengths[s] = float(np.linalg.norm(v[i] - v[j]))

    return ClothModelData(
        positions=v.copy(),
        faces=f.copy(),
        masses=masses.astype(np.float32),
        inverse_masses=inverse_masses.astype(np.float32),
        spring_indices=spring_indices,
        spring_rest_lengths=rest_lengths.astype(np.float32),
        bend_indices=interior,
        segmentation=labels,
        stitch_vertices=stitch_vertices,
        triangle_areas=areas.astype(np.float32),
    )
