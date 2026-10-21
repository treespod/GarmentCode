"""CPU broad phase for the XPBD point-triangle and edge-edge contacts.

The same candidate/contact payloads are consumed by WGSL kernels. Contact
resolution remains GPU-side; this module provides the mesh-query broad phase
that WebGPU does not expose as a built-in primitive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class SelfContactData:
    point_tri_indices: np.ndarray  # [contact, point, tri-v0, tri-v1, tri-v2]
    point_tri_sidedness: np.ndarray  # bool [contact]
    edge_indices: np.ndarray  # [contact, p-a, p-b, edge0, edge1]
    edge_normals: np.ndarray  # frame-start separating direction [contact, 3]
    intersecting_particles: np.ndarray  # bool [particle], global self-intersection filter


def closest_point_triangle_barycentric(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Ericson's closest-point-on-triangle routine, returning barycentrics."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return np.array([1.0, 0.0, 0.0])

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return np.array([0.0, 1.0, 0.0])

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return np.array([1.0 - v, v, 0.0])

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return np.array([0.0, 0.0, 1.0])

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return np.array([1.0 - w, 0.0, w])

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return np.array([0.0, 1.0 - w, w])

    denom = va + vb + vc
    if abs(denom) < 1.0e-20:
        # Degenerate triangle: choose the nearest of its three segments.
        choices = []
        for x, y, bary_x, bary_y in (
            (a, b, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            (b, c, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (c, a, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ):
            e = y - x
            t = np.clip(np.dot(p - x, e) / max(float(np.dot(e, e)), 1.0e-20), 0.0, 1.0)
            q = x + t * e
            bary = (1.0 - t) * np.asarray(bary_x) + t * np.asarray(bary_y)
            choices.append((float(np.dot(p - q, p - q)), bary))
        return min(choices, key=lambda item: item[0])[1]
    inv = 1.0 / denom
    v = vb * inv
    w = vc * inv
    return np.array([1.0 - v - w, v, w])


def closest_segment_parameters(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> tuple[float, float, float]:
    """Closest parameters/distance for two finite 3-D line segments."""
    d1 = a1 - a0
    d2 = b1 - b0
    r = a0 - b0
    aa = float(np.dot(d1, d1))
    ee = float(np.dot(d2, d2))
    ff = float(np.dot(d2, r))
    eps = 1.0e-12
    if aa <= eps and ee <= eps:
        s = t = 0.0
    elif aa <= eps:
        s = 0.0
        t = np.clip(ff / ee, 0.0, 1.0)
    else:
        cc = float(np.dot(d1, r))
        if ee <= eps:
            t = 0.0
            s = np.clip(-cc / aa, 0.0, 1.0)
        else:
            bb = float(np.dot(d1, d2))
            denom = aa * ee - bb * bb
            s = np.clip((bb * ff - cc * ee) / denom, 0.0, 1.0) if abs(denom) > eps else 0.0
            t = (bb * s + ff) / ee
            if t < 0.0:
                t = 0.0
                s = np.clip(-cc / aa, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((bb - cc) / aa, 0.0, 1.0)
    pa = a0 + s * d1
    pb = b0 + t * d2
    return float(s), float(t), float(np.linalg.norm(pb - pa))


def _global_intersection_filter(
    positions: np.ndarray, faces: np.ndarray, spring_indices: np.ndarray, mesh: trimesh.Trimesh
) -> np.ndarray:
    """Mirror Warp's segment-vs-mesh test used by global collision filtering."""
    p = positions
    springs = np.asarray(spring_indices, dtype=np.int32).reshape(-1, 2)
    flags = np.zeros(len(p), dtype=bool)
    if not len(springs):
        return flags
    starts = p[springs[:, 0]]
    vectors = p[springs[:, 1]] - starts
    lengths = np.linalg.norm(vectors, axis=1)
    valid_ids = np.flatnonzero(lengths > 1.0e-10)
    ray_indexer = mesh.ray
    # Keep batches small: a folded garment can produce a large number of
    # ray/triangle candidates even though only a handful are true crossings.
    chunk_size = 128
    for begin in range(0, len(valid_ids), chunk_size):
        edge_ids = valid_ids[begin : begin + chunk_size]
        directions = vectors[edge_ids] / lengths[edge_ids, None]
        face_ids, ray_ids, locations = ray_indexer.intersects_id(
            ray_origins=starts[edge_ids],
            ray_directions=directions,
            return_locations=True,
            multiple_hits=True,
        )
        for ray_id, face_id, location in zip(ray_ids, face_ids, locations):
            edge_id = int(edge_ids[int(ray_id)])
            i, j = (int(x) for x in springs[edge_id])
            tri = faces[int(face_id)]
            # mesh_query_edge excludes triangles sharing either segment endpoint.
            if i in tri or j in tri:
                continue
            distance = float(np.dot(location - starts[edge_id], directions[int(ray_id)]))
            if 0.0 <= distance < lengths[edge_id]:
                flags[i] = True
                flags[j] = True
    return flags


def build_self_contact_data(
    positions: np.ndarray,
    velocities: np.ndarray,
    faces: np.ndarray,
    spring_indices: np.ndarray,
    inverse_masses: np.ndarray,
    *,
    particle_radius: float = 0.1,
    dt: float = 1.0 / 60.0,
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    global_collision_filter: bool = False,
    max_point_tri_contacts: int | None = None,
    max_edge_edge_contacts: int | None = None,
) -> SelfContactData:
    """Build candidate contacts for one GarmentCode frame.

    Broad-phase bounds, prediction thresholds and optional global self-
    intersection filtering follow the source Warp fork. GarmentCode's
    label-based panel/body filters and cloth-reference drag policy are not
    applied here; the adapter reports those separately.
    """
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    v = np.asarray(velocities, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    springs = np.asarray(spring_indices, dtype=np.int32).reshape(-1, 2)
    invm = np.asarray(inverse_masses, dtype=np.float64).reshape(-1)
    radius = float(particle_radius)
    g = np.asarray(gravity, dtype=np.float64)
    mesh = trimesh.Trimesh(vertices=p, faces=f, process=False)
    triangle_tree = mesh.triangles_tree
    intersecting = (
        _global_intersection_filter(p, f, springs, mesh)
        if global_collision_filter
        else np.zeros(len(p), dtype=bool)
    )

    point_tri: list[tuple[int, int, int, int]] = []
    sidedness: list[bool] = []
    point_query_dist = radius * 3.0
    point_contact_threshold = 2.0 * radius + 0.5 * radius
    p_next = p + v * float(dt) + invm[:, None] * (float(dt) ** 2) * g

    for point_id, point in enumerate(p):
        if intersecting[point_id]:
            continue
        lower = point - point_query_dist
        upper = point + point_query_dist
        bounds = (*lower.tolist(), *upper.tolist())
        candidate_faces = triangle_tree.intersection(bounds)
        for face_id in candidate_faces:
            tri = f[int(face_id)]
            if point_id in tri:
                continue
            a, b, c = p[tri]
            bary = closest_point_triangle_barycentric(point, a, b, c)
            closest = bary[0] * a + bary[1] * b + bary[2] * c
            direction = closest - point
            distance = float(np.linalg.norm(direction))
            if distance <= 1.0e-12:
                continue
            direction /= distance
            tri_next = p_next[tri]
            tri_delta = tri_next - p[tri]
            point_delta = p_next[point_id] - point
            contact_delta = np.sum(tri_delta * bary[:, None], axis=0)
            distance_after = distance - float(np.dot(point_delta, direction)) + float(
                np.dot(contact_delta, direction)
            )
            if distance_after < point_contact_threshold:
                normal = np.cross(b - a, c - a)
                side = bool(np.dot(normal, point - a) < 0.0)
                point_tri.append((point_id, int(tri[0]), int(tri[1]), int(tri[2])))
                sidedness.append(side)
                if max_point_tri_contacts is not None and len(point_tri) >= max_point_tri_contacts:
                    break
        if max_point_tri_contacts is not None and len(point_tri) >= max_point_tri_contacts:
            break

    edge_pairs: list[tuple[int, int, int, int]] = []
    edge_normals: list[np.ndarray] = []
    radius_pad = radius * 3.0
    edge_contact_threshold = 2.0 * radius + 0.5 * radius

    # Build a 3-D R-tree of unique directed cloth edges. The source queries
    # triangle AABBs and then deduplicates reverse-oriented shared edges; an
    # edge AABB tree is equivalent for contact candidates but avoids iterating
    # unrelated edges from every nearby triangle.
    target_edges: list[tuple[int, int]] = []
    seen_undirected: set[tuple[int, int]] = set()
    for tri in f:
        p0, p1, p2 = (int(x) for x in tri)
        for e0, e1 in ((p0, p1), (p1, p2), (p2, p0)):
            key = (min(e0, e1), max(e0, e1))
            if key not in seen_undirected:
                seen_undirected.add(key)
                target_edges.append((e0, e1))
    target_edges_array = np.asarray(target_edges, dtype=np.int32).reshape(-1, 2)
    target_a = p[target_edges_array[:, 0]]
    target_b = p[target_edges_array[:, 1]]
    target_bounds = np.concatenate((np.minimum(target_a, target_b), np.maximum(target_a, target_b)), axis=1)
    from rtree import index as rtree_index
    edge_tree_properties = rtree_index.Property(dimension=3)
    edge_tree = rtree_index.Index(
        (
            (edge_id, tuple(float(x) for x in bounds), None)
            for edge_id, bounds in enumerate(target_bounds)
        ),
        properties=edge_tree_properties,
    )

    for pa_raw, pb_raw in springs:
        pa, pb = int(pa_raw), int(pb_raw)
        if intersecting[pa] or intersecting[pb]:
            continue
        a, b = p[pa], p[pb]
        mid = 0.5 * (a + b)
        edge_len = float(np.linalg.norm(a - b))
        max_distance = edge_len * 0.5 + radius_pad
        lo, hi = mid - max_distance, mid + max_distance
        bounds = (*lo.tolist(), *hi.tolist())
        for edge_id in sorted(edge_tree.intersection(bounds)):
            e0, e1 = (int(x) for x in target_edges_array[edge_id])
            if e0 == pa or e0 == pb or e1 == pa or e1 == pb:
                continue
            t_alpha, t_beta, distance = closest_segment_parameters(
                a, b, p[e0], p[e1]
            )
            if (
                t_alpha * edge_len < radius
                or (1.0 - t_alpha) * edge_len < radius
                or t_beta * edge_len < radius
                or (1.0 - t_beta) * edge_len < radius
            ):
                continue
            alpha_point = a * (1.0 - t_alpha) + b * t_alpha
            beta_point = p[e0] * (1.0 - t_beta) + p[e1] * t_beta
            delta = beta_point - alpha_point
            normal = np.zeros(3, dtype=np.float64) if distance <= 1.0e-12 else delta / distance
            alpha_next = p_next[pa] * (1.0 - t_alpha) + p_next[pb] * t_alpha
            beta_next = p_next[e0] * (1.0 - t_beta) + p_next[e1] * t_beta
            dist_after = distance - float(np.dot(alpha_next - alpha_point, normal)) + float(
                np.dot(beta_next - beta_point, normal)
            )
            if t_alpha < 1.0e-5 or dist_after >= edge_contact_threshold:
                continue
            edge_pairs.append((pa, pb, e0, e1))
            edge_normals.append(normal)
            if max_edge_edge_contacts is not None and len(edge_pairs) >= max_edge_edge_contacts:
                break
        if max_edge_edge_contacts is not None and len(edge_pairs) >= max_edge_edge_contacts:
            break

    return SelfContactData(
        point_tri_indices=np.asarray(point_tri, dtype=np.int32).reshape(-1, 4),
        point_tri_sidedness=np.asarray(sidedness, dtype=np.bool_),
        edge_indices=np.asarray(edge_pairs, dtype=np.int32).reshape(-1, 4),
        edge_normals=np.asarray(edge_normals, dtype=np.float32).reshape(-1, 3),
        intersecting_particles=intersecting,
    )
