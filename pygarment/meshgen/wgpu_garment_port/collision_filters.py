"""Garment-to-body semantic collision filtering for the WGPU simulation path."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REFERENCE_DRAG_PAIRS = {
    ("left_arm", "body"),
    ("right_arm", "body"),
    ("left_leg", "right_leg"),
    ("left_arm", "left_leg"),
    ("right_arm", "left_leg"),
    ("left_arm", "right_leg"),
    ("right_arm", "right_leg"),
    ("left_arm", "legs"),
    ("right_arm", "legs"),
}

_SMPL_SEGMENTS = {
    "left_arm": ("leftArm", "leftForeArm", "leftHand", "leftHandIndex1"),
    "right_arm": ("rightArm", "rightForeArm", "rightHand", "rightHandIndex1"),
    "left_leg": ("leftUpLeg", "leftLeg", "leftFoot", "leftToeBase", "hips"),
    "right_leg": ("rightUpLeg", "rightLeg", "rightFoot", "rightToeBase", "hips"),
    "body": (
        "spine", "spine1", "spine2", "neck", "leftShoulder", "rightShoulder",
        "hips", "leftUpLeg", "leftLeg", "leftFoot", "leftToeBase",
        "rightUpLeg", "rightLeg", "rightFoot", "rightToeBase",
    ),
}


@dataclass
class BodyCollisionFilterData:
    particle_filter_ids: np.ndarray
    face_filter_masks: np.ndarray
    particle_reference_labels: list[str]
    body_face_labels: list[str]


def _expand_body_segments(
    segmentation: dict[str, list[int]], *, smpl_body: bool
) -> dict[str, list[int]]:
    if not smpl_body:
        return segmentation
    expanded = {
        name: [
            int(vertex_id)
            for source_name in source_names
            for vertex_id in segmentation.get(source_name, ())
        ]
        for name, source_names in _SMPL_SEGMENTS.items()
    }
    return expanded


def _face_majority_labels(
    faces: np.ndarray, vertex_labels: list[str]
) -> list[str]:
    face_labels: list[str] = []
    for face in faces:
        labels = [vertex_labels[int(vertex)] for vertex in face]
        face_labels.append(max(dict.fromkeys(labels), key=labels.count))
    return face_labels


def _body_filter_mask(face_labels: list[str], parts: set[str]) -> np.ndarray:
    return np.asarray([label in parts for label in face_labels], dtype=bool)


def build_body_collision_filter_data(
    cloth_labels: list[str],
    cloth_vertices: np.ndarray,
    cloth_faces: np.ndarray,
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    body_segmentation_path: str | Path,
    *,
    panel_initial_labels: dict[str, str] | None = None,
    smpl_body: bool = False,
) -> BodyCollisionFilterData:
    """Map garment panels to body regions and construct the configured masks."""
    import trimesh

    cloth_vertices = np.asarray(cloth_vertices, dtype=np.float64).reshape(-1, 3)
    cloth_faces = np.asarray(cloth_faces, dtype=np.int32).reshape(-1, 3)
    body_vertices = np.asarray(body_vertices, dtype=np.float64).reshape(-1, 3)
    body_faces = np.asarray(body_faces, dtype=np.int32).reshape(-1, 3)
    if len(cloth_labels) != len(cloth_vertices):
        raise ValueError("cloth labels must have one entry per cloth vertex")
    if body_faces.size and (
        int(body_faces.min()) < 0 or int(body_faces.max()) >= len(body_vertices)
    ):
        raise ValueError("body face index is out of range for body vertices")
    raw_segmentation: dict[str, list[int]] = json.loads(
        Path(body_segmentation_path).read_text(encoding="utf-8")
    )
    segmentation = _expand_body_segments(raw_segmentation, smpl_body=smpl_body)
    body_vertex_labels = [""] * len(body_vertices)
    for part, vertex_ids in segmentation.items():
        for vertex_id in vertex_ids:
            if 0 <= vertex_id < len(body_vertex_labels):
                body_vertex_labels[vertex_id] = part
    body_face_labels = _face_majority_labels(body_faces, body_vertex_labels)

    query_mesh = trimesh.Trimesh(
        vertices=np.asarray(body_vertices), faces=np.asarray(body_faces), process=False
    )
    _, _, closest_face_ids = trimesh.proximity.ProximityQuery(query_mesh).on_surface(
        np.asarray(cloth_vertices)
    )
    body_vertex_parts: list[list[str]] = [[] for _ in range(len(body_vertices))]
    for part, vertex_ids in segmentation.items():
        for vertex_id in vertex_ids:
            if 0 <= vertex_id < len(body_vertex_parts):
                body_vertex_parts[vertex_id].append(part)

    panel_initial_labels = panel_initial_labels or {}
    assigned_labels = [""] * len(cloth_vertices)
    panel_names = list(dict.fromkeys(cloth_labels))
    for panel_name in panel_names:
        if not panel_name or panel_name.lower().startswith("stitch"):
            continue
        panel_vertex_ids = np.flatnonzero(np.asarray(cloth_labels) == panel_name)
        scores = {part: 0.0 for part in segmentation}
        for face_id in closest_face_ids[panel_vertex_ids]:
            for vertex_id in body_faces[int(face_id)]:
                for part in body_vertex_parts[int(vertex_id)]:
                    scores[part] += 1.0 / 3.0
        preferred = panel_initial_labels.get(panel_name, "")
        if preferred:
            scores = {
                part: score if preferred in part else -1.0
                for part, score in scores.items()
            }
        if not scores:
            continue
        assigned = max(scores, key=scores.get)
        if (
            assigned in {"left_leg", "right_leg"}
            and "pant" not in panel_name.lower()
        ):
            assigned = "legs"
        for vertex_id in panel_vertex_ids:
            assigned_labels[int(vertex_id)] = assigned

    neighbors: list[set[int]] = [set() for _ in range(len(cloth_vertices))]
    for face in np.asarray(cloth_faces, dtype=np.int64):
        for vertex_id in face:
            neighbors[int(vertex_id)].update(int(other) for other in face if other != vertex_id)
    for vertex_id, label in enumerate(cloth_labels):
        if not label.lower().startswith("stitch"):
            continue
        adjacent_labels = [
            assigned_labels[neighbor]
            for neighbor in neighbors[vertex_id]
            if assigned_labels[neighbor]
        ]
        if adjacent_labels:
            assigned_labels[vertex_id] = max(
                set(adjacent_labels), key=adjacent_labels.count
            )

    particle_filter_ids = np.full(len(cloth_vertices), -1, dtype=np.int32)
    for vertex_id, label in enumerate(assigned_labels):
        if label in {"left_leg", "right_leg", "legs"}:
            particle_filter_ids[vertex_id] = 0
        if label == "body" and particle_filter_ids[vertex_id] < 0:
            particle_filter_ids[vertex_id] = 1

    masks = np.stack(
        [
            _body_filter_mask(body_face_labels, {"left_arm", "right_arm", "arms"}),
            _body_filter_mask(body_face_labels, {"face_internal"}),
        ]
    )
    return BodyCollisionFilterData(
        particle_filter_ids=particle_filter_ids,
        face_filter_masks=masks,
        particle_reference_labels=assigned_labels,
        body_face_labels=body_face_labels,
    )
