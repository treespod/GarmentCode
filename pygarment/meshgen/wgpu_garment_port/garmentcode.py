"""Adapter from GarmentCode's generated OBJ/config to the WGPU XPBD solver."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (
    BodyContactQuery,
    build_cloth_model_data,
    read_obj_mesh,
    write_obj_with_positions,
)
from .collision_filters import REFERENCE_DRAG_PAIRS, build_body_collision_filter_data
from .self_collision import build_self_contact_data
from .solver import WgpuXPBDSolver, make_bend_arrays, make_spring_arrays


class UnsupportedFeatureError(NotImplementedError):
    """Raised instead of silently claiming parity for unported contacts."""


def _simulation_config(props: dict[str, Any]) -> dict[str, Any]:
    if "sim" in props and isinstance(props["sim"], dict):
        return props["sim"].get("config", {})
    return props.get("config", props)


def _build_attachment_data(
    positions: np.ndarray,
    *,
    vertex_labels_path: str | Path | None,
    body_measurements_path: str | Path | None,
    options: dict[str, Any],
    y_shift: float,
) -> tuple[np.ndarray, np.ndarray]:
    particle_count = len(positions)
    per_particle: list[list[np.ndarray]] = [[] for _ in range(particle_count)]
    if not bool(options.get("enable_attachment_constraint", False)):
        return np.empty((0, 3, 4), dtype=np.float32), np.zeros(
            particle_count + 1, dtype=np.uint32
        )
    if vertex_labels_path is None or not Path(vertex_labels_path).is_file():
        warnings.warn(
            "Attachment constraints are enabled, but the garment vertex-label file "
            "is unavailable; no vertices will be attached.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.empty((0, 3, 4), dtype=np.float32), np.zeros(
            particle_count + 1, dtype=np.uint32
        )
    import yaml

    labels = yaml.safe_load(Path(vertex_labels_path).read_text(encoding="utf-8")) or {}
    requested = list(options.get("attachment_label_names", ()) or ())
    stiffness = list(options.get("attachment_stiffness", ()) or ())
    damping = list(options.get("attachment_damping", ()) or ())
    if len(stiffness) < len(requested) or len(damping) < len(requested):
        raise ValueError(
            "attachment_stiffness and attachment_damping must have one value "
            "for each attachment_label_name"
        )
    active = [
        (index, name, labels.get(name, ()))
        for index, name in enumerate(requested)
        if labels.get(name)
    ]
    if not active:
        warnings.warn(
            f"None of the requested attachment labels are present: {requested}",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.empty((0, 3, 4), dtype=np.float32), np.zeros(
            particle_count + 1, dtype=np.uint32
        )
    if body_measurements_path is None or not Path(body_measurements_path).is_file():
        raise FileNotFoundError(
            "Body measurements are required to position garment attachment constraints"
        )
    body = yaml.safe_load(
        Path(body_measurements_path).read_text(encoding="utf-8")
    )["body"]

    def attachment_target(name: str) -> tuple[np.ndarray, np.ndarray]:
        if name == "lower_interface":
            y = (
                body["_waist_level"]
                if "_waist_level" in body
                else body["height"] - body["head_l"] - body["waist_line"]
            )
            return np.array([0.0, float(y) + y_shift, 0.0]), np.array([0.0, 1.0, 0.0])
        if name == "right_collar":
            return (
                np.array([-0.5 * (body["neck_w"] - 2.0), 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            )
        if name == "left_collar":
            return (
                np.array([0.5 * (body["neck_w"] - 2.0), 0.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
            )
        if name == "strapless_top":
            return (
                np.array([
                    0.0,
                    body["height"] - body["head_l"] - body["armscye_depth"] + y_shift,
                    0.0,
                ]),
                np.array([0.0, 1.0, 0.0]),
            )
        raise UnsupportedFeatureError(
            f"Attachment target geometry is not defined for label {name!r}"
        )

    for label_index, name, vertex_ids in active:
        anchor, normal = attachment_target(name)
        k = float(stiffness[label_index])
        d = float(damping[label_index])
        if k < 0.0 or d < 0.0:
            raise ValueError("Attachment stiffness and damping must be nonnegative")
        record = np.zeros((3, 4), dtype=np.float32)
        record[0, :3] = anchor
        record[1, :3] = normal
        record[2, :2] = (k, d)
        for vertex_id in vertex_ids:
            vertex_id = int(vertex_id)
            if vertex_id < 0 or vertex_id >= particle_count:
                raise ValueError(
                    f"Attachment label {name!r} contains invalid vertex index {vertex_id}"
                )
            per_particle[vertex_id].append(record)

    offsets = np.zeros(particle_count + 1, dtype=np.uint32)
    for vertex_id, records in enumerate(per_particle):
        offsets[vertex_id + 1] = offsets[vertex_id] + len(records)
    data = (
        np.asarray(
            [record for records in per_particle for record in records],
            dtype=np.float32,
        ).reshape(-1, 3, 4)
        if offsets[-1]
        else np.empty((0, 3, 4), dtype=np.float32)
    )
    return data, offsets


def _missing_features(
    config: dict[str, Any],
    *,
    body_mesh_available: bool,
    body_segmentation_available: bool,
    cloth_segmentation_available: bool,
) -> list[str]:
    options = config.get("options", {})
    missing: list[str] = []
    if bool(options.get("enable_particle_particle_collisions", False)):
        missing.append("enable_particle_particle_collisions")
    collision_filter_data_available = (
        body_mesh_available
        and body_segmentation_available
        and cloth_segmentation_available
    )
    if bool(options.get("enable_body_collision_filters", False)) and not collision_filter_data_available:
        missing.append("enable_body_collision_filters")
    if bool(options.get("enable_cloth_reference_drag", False)) and not collision_filter_data_available:
        missing.append("cloth/self reference drag (body reference drag is implemented)")
    if bool(config.get("ground", False)):
        missing.append("ground contacts")
    return missing


def run_garmentcode_mesh(
    *,
    box_mesh_obj: str | Path,
    simulation_props: dict[str, Any],
    output_obj: str | Path,
    segmentation_path: str | Path | None = None,
    original_lengths_path: str | Path | None = None,
    body_obj: str | Path | None = None,
    body_segmentation_path: str | Path | None = None,
    panel_specification_path: str | Path | None = None,
    body_measurements_path: str | Path | None = None,
    vertex_labels_path: str | Path | None = None,
    smpl_body: bool = False,
    allow_unsupported: bool = False,
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
    fps: float = 60.0,
    substeps: int = 10,
) -> dict[str, Any]:
    """Run the supported XPBD cloth subset and write a GarmentCode-style OBJ.

    This function consumes the exact box mesh produced by ``BoxMesh``. It does
    not need Warp. The input mesh is not regenerated or remeshed.
    """
    config = _simulation_config(simulation_props)
    frames = int(config.get("max_sim_steps", 1))
    options = config.get("options", {})
    body_mesh_available = body_obj is not None and Path(body_obj).is_file()
    body_segmentation_available = (
        body_segmentation_path is not None and Path(body_segmentation_path).is_file()
    )
    cloth_segmentation_available = (
        segmentation_path is not None and Path(segmentation_path).is_file()
    )
    missing = _missing_features(
        config,
        body_mesh_available=body_mesh_available,
        body_segmentation_available=body_segmentation_available,
        cloth_segmentation_available=cloth_segmentation_available,
    )
    if missing and not allow_unsupported:
        raise UnsupportedFeatureError(
            "This WGPU port implements prediction, springs, bending, body contacts, "
            "per-frame self contacts, semantic body filters, attachments, and reference drag "
            "when their source data is available, but not: " + ", ".join(missing)
            + ". Disable those features explicitly for a supported parity run, or pass "
            "allow_unsupported=True for an approximate run (not a parity result)."
        )
    if missing:
        warnings.warn(
            "Running with omitted features: " + ", ".join(missing),
            RuntimeWarning,
            stacklevel=2,
        )
    omitted_features = list(missing)

    vertices, faces = read_obj_mesh(box_mesh_obj)
    y_shift = 0.0
    body_vertices = None
    body_faces = None
    if body_obj is not None and Path(body_obj).exists():
        body_vertices, body_faces = read_obj_mesh(body_obj)
        body_vertices = body_vertices * 100.0  # same scale as pygarment.meshgen.garment.Cloth
        min_y = float(body_vertices[:, 1].min())
        if min_y < 0.0:
            y_shift = -min_y
        body_vertices[:, 1] += y_shift
    vertices[:, 1] += y_shift

    material = config.get("material", {})
    model = build_cloth_model_data(
        vertices,
        faces,
        segmentation_path=segmentation_path,
        original_lengths_path=original_lengths_path,
        fabric_density=float(material.get("fabric_density", 1.0)),
        particle_mass_seed=0.1,
    )
    attachment_data, attachment_offsets = _build_attachment_data(
        model.positions,
        vertex_labels_path=vertex_labels_path,
        body_measurements_path=body_measurements_path,
        options=options,
        y_shift=y_shift,
    )
    if (
        bool(options.get("enable_attachment_constraint", False))
        and not int(attachment_offsets[-1])
    ):
        omitted_features.append(
            "attachment constraints (no matching garment vertex labels)"
        )
    spring_idx, spring_props = make_spring_arrays(
        model.spring_indices,
        model.spring_rest_lengths,
        float(material.get("spring_ke", 50000.0)),
        float(material.get("spring_kd", 10.0)),
    )
    bend_idx, bend_props = make_bend_arrays(
        model.bend_indices,
        float(material.get("garment_edge_ke", 50000.0)),
        float(material.get("garment_edge_kd", 10.0)),
        rest_angle=0.0,
    )
    options = config.get("options", {})
    particle_radius = float(material.get("fabric_thickness", 0.1))
    body_thickness = float(options.get("body_collision_thickness", 0.0))
    body_contact_data = None
    body_contact_count = 0
    body_drag_enabled = bool(options.get("enable_cloth_reference_drag", False))
    body_contact_query = None
    body_collision_filter_data = None
    use_semantic_body_filter = (
        body_mesh_available
        and body_segmentation_available
        and cloth_segmentation_available
        and (
            bool(options.get("enable_body_collision_filters", False))
            or body_drag_enabled
        )
    )
    if use_semantic_body_filter:
        panel_initial_labels: dict[str, str] = {}
        if panel_specification_path is not None and Path(panel_specification_path).is_file():
            specification = json.loads(
                Path(panel_specification_path).read_text(encoding="utf-8")
            )
            panel_initial_labels = {
                name: str(panel.get("label", "") or "")
                for name, panel in specification.get("panels", {}).items()
            }
        body_collision_filter_data = build_body_collision_filter_data(
            model.segmentation,
            model.positions,
            faces,
            body_vertices,
            body_faces,
            body_segmentation_path,
            panel_initial_labels=panel_initial_labels,
            smpl_body=smpl_body,
        )
    if body_vertices is not None and body_faces is not None:
        body_contact_query = BodyContactQuery(
            body_vertices,
            body_faces,
            particle_radius=particle_radius,
            body_thickness=body_thickness,
            margin=0.2,
            enable_reference_drag=body_drag_enabled,
            face_filter_masks=(
                body_collision_filter_data.face_filter_masks
                if body_collision_filter_data is not None
                and bool(options.get("enable_body_collision_filters", False))
                else None
            ),
            particle_filter_ids=(
                body_collision_filter_data.particle_filter_ids
                if body_collision_filter_data is not None
                and bool(options.get("enable_body_collision_filters", False))
                else None
            ),
            body_face_labels=(
                body_collision_filter_data.body_face_labels
                if body_collision_filter_data is not None
                else None
            ),
            particle_reference_labels=(
                body_collision_filter_data.particle_reference_labels
                if body_collision_filter_data is not None
                else None
            ),
            reference_drag_pairs=(
                REFERENCE_DRAG_PAIRS
                if body_collision_filter_data is not None and body_drag_enabled
                else None
            ),
        )
        body_contact_data = body_contact_query(model.positions)
        body_contact_count = int(np.count_nonzero(body_contact_data[:, 0, 3]))

    frames = int(config.get("max_sim_steps", 1))
    zero_gravity_steps = int(config.get("zero_gravity_steps", 0))
    frame0_gravity = (0.0, 0.0, 0.0) if zero_gravity_steps > 0 else gravity
    point_tri_indices = np.empty((0, 4), dtype=np.uint32)
    point_tri_sidedness = np.empty(0, dtype=np.uint32)
    edge_edge_indices = np.empty((0, 4), dtype=np.uint32)
    edge_edge_normals = np.empty((0, 3), dtype=np.float32)
    self_intersecting_particles = 0
    self_contacts_enabled = bool(
        options.get("enable_triangle_particle_collisions", False)
    ) or bool(options.get("enable_edge_edge_collisions", False))
    if self_contacts_enabled:
        self_contacts = build_self_contact_data(
            model.positions,
            np.zeros_like(model.positions),
            faces,
            model.spring_indices,
            model.inverse_masses,
            particle_radius=particle_radius,
            dt=1.0 / float(fps),
            gravity=frame0_gravity,
            global_collision_filter=bool(options.get("enable_global_collision_filter", False)),
        )
        self_intersecting_particles = int(np.count_nonzero(self_contacts.intersecting_particles))
        if bool(options.get("enable_triangle_particle_collisions", False)):
            point_tri_indices = self_contacts.point_tri_indices.astype(np.uint32)
            point_tri_sidedness = self_contacts.point_tri_sidedness.astype(np.uint32)
        if bool(options.get("enable_edge_edge_collisions", False)):
            edge_edge_indices = self_contacts.edge_indices.astype(np.uint32)
            edge_edge_normals = self_contacts.edge_normals
    initial_point_triangle_contacts = len(point_tri_indices)
    initial_edge_edge_contacts = len(edge_edge_indices)

    def update_self_contacts(
        positions: np.ndarray, velocities: np.ndarray, frame: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame_gravity = (
            (0.0, 0.0, 0.0) if frame < zero_gravity_steps else gravity
        )
        contacts = build_self_contact_data(
            positions,
            velocities,
            faces,
            model.spring_indices,
            model.inverse_masses,
            particle_radius=particle_radius,
            dt=1.0 / float(fps),
            gravity=frame_gravity,
            global_collision_filter=bool(
                options.get("enable_global_collision_filter", False)
            ),
        )
        return (
            contacts.point_tri_indices
            if bool(options.get("enable_triangle_particle_collisions", False))
            else np.empty((0, 4), dtype=np.int32),
            contacts.point_tri_sidedness
            if bool(options.get("enable_triangle_particle_collisions", False))
            else np.empty(0, dtype=np.uint32),
            contacts.edge_indices
            if bool(options.get("enable_edge_edge_collisions", False))
            else np.empty((0, 4), dtype=np.int32),
            contacts.edge_normals
            if bool(options.get("enable_edge_edge_collisions", False))
            else np.empty((0, 3), dtype=np.float32),
        )

    solver = WgpuXPBDSolver(
        model.positions,
        model.inverse_masses,
        spring_idx,
        spring_props,
        bend_idx,
        bend_props,
        damping=float(options.get("global_damping_factor", 0.25)),
        damping_min_speed=float(options.get("global_damping_effective_velocity", 0.0)),
        max_velocity=float(options.get("global_max_velocity", 25.0)),
        body_contact_data=body_contact_data,
        attachment_data=attachment_data,
        attachment_offsets=attachment_offsets,
        point_tri_indices=point_tri_indices,
        point_tri_sidedness=point_tri_sidedness,
        edge_edge_indices=edge_edge_indices,
        edge_edge_normals=edge_edge_normals,
        particle_radius=particle_radius,
        body_mu=0.5 * (
            float(material.get("fabric_friction", 0.5))
            + float(options.get("body_friction", 0.5))
        ),
        contact_relaxation=0.9,
        body_drag_enabled=body_drag_enabled,
    )
    stats = solver.simulate(
        frames=frames,
        fps=fps,
        substeps=substeps,
        gravity=gravity,
        zero_gravity_steps=zero_gravity_steps,
        attachment_frames=(
            int(options.get("attachment_frames", 100))
            if bool(options.get("enable_attachment_constraint", False))
            else 0
        ),
        body_contact_updater=body_contact_query,
        self_contact_updater=update_self_contacts if self_contacts_enabled else None,
    )
    final_positions = solver.current_positions()
    write_obj_with_positions(box_mesh_obj, output_obj, final_positions)

    result = {
        "backend": "wgpu-py",
        "adapter": stats.adapter,
        "adapter_backend": stats.backend,
        "particles": stats.particles,
        "triangles": int(len(faces)),
        "springs": stats.springs,
        "bending_constraints": stats.bends,
        "attachment_constraints": int(attachment_offsets[-1]),
        "body_contacts_at_initial_state": body_contact_count,
        "body_collision_filtered_particles": (
            int(np.count_nonzero(body_collision_filter_data.particle_filter_ids >= 0))
            if body_collision_filter_data is not None
            and bool(options.get("enable_body_collision_filters", False))
            else 0
        ),
        "cloth_reference_drag_enabled": bool(
            body_collision_filter_data is not None and body_drag_enabled
        ),
        "point_triangle_contacts_at_initial_state": initial_point_triangle_contacts,
        "edge_edge_contacts_at_initial_state": initial_edge_edge_contacts,
        "self_intersecting_particles_at_initial_state": self_intersecting_particles,
        "frames": stats.frames,
        "substeps_per_frame": stats.substeps,
        "elapsed_seconds": stats.elapsed_seconds,
        "y_shift": y_shift,
        "omitted_features": omitted_features,
        "input_obj": str(Path(box_mesh_obj).resolve()),
        "output_obj": str(Path(output_obj).resolve()),
    }
    report_path = Path(output_obj).with_suffix(".wgpu.json")
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_from_paths(paths: Any, simulation_props: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Convenience adapter for the ``PathCofig`` object in GarmentCode."""
    return run_garmentcode_mesh(
        box_mesh_obj=paths.g_box_mesh,
        simulation_props=simulation_props,
        output_obj=paths.g_sim,
        segmentation_path=paths.g_mesh_segmentation,
        original_lengths_path=paths.g_orig_edge_len,
        body_obj=paths.in_body_obj,
        body_segmentation_path=paths.body_seg,
        panel_specification_path=getattr(paths, "g_specs", None),
        body_measurements_path=paths.in_body_mes,
        vertex_labels_path=getattr(paths, "g_vert_labels", None),
        smpl_body=bool(getattr(paths, "use_smpl_seg", False)),
        **kwargs,
    )
