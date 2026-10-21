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
from .self_collision import build_self_contact_data
from .solver import WgpuXPBDSolver, make_bend_arrays, make_spring_arrays


class UnsupportedFeatureError(NotImplementedError):
    """Raised instead of silently claiming parity for unported contacts."""


def _simulation_config(props: dict[str, Any]) -> dict[str, Any]:
    if "sim" in props and isinstance(props["sim"], dict):
        return props["sim"].get("config", {})
    return props.get("config", props)


def _attachment_labels_present(
    vertex_labels_path: str | Path | None, options: dict[str, Any]
) -> bool:
    if vertex_labels_path is None or not Path(vertex_labels_path).exists():
        # Unknown is treated as potentially active so an attachment option
        # cannot be silently ignored by callers that omit its label file.
        return True
    import yaml

    labels = yaml.safe_load(Path(vertex_labels_path).read_text(encoding="utf-8")) or {}
    requested = options.get("attachment_label_names", ()) or ()
    return any(bool(labels.get(name)) for name in requested)


def _missing_features(
    config: dict[str, Any],
    *,
    frames: int,
    attachment_labels_present: bool,
    body_present: bool,
) -> list[str]:
    options = config.get("options", {})
    missing: list[str] = []
    if bool(options.get("enable_particle_particle_collisions", False)):
        missing.append("enable_particle_particle_collisions")
    if bool(options.get("enable_body_collision_filters", False)):
        missing.append("enable_body_collision_filters")
    if bool(options.get("enable_attachment_constraint", False)) and attachment_labels_present:
        missing.append("attachment constraints")
    if bool(options.get("enable_cloth_reference_drag", False)):
        missing.append("cloth/self reference drag (body reference drag is implemented)")
    self_contacts_enabled = bool(options.get("enable_triangle_particle_collisions", False)) or bool(
        options.get("enable_edge_edge_collisions", False)
    )
    if self_contacts_enabled and frames > 1:
        missing.append("per-frame self-contact regeneration (contact pairs are prepared at frame 0)")
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
    vertex_labels_path: str | Path | None = None,
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
    attachment_labels_present = _attachment_labels_present(vertex_labels_path, options)
    body_present = body_obj is not None and Path(body_obj).exists()
    missing = _missing_features(
        config,
        frames=frames,
        attachment_labels_present=attachment_labels_present,
        body_present=body_present,
    )
    if missing and not allow_unsupported:
        raise UnsupportedFeatureError(
            "This WGPU port implements prediction, springs, bending, body contacts, and "
            "single-frame point-triangle/edge-edge contacts, but not: " + ", ".join(missing)
            + ". Disable those features explicitly for a supported parity run, or pass "
            "allow_unsupported=True for an approximate run (not a parity result)."
        )
    if missing:
        warnings.warn(
            "Running with omitted features: " + ", ".join(missing),
            RuntimeWarning,
            stacklevel=2,
        )

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
    if body_vertices is not None and body_faces is not None:
        body_contact_query = BodyContactQuery(
            body_vertices,
            body_faces,
            particle_radius=particle_radius,
            body_thickness=body_thickness,
            margin=0.2,
            enable_reference_drag=body_drag_enabled,
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
    if bool(options.get("enable_triangle_particle_collisions", False)) or bool(
        options.get("enable_edge_edge_collisions", False)
    ):
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
        body_contact_updater=body_contact_query,
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
        "body_contacts_at_initial_state": body_contact_count,
        "point_triangle_contacts_at_initial_state": stats.point_triangle_contacts,
        "edge_edge_contacts_at_initial_state": stats.edge_edge_contacts,
        "self_intersecting_particles_at_initial_state": self_intersecting_particles,
        "frames": stats.frames,
        "substeps_per_frame": stats.substeps,
        "elapsed_seconds": stats.elapsed_seconds,
        "y_shift": y_shift,
        "omitted_features": missing,
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
        vertex_labels_path=getattr(paths, "g_vert_labels", None),
        **kwargs,
    )
