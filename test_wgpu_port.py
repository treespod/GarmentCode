import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import trimesh

from pygarment.meshgen.simulation import run_sim
from pygarment.meshgen.wgpu_garment_port.garmentcode import (
    UnsupportedFeatureError,
    run_garmentcode_mesh,
)
from pygarment.meshgen.wgpu_garment_port.solver import (
    WgpuXPBDSolver,
    make_bend_arrays,
    make_spring_arrays,
)
from pygarment.meshgen.wgpu_garment_port.collision_filters import (
    build_body_collision_filter_data,
)
from pygarment.meshgen.wgpu_garment_port.geometry import BodyContactQuery


class Properties(dict):
    @property
    def properties(self):
        return self


class WgpuPortTests(unittest.TestCase):
    def test_run_sim_dispatches_to_wgpu_adapter(self):
        props = Properties({
            "sim": {
                "config": {
                    "max_sim_steps": 12,
                    "allow_unsupported_wgpu": True,
                    "sim_substeps": 4,
                },
                "stats": {
                    "sim_time": {},
                    "spf": {},
                    "fin_frame": {},
                    "body_collisions": {},
                    "self_collisions": {},
                },
            },
            "render": {"enabled": False, "stats": {"render_time": {}}},
        })
        paths = SimpleNamespace()
        result = {
            "frames": 12,
            "elapsed_seconds": 2.4,
            "body_contacts_at_initial_state": 3,
            "self_intersecting_particles_at_initial_state": 0,
        }

        with patch("pygarment.meshgen.simulation.run_from_paths", return_value=result) as run_backend:
            returned = run_sim("shirt", props, paths)

        run_backend.assert_called_once_with(
            paths,
            props,
            allow_unsupported=True,
            substeps=4,
        )
        self.assertIs(returned, result)
        self.assertEqual(props["sim"]["stats"]["fin_frame"]["shirt"], 11)
        self.assertEqual(
            props["sim"]["stats"]["body_contacts_at_initial_state"]["shirt"], 3
        )
        self.assertEqual(
            props["sim"]["stats"]["self_intersecting_particles_at_initial_state"]["shirt"],
            0,
        )
        self.assertEqual(props["render"]["stats"]["render_time"]["shirt"], 0.0)

    def test_unsupported_features_are_rejected_before_mesh_io(self):
        props = {
            "sim": {
                "config": {
                    "max_sim_steps": 10,
                    "options": {"enable_body_collision_filters": True},
                },
            },
        }

        with self.assertRaises(UnsupportedFeatureError):
            run_garmentcode_mesh(
                box_mesh_obj="missing-box.obj",
                simulation_props=props,
                output_obj="unused-output.obj",
            )

    def test_mesh_adapter_writes_obj_and_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_obj = Path(temp_dir) / "quad.obj"
            output_obj = Path(temp_dir) / "simulated.obj"
            input_obj.write_text(
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 1 1 0\n"
                "v 0 1 0\n"
                "f 1 2 3\n"
                "f 1 3 4\n",
                encoding="utf-8",
            )
            solver_instances = []

            class FakeSolver:
                def __init__(
                    self, positions, inverse_masses, spring_indices, spring_properties,
                    bend_indices, bend_properties, **kwargs
                ):
                    self.positions = positions
                    self.inverse_masses = inverse_masses
                    self.spring_indices = spring_indices
                    self.spring_properties = spring_properties
                    self.bend_indices = bend_indices
                    self.bend_properties = bend_properties
                    self.options = kwargs
                    solver_instances.append(self)

                def simulate(self, **kwargs):
                    return SimpleNamespace(
                        adapter="test adapter",
                        backend="test backend",
                        particles=len(self.positions),
                        springs=len(self.spring_indices),
                        bends=len(self.bend_indices),
                        point_triangle_contacts=0,
                        edge_edge_contacts=0,
                        frames=1,
                        substeps=kwargs["substeps"],
                        elapsed_seconds=0.01,
                    )

                def current_positions(self):
                    return self.positions

            with patch(
                "pygarment.meshgen.wgpu_garment_port.garmentcode.WgpuXPBDSolver",
                FakeSolver,
            ), self.assertWarnsRegex(RuntimeWarning, "vertex-label file is unavailable"):
                result = run_garmentcode_mesh(
                    box_mesh_obj=input_obj,
                    simulation_props={
                        "sim": {
                            "config": {
                                "max_sim_steps": 1,
                                "material": {},
                                "options": {"enable_attachment_constraint": True},
                            }
                        }
                    },
                    output_obj=output_obj,
                    substeps=2,
                )

            self.assertEqual(result["backend"], "wgpu-py")
            self.assertEqual(result["particles"], 4)
            self.assertEqual(result["springs"], 5)
            self.assertEqual(result["bending_constraints"], 1)
            self.assertEqual(result["substeps_per_frame"], 2)
            self.assertEqual(
                result["omitted_features"],
                ["attachment constraints (no matching garment vertex labels)"],
            )
            solver = solver_instances[0]
            self.assertEqual(solver.inverse_masses.shape, (4,))
            self.assertEqual(solver.spring_properties.shape, (5, 4))
            self.assertEqual(solver.bend_properties.shape, (1, 4))
            self.assertFalse(solver.options["body_drag_enabled"])
            self.assertTrue(output_obj.is_file())
            report_path = output_obj.with_suffix(".wgpu.json")
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), result)

    def test_body_collision_filters_exclude_semantic_faces(self):
        body = trimesh.creation.box()
        top_vertices = np.flatnonzero(body.vertices[:, 2] > 0.0).tolist()
        bottom_vertices = np.flatnonzero(body.vertices[:, 2] <= 0.0).tolist()
        cloth_vertices = np.array(
            [[-0.1, -0.1, 0.6], [0.1, -0.1, 0.6], [0.0, 0.1, 0.6]],
            dtype=np.float64,
        )
        cloth_faces = np.array([[0, 1, 2]], dtype=np.int32)

        with tempfile.TemporaryDirectory() as temp_dir:
            segmentation_path = Path(temp_dir) / "body_segmentation.json"
            segmentation_path.write_text(
                json.dumps({
                    "body": bottom_vertices,
                    "left_leg": top_vertices,
                    "left_arm": top_vertices,
                }),
                encoding="utf-8",
            )
            filter_data = build_body_collision_filter_data(
                ["skirt_panel"] * 3,
                cloth_vertices,
                cloth_faces,
                body.vertices,
                body.faces,
                segmentation_path,
            )

        self.assertTrue(np.all(filter_data.particle_filter_ids == 0))
        self.assertEqual(filter_data.face_filter_masks.shape, (2, len(body.faces)))
        self.assertTrue(np.any(filter_data.face_filter_masks[0]))

        query = BodyContactQuery(
            body.vertices,
            body.faces,
            particle_filter_ids=filter_data.particle_filter_ids,
            face_filter_masks=filter_data.face_filter_masks,
        )
        filtered_contacts = query(cloth_vertices)
        unfiltered_contacts = BodyContactQuery(body.vertices, body.faces)(cloth_vertices)
        filtered_distance = np.linalg.norm(
            filtered_contacts[:, 2, :3] - cloth_vertices, axis=1
        )
        unfiltered_distance = np.linalg.norm(
            unfiltered_contacts[:, 2, :3] - cloth_vertices, axis=1
        )
        self.assertTrue(np.all(filtered_distance > unfiltered_distance))

    def test_reference_drag_uses_configured_garment_body_pairs(self):
        body = trimesh.creation.box()
        inside_point = np.array([[0.0, 0.0, 0.0]])
        body_face_labels = ["body"] * len(body.faces)
        allowed = BodyContactQuery(
            body.vertices,
            body.faces,
            enable_reference_drag=True,
            body_face_labels=body_face_labels,
            particle_reference_labels=["left_arm"],
            reference_drag_pairs={("left_arm", "body")},
        )(inside_point)
        rejected = BodyContactQuery(
            body.vertices,
            body.faces,
            enable_reference_drag=True,
            body_face_labels=body_face_labels,
            particle_reference_labels=["left_leg"],
            reference_drag_pairs={("left_arm", "body")},
        )(inside_point)

        self.assertEqual(allowed[0, 1, 3], 1.0)
        self.assertEqual(rejected[0, 1, 3], 0.0)

    @unittest.skipUnless(
        os.environ.get("GARMENTCODE_RUN_WGPU_TESTS") == "1",
        "set GARMENTCODE_RUN_WGPU_TESTS=1 on a machine with a WebGPU adapter",
    )
    def test_real_wgpu_mesh_simulation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_obj = Path(temp_dir) / "quad.obj"
            output_obj = Path(temp_dir) / "simulated.obj"
            vertex_labels_path = Path(temp_dir) / "vertex_labels.yaml"
            body_measurements_path = Path(temp_dir) / "body_measurements.yaml"
            input_obj.write_text(
                "v 0 0 0\n"
                "v 1 0 0\n"
                "v 1 1 0\n"
                "v 0 1 0\n"
                "f 1 2 3\n"
                "f 1 3 4\n",
                encoding="utf-8",
            )
            vertex_labels_path.write_text(
                "lower_interface: [0]\nright_collar: [1]\n", encoding="utf-8"
            )
            body_measurements_path.write_text(
                "body:\n  height: 10\n  head_l: 2\n  waist_line: 7\n"
                "  neck_w: 4\n",
                encoding="utf-8",
            )

            result = run_garmentcode_mesh(
                box_mesh_obj=input_obj,
                simulation_props={
                    "sim": {
                        "config": {
                            "max_sim_steps": 1,
                            "material": {},
                            "options": {
                                "enable_attachment_constraint": True,
                                "attachment_label_names": ["lower_interface", "right_collar"],
                                "attachment_stiffness": [1000.0, 1000.0],
                                "attachment_damping": [10.0, 0.0],
                                "attachment_frames": 2,
                            },
                        }
                    }
                },
                output_obj=output_obj,
                vertex_labels_path=vertex_labels_path,
                body_measurements_path=body_measurements_path,
                substeps=1,
                gravity=(0.0, 0.0, 0.0),
            )

            self.assertEqual(result["backend"], "wgpu-py")
            self.assertEqual(result["frames"], 1)
            self.assertEqual(result["particles"], 4)
            self.assertEqual(result["attachment_constraints"], 2)
            self.assertGreater(result["springs"], 0)
            self.assertEqual(result["bending_constraints"], 1)
            self.assertTrue(np.isfinite(np.asarray(result["elapsed_seconds"])))
            self.assertTrue(output_obj.is_file())
            self.assertTrue(output_obj.with_suffix(".wgpu.json").is_file())

            positions = np.array(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
            )
            spring_indices, spring_properties = make_spring_arrays(
                np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.uint32),
                np.ones(4, dtype=np.float32),
                100.0,
                1.0,
            )
            bend_indices, bend_properties = make_bend_arrays(
                np.array([[1, 3, 0, 2]], dtype=np.uint32), 10.0, 1.0
            )
            solver = WgpuXPBDSolver(
                positions,
                np.ones(4, dtype=np.float32),
                spring_indices,
                spring_properties,
                bend_indices,
                bend_properties,
            )
            update_frames = []

            def update_contacts(current, velocities, frame):
                self.assertEqual(current.shape, (4, 3))
                self.assertEqual(velocities.shape, (4, 3))
                update_frames.append(frame)
                if frame == 1:
                    return (
                        np.array([[0, 1, 2, 3]], dtype=np.uint32),
                        np.array([0], dtype=np.uint32),
                        np.array([[0, 1, 2, 3]], dtype=np.uint32),
                        np.array([[0, 0, 1]], dtype=np.float32),
                    )
                return (
                    np.empty((0, 4), dtype=np.uint32),
                    np.empty(0, dtype=np.uint32),
                    np.empty((0, 4), dtype=np.uint32),
                    np.empty((0, 3), dtype=np.float32),
                )

            stats = solver.simulate(
                frames=3,
                substeps=1,
                gravity=(0.0, 0.0, 0.0),
                self_contact_updater=update_contacts,
            )
            self.assertEqual(update_frames, [1, 2])
            self.assertEqual(stats.point_triangle_contacts, 0)
            self.assertEqual(stats.edge_edge_contacts, 0)
            self.assertTrue(np.isfinite(solver.current_positions()).all())

if __name__ == "__main__":
    unittest.main()
