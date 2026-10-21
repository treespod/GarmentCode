import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pygarment.meshgen.simulation import run_sim
from pygarment.meshgen.wgpu_garment_port.garmentcode import (
    UnsupportedFeatureError,
    run_garmentcode_mesh,
)


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
            ):
                result = run_garmentcode_mesh(
                    box_mesh_obj=input_obj,
                    simulation_props={
                        "sim": {
                            "config": {
                                "max_sim_steps": 1,
                                "material": {},
                                "options": {},
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
            solver = solver_instances[0]
            self.assertEqual(solver.inverse_masses.shape, (4,))
            self.assertEqual(solver.spring_properties.shape, (5, 4))
            self.assertEqual(solver.bend_properties.shape, (1, 4))
            self.assertFalse(solver.options["body_drag_enabled"])
            self.assertTrue(output_obj.is_file())
            report_path = output_obj.with_suffix(".wgpu.json")
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), result)

    @unittest.skipUnless(
        os.environ.get("GARMENTCODE_RUN_WGPU_TESTS") == "1",
        "set GARMENTCODE_RUN_WGPU_TESTS=1 on a machine with a WebGPU adapter",
    )
    def test_real_wgpu_mesh_simulation(self):
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

            result = run_garmentcode_mesh(
                box_mesh_obj=input_obj,
                simulation_props={
                    "sim": {
                        "config": {
                            "max_sim_steps": 1,
                            "material": {},
                            "options": {},
                        }
                    }
                },
                output_obj=output_obj,
                substeps=1,
            )

            self.assertEqual(result["backend"], "wgpu-py")
            self.assertEqual(result["frames"], 1)
            self.assertEqual(result["particles"], 4)
            self.assertGreater(result["springs"], 0)
            self.assertEqual(result["bending_constraints"], 1)
            self.assertTrue(np.isfinite(np.asarray(result["elapsed_seconds"])))
            self.assertTrue(output_obj.is_file())
            self.assertTrue(output_obj.with_suffix(".wgpu.json").is_file())


if __name__ == "__main__":
    unittest.main()
