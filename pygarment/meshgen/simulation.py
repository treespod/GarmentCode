# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Garment simulation adapter
###########################################################################

import time
import trimesh

# Custom code
from pygarment.meshgen.sim_config import PathCofig
from pygarment.meshgen.wgpu_garment_port.garmentcode import run_from_paths
from pygarment.meshgen.wgpu_garment_port.geometry import read_obj_mesh

def optimize_garment_storage(paths: PathCofig):
    """Compact generated meshes and remove bulky texture files."""
    try:
        boxmesh = trimesh.load(paths.g_box_mesh)
        boxmesh.export(paths.g_box_mesh_compressed)
        paths.g_box_mesh.unlink()
    except BaseException:
        pass

    try:
        simmesh = trimesh.load(paths.g_sim)
        simmesh.export(paths.g_sim_compressed)
        paths.g_sim.unlink()
    except BaseException:
        pass

    paths.g_texture_fabric.unlink(missing_ok=True)
    paths.g_mtl.unlink(missing_ok=True)

def run_sim(
        cloth_name, props, paths: PathCofig, 
        save_v_norms=False, store_usd=False, 
        optimize_storage=False,
        verbose=False): 
    """Simulate a generated garment mesh with the wgpu-py XPBD backend."""
    if save_v_norms or store_usd:
        raise NotImplementedError("The wgpu-py backend does not export USD frames or vertex normals")

    sim_props = props['sim']
    sim_config = sim_props['config']
    render_props = props['render']
    start_time = time.perf_counter()

    print("WGPU simulation..")
    result = run_from_paths(
        paths,
        props.properties,
        allow_unsupported=bool(sim_config.get('allow_unsupported_wgpu', False)),
        substeps=int(sim_config.get('sim_substeps', 10)),
    )

    frames = result['frames']
    sim_time = result['elapsed_seconds']
    sim_props['stats']['sim_time'][cloth_name] = sim_time
    sim_props['stats']['spf'][cloth_name] = sim_time / frames if frames else sim_time
    sim_props['stats']['fin_frame'][cloth_name] = frames - 1
    sim_props['stats'].setdefault('body_contacts_at_initial_state', {})[cloth_name] = result['body_contacts_at_initial_state']
    sim_props['stats'].setdefault('self_intersecting_particles_at_initial_state', {})[cloth_name] = result['self_intersecting_particles_at_initial_state']

    print(f"Simulation took #frames={frames}")
    print(f"Initial body contacts: {result['body_contacts_at_initial_state']}")
    print(f"Initial self-intersecting particles: {result['self_intersecting_particles_at_initial_state']}")

    if render_props.get('enabled', True):
        from pygarment.meshgen.render.pythonrender import render_images

        body_vertices, body_faces = read_obj_mesh(paths.in_body_obj)
        start_render = time.perf_counter()
        render_images(paths, body_vertices, body_faces, render_props['config'])
        render_time = time.perf_counter() - start_render
        render_props['stats']['render_time'][cloth_name] = render_time
        print(f"Rendering {cloth_name} took {render_time}s")
    else:
        render_props['stats']['render_time'][cloth_name] = 0.0

    if optimize_storage:
        optimize_garment_storage(paths)

    total_time = time.perf_counter() - start_time
    print(f"\nSimulation pipeline took: {total_time:.3f}s")
    return result
