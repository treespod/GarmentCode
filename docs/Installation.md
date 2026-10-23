# Installation

## Local paths setup 

Create system.json file in the root of this directory with your machine's file paths using `system.template.json` as a template. 
`system.json` should include the following: 
* Path for creating logs for one-off scripts (`'output'`)
* Path to the folder with (generated) datasets of sewing patterns (`'datasets_path'`)
* Path to the folder with simulation results on the datasets of sewing patterns (`'datasets_sim'`)

* Data generation & Simulation resources  
    * path to folder with simulation\rendering configurations (`'sim_configs_path'`)
    * path to folder containing body files for neutral body and other base body models (`'bodies_default_path'`)
    * path to folder containing datasets of body shape samples (`'body_samples_path'`)
    

## Simulation backend

Garment simulation uses the wgpu-py XPBD backend. Install `wgpu` through the project dependencies and ensure the machine has a WebGPU adapter (Vulkan on Linux, Metal on macOS, or D3D12 on Windows); Mesa lavapipe can provide a CPU Vulkan adapter. The backend supports per-frame point-triangle/edge-edge contacts, semantic body collision filtering, labeled attachment constraints, and panel/body reference drag when the corresponding mesh segmentation and measurement inputs are available. Particle-particle collisions, ground contacts, USD frame caching, and vertex-normal export are not implemented.

`allow_unsupported_wgpu` allows a run to proceed when a requested feature cannot be applied; it does not implement that feature. Omitted features are recorded in the WGPU JSON report.

Run the one-frame core verification with `python test_garment_sim.py --sim_config assets/Sim_props/wgpu_core_verify.yaml`. This config disables rendering and features not yet implemented by the WGPU port.

On a machine with a WebGPU adapter, run the device-backed integration test with `GARMENTCODE_RUN_WGPU_TESTS=1 python -m unittest test_wgpu_port.WgpuPortTests.test_real_wgpu_mesh_simulation -v` (use your shell's syntax to set the environment variable).

## Using pip

With pip, you can install the core pygarment library and its dependencies to start writing your own garment programs!

```
pip install pygarment
```

## Manual Installation

If required, you could install the library and dependecies manually

### pygarment uses the following dependencies:

These dependencies are installed through [setup.cfg](https://github.com/maria-korosteleva/GarmentCode/blob/main/setup.cfg#L23).

* Python 3.9
* numpy<2
* scipy
* pyyaml >= 6.0
* [svgwrite](https://pypi.org/project/svgwrite/)
* psutil
* matplotlib
* [svgpathtools](https://github.com/mathandy/svgpathtools)
* [cairoSVG](https://cairosvg.org/)
    NOTE: this lib has some quirks on Windows, which we resolve with including needed dlls in `./pygarment/pattern/cairo_dlls` and adding the ditrectory to PATH in runtime
* [NiceGUI](https://nicegui.io/#installation)
* [trimesh](https://trimesh.org/)
* [libigl](https://libigl.github.io/libigl-python-bindings/)
* [pyrender](https://pyrender.readthedocs.io/en/latest/index.html)
* [Triangle](https://pypi.org/project/triangle/)
* [wgpu-py](https://pypi.org/project/wgpu/)
* [rtree](https://pypi.org/project/rtree/)

All python dependencies can be installed with `pip install` / `conda install`:

```
conda create -n garmentcode python=3.9
conda activate garmentcode
pip install -e . # installs in editable mode, installs requirements from setup.cfg
```

Add the root repository to `PYTHONPATH`.

=> The code is ready to run

> NOTE: check out a full environment setup and running process from our early adopter: https://github.com/maria-korosteleva/GarmentCode/issues/17