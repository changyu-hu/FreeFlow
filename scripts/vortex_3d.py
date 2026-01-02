# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Changyu Hu

# Commons Clause addition:
# This software is provided for non-commercial use only. See LICENSE file for details.

import matplotlib.cm as cm
import matplotlib
import taichi as ti
from pathlib import Path
import sys
import os
import numpy as np

import fsi_simulator as fsi

NX = 128
NY = 128
NZ = 128

mesh_path = Path(__file__).parent.parent / "assets" / \
    "mesh" / "sphere_fine.mesh"


def make_config(config_path: str, output_path: str):
    config = {
        "dimension": 3,
        "fluid_viscosity": 0.01,
        "fluid_density": 500.0,
        "fluid_nx": NX,
        "fluid_ny": NY,
        "fluid_nz": NZ,
        "fluid_dx": 0.01,
        "solid_solver_type": "static",
        "total_time": 10.0,
        "dt": 5e-3,
        "output_frequency": 200,
        "output_path": output_path,
        "log_level": "info",
        "log_file": "simulation_3d.log",
        "global_fem_options": {
            "optimizer_type": "newton",
            "iterations": 10,
            "verbose_level": 1,
            "line_search_method": "backtracking",
            "ls_max_iter": 10,
            "ls_beta": 0.5,
            "ls_alpha": 1e-4,
            "linear_solver_type": "direct_ldlt",
            "substeps": 3,
            "vbd_iterations": 30,
            "omega": 0.8
        },
        "solids": [],
        "boundaries": [],
        "boundary_velocities": [0.15, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    solid_body = {
        "mesh_path": str(mesh_path),
        "type": 'static',
        "density": 1000.0,
        "youngs_modulus": 1e5,
        "poisson_ratio": 0.45,
        "translate": [0.64, 0.64, 0.64],
        "scale": [0.15, 0.15, 0.15],
    }
    config["solids"].append(solid_body)

    print(config)

    boundaries = []

    nx = config["fluid_nx"]
    ny = config["fluid_ny"]
    nz = config["fluid_nz"]

    for j in range(ny):
        for k in range(nz):
            if j >= ny//4 and j <= 3*ny//4 and k >= nz//4 and k <= 3*nz//4:
                boundaries.append({"type": "InletLeft", "pos": [0, j, k]})
            else:
                boundaries.append({"type": "Wall", "pos": [0, j, k]})
            boundaries.append({"type": "OutletRight", "pos": [nx - 1, j, k]})

    for i in range(nx):
        for k in range(nz):
            boundaries.append({"type": "Wall", "pos": [i, 0, k]})
            boundaries.append({"type": "Wall", "pos": [i, ny - 1, k]})

    for i in range(nx):
        for j in range(ny):
            boundaries.append({"type": "Wall", "pos": [i, j, 0]})
            boundaries.append({"type": "Wall", "pos": [i, j, nz - 1]})

    config["boundaries"] = boundaries

    import json
    try:
        output_dir = os.path.dirname(config_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {config_path}")
        return config
    except IOError as e:
        print(f"Error: Failed to write to file {config_path}. Reason: {e}")
        exit(-1)


def run_test():
    config_path = Path(__file__).parent.parent / \
        "assets" / "configs" / "vortex3d.json"
    output_path = Path(__file__).parent.parent / "output" / "vortex_3d"
    # if not config_path.exists():
    make_config(str(config_path), str(output_path))
    config_loader = fsi.Config3D()

    config_loader.load(str(config_path))
    params = config_loader.get_params()

    simulator = fsi.Simulator3D(params)
    simulator.initialize()

    # simulator.apply_kinematic_control(0, np.array([[0.75, 0.75], [1.0, 0.5], [0.75, 0.25], [0.5, 0.5]]), 0.1)

    gui = ti.GUI("LBM3D", (NX, NY))

    def vis_slice(f, type="magnitude", slice_idx=NZ // 2):
        fMom1 = simulator.get_fluid_moments().transpose(2, 1, 0, 3)
        if type == "magnitude":
            vel = (fMom1[:, :, slice_idx, 1] ** 2 + fMom1[:, :,
                   slice_idx, 2] ** 2 + fMom1[:, :, slice_idx, 3] ** 2) ** 0.5
            vel_img = cm.plasma(vel / 0.3)
        elif type == "vorticity":
            ugrad = np.gradient(fMom1[:, :, 1])
            vgrad = np.gradient(fMom1[:, :, 2])
            vor = ugrad[1] - vgrad[0]
            # vor[flag == lbm.SOLID_DYNAMIC] = 0.02
            colors = [
                (151/255, 139/255, 229/255),
                (255/255, 255/255, 255/255),
                (209/255, 83/255, 124/255),
            ]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "my_cmap", colors)
            vel_img = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(
                vmin=-0.02, vmax=0.02), cmap=my_cmap).to_rgba(vor)

        gui.set_image(vel_img)

        gui.show(str(output_path / f"{f:03d}.png"))

    simulator.begin_profiler("Vortex 3D Simulation")
    for i in range(4000):
        simulator.step()
        if i % 20 == 0:
            vis_slice(i // 20, "magnitude")

    simulator.end_profiler()


if __name__ == "__main__":
    run_test()
