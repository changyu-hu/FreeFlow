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

NX = 801
NY = 201


def generate_sphere_2d(mesh_path, radius=0.5, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    boundary_points = np.array(
        [[radius * np.cos(t), radius * np.sin(t)] for t in theta])

    center_point = np.array([[0, 0]])
    points = np.vstack([center_point, boundary_points])

    triangles = []
    for i in range(num_points):
        next_i = (i + 1) % num_points
        triangles.append([0, i + 1, next_i + 1])

    triangles = np.array(triangles)

    with open(mesh_path, 'w') as f:
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 2\n\n")

        f.write("Vertices\n")
        f.write(f"{len(points)}\n")
        for i, point in enumerate(points):
            f.write(f"{point[0]:.15f} {point[1]:.15f}\n")
        f.write("\n")

        f.write("Triangles\n")
        f.write(f"{len(triangles)}\n")
        for tri in triangles:
            f.write(f"{tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

    return points, triangles


def make_config(config_path: str, output_path: str):
    config = {
        "dimension": 2,
        "fluid_viscosity": 0.01,
        "fluid_density": 1000.0,
        "fluid_nx": NX,
        "fluid_ny": NY,
        "fluid_dx": 0.005,
        "solid_solver_type": "static",
        "total_time": 10.0,
        "dt": 2.5e-3,
        "output_frequency": 200,
        "output_path": output_path,
        "log_level": "info",
        "log_file": "simulation_2d.log",
        "global_fem_options": {
            "optimizer_type": "newton",
            "iterations": 10,
            "verbose_level": 1,
            "line_search_method": "backtracking",
            "ls_max_iter": 10,
            "ls_beta": 0.5,
            "ls_alpha": 1e-4,
            "linear_solver_type": "direct_ldlt"
        },
        "solids": [],
        "boundaries": [],
        "boundary_velocities": [0.2, 0.0, 0.0, 0.0]
    }

    from pathlib import Path
    mesh_path = Path(__file__).parent.parent / \
        "assets" / "mesh" / "sphere2d.mesh"

    generate_sphere_2d(mesh_path, radius=0.5, num_points=100)

    solid_body = {
        "mesh_path": str(mesh_path),
        "type": 'static',
        "translate": [0.5, 0.5],
        "scale": [0.15, 0.15],
    }
    config["solids"].append(solid_body)

    print(config)

    boundaries = []

    nx = config["fluid_nx"]
    ny = config["fluid_ny"]

    for i in range(nx):
        # boundaries.append({"type": "OutletDown", "pos": [i, 0]})
        # boundaries.append({"type": "OutletUp", "pos": [i, ny - 1]})
        boundaries.append({"type": "Wall", "pos": [i, 0]})
        boundaries.append({"type": "Wall", "pos": [i, ny - 1]})

    for j in range(ny):
        # if j >= ny // 4 and j <= 3 * ny // 4:
        boundaries.append({"type": "InletLeft", "pos": [0, j]})
        # else:
        #     boundaries.append({"type": "Wall", "pos": [0, j]})

        boundaries.append({"type": "OutletRight", "pos": [nx - 1, j]})

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
    config_path = Path(__file__).parent.parent / "assets" / \
        "configs" / "vortexstreet2d.json"
    output_path = Path(__file__).parent.parent / "output" / "vortex_street_2d"
    # if not config_path.exists():
    make_config(str(config_path), str(output_path))
    config_loader = fsi.Config2D()

    config_loader.load(str(config_path))
    params = config_loader.get_params()

    simulator = fsi.Simulator2D(params)
    simulator.initialize()

    simulator.apply_kinematic_control(0, np.array(
        [[0.75, 0.75], [1.0, 0.5], [0.75, 0.25], [0.5, 0.5]]), 0.1)

    gui = ti.GUI("LBM2D", (NX, NY))

    def vis(f, type="magnitude"):
        fMom1 = simulator.get_fluid_moments().transpose(1, 0, 2)
        flags = np.array(simulator.fillSolid(), dtype=np.int32)
        if type == "magnitude":
            vel = (fMom1[:, :, 1] ** 2 + fMom1[:, :, 2] ** 2) ** 0.5
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

        vel_img[flags[:, 0], flags[:, 1]] = np.array(
            [162/255, 153/255, 161/255, 1.0])
        gui.set_image(vel_img)

        gui.show(str(output_path / f"{f:03d}.png"))

    for i in range(10000):
        simulator.step()
        if i % 50 == 0:
            vis(i // 50, "vorticity")


if __name__ == "__main__":
    run_test()
