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

NX = 512
NY = 128
NZ = 128

mesh_path = Path(__file__).parent.parent / "assets" / "mesh" / "clownfish.mesh"


def make_config(config_path: str, output_path: str):
    config = {
        "dimension": 3,
        "fluid_viscosity": 0.02,
        "fluid_density": 500.0,
        "fluid_nx": NX,
        "fluid_ny": NY,
        "fluid_nz": NZ,
        "fluid_dx": 0.02,
        "solid_solver_type": "vbd",
        "total_time": 10.0,
        "dt": 5e-3,
        "output_frequency": 200,
        "output_path": output_path,
        "log_level": "info",
        "log_file": "simulation_3d.log",
        "global_fem_options": {
            "optimizer_type": "newton",
            "iterations": 20,
            "verbose_level": 1,
            "line_search_method": "backtracking",
            "force_density_abs_tol": 1e-2,
            "ls_max_iter": 20,
            "ls_beta": 0.3,
            "ls_alpha": 1e-4,
            "linear_solver_type": "cuda_lu",
            "grad_check": False,
            "substeps": 3,
            "vbd_iterations": 30,
            "omega": 0.8
        },
        "solids": [],
        "boundaries": [],
        "boundary_velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    nx = config["fluid_nx"]
    ny = config["fluid_ny"]
    nz = config["fluid_nz"]
    dx = config["fluid_dx"]

    solid_body = {
        "mesh_path": str(mesh_path),
        "type": 'static',
        "density": 1000.0,
        "youngs_modulus": 1e5,
        "poisson_ratio": 0.45,
        "translate": [0.25 * nx * dx, 0.5 * ny * dx, 0.5 * nz * dx],
        "scale": [1.3, 1.3, 1.3],
        "lbs_control_config": {
            "cnum": 3,
            "omega": 0.3,
            "stiffness": 10.0,
        }
    }
    config["solids"].append(solid_body)

    print(config)

    boundaries = []

    for j in range(ny):
        for k in range(nz):
            boundaries.append({"type": "OutletLeft", "pos": [0, j, k]})
            boundaries.append({"type": "OutletRight", "pos": [nx - 1, j, k]})

    for i in range(nx):
        for k in range(nz):
            boundaries.append({"type": "OutletFront", "pos": [i, 0, k]})
            boundaries.append({"type": "OutletBack", "pos": [i, ny - 1, k]})
            # boundaries.append({"type": "Wall", "pos": [i, 0, k]})
            # boundaries.append({"type": "Wall", "pos": [i, ny - 1, k]})

    for i in range(nx):
        for j in range(ny):
            boundaries.append({"type": "OutletDown", "pos": [i, j, 0]})
            boundaries.append({"type": "OutletUp", "pos": [i, j, nz - 1]})
            # boundaries.append({"type": "Wall", "pos": [i, j, 0]})
            # boundaries.append({"type": "Wall", "pos": [i, j, nz - 1]})

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


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                     [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                     [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])


def run_test():
    config_path = Path(__file__).parent.parent / "assets" / \
        "configs" / "fish3d_lbs.json"
    output_path = Path(__file__).parent.parent / "output" / "fish3d_lbs"
    # if not config_path.exists():
    make_config(str(config_path), str(output_path))
    config_loader = fsi.Config3D()

    config_loader.load(str(config_path))
    params = config_loader.get_params()

    simulator = fsi.Simulator3D(params)
    simulator.initialize()

    gui = ti.GUI("LBM3D", (NX, NY))

    def vis_slice(f, type="magnitude", slice_idx=50):
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

    simulator.begin_profiler("3d clownfish actuated by lbs control")
    axis = np.array([0.0, 0.0, 1.0])
    for _ in range(3):
        for i in range(4000):
            if i % 40 == 0:
                pos = np.array(simulator.getVertices())
                x = pos.mean(axis=0)
                print(f"Step {i} mean x {x}")
                angle = np.sin(2 * np.pi * i / 600) * np.pi / 4
                shift = np.zeros((3, 3))
                rotation = np.array(
                    [rotation_matrix(axis, angle), rotation_matrix(axis, -angle), np.eye(3)])
                simulator.apply_lbs_control(0, shift, rotation)
                # print(shift, rotation)
            simulator.step()
        # if i % 20 == 0:
        #     simulator.save_frame_data(i // 20, False, True)
            # vis_slice(i // 20, "magnitude", NZ // 2)
        simulator.reset()
    simulator.end_profiler()


if __name__ == "__main__":
    run_test()
