# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Changyu Hu

# Commons Clause addition:
# This software is provided for non-commercial use only. See LICENSE file for details.

import matplotlib.cm as cm
from time import sleep
import time
import matplotlib
import taichi as ti
from pathlib import Path
import sys
import os
import numpy as np

import fsi_simulator as fsi

NX = 1200
NY = 400
T = 1.0
act_alpha = 0.15
mesh_path = Path(__file__).parent.parent / "assets" / "mesh" / "fish2d.mesh"
ball_path = Path(__file__).parent.parent / "assets" / "mesh" / "sphere2d.mesh"


def make_config(config_path: str, output_path: str):
    config = {
        "dimension": 2,
        "fluid_viscosity": 0.005,
        "fluid_density": 10.0,
        "fluid_nx": NX,
        "fluid_ny": NY,
        "fluid_dx": 0.005,
        "solid_solver_type": "vbd",
        "total_time": 10.0,
        "dt": 2.5e-3,
        "output_frequency": 200,
        "output_path": output_path,
        "log_level": "info",
        "log_file": "simulation_2d.log",
        "global_fem_options": {
            "optimizer_type": "newton",
            "iterations": 50,
            "verbose_level": 1,
            "line_search_method": "backtracking",
            "force_density_abs_tol": 1e-2,
            "ls_max_iter": 10,
            "ls_beta": 0.3,
            "ls_alpha": 1e-4,
            "linear_solver_type": "eigen_ldlt",
            "grad_check": False,
            "substeps": 3,
            "vbd_iterations": 30,
            "omega": 0.8
        },
        "solids": [],
        "boundaries": [],
        "boundary_velocities": [0.0, 0.0, 0.0, 0.0]
    }

    solid_body = {
        "mesh_path": str(mesh_path),
        "density": 10.0,
        "youngs_modulus": 1e4,
        "poisson_ratio": 0.4,
        "translate": [1.0, 1.0],
        "rotate": 180,
        "scale": [0.8, 0.7],
        "lbs_control_config": {
            "cnum": 2,
            "omega": 0.3,
            "stiffness": 0.1,
        }
    }
    config["solids"].append(solid_body)

    print(config)

    boundaries = []

    nx = config["fluid_nx"]
    ny = config["fluid_ny"]

    for i in range(nx):
        boundaries.append({"type": "OutletDown", "pos": [i, 0]})
        boundaries.append({"type": "OutletUp", "pos": [i, ny - 1]})

    for j in range(ny):
        boundaries.append({"type": "OutletLeft", "pos": [0, j]})
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
        "configs" / "fish2d_lbs_control.json"
    output_path = Path(__file__).parent.parent / \
        "output" / "fish2d_lbs_control"
    # if not config_path.exists():
    make_config(str(config_path), str(output_path))
    config_loader = fsi.Config2D()

    config_loader.load(str(config_path))
    params = config_loader.get_params()

    simulator = fsi.Simulator2D(params)
    simulator.initialize()

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

    simulator.begin_profiler("2d fish actuated by lbs control")
    for _ in range(5):
        for i in range(4000):
            if i % 40 == 0:
                angle = np.sin(2 * np.pi * i / 600) * np.pi * 0.5
                shift = np.zeros((2, 2))
                rotation = np.array([angle, -angle])
                # print("Step {}, angle = {:.3f}".format(i, angle))
                simulator.apply_lbs_control(0, shift, rotation)
            simulator.step()
            if i % 20 == 0:
                vis(i // 20, "vorticity")
                # sleep(0.01)
        simulator.reset()
    simulator.end_profiler()


if __name__ == "__main__":
    run_test()
