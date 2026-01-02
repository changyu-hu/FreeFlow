# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Changyu Hu

# Commons Clause addition:
# This software is provided for non-commercial use only. See LICENSE file for details.

import json
from utils import create_folder, to_integer_array, to_real_array, eulerAnglesToRotationMatrix3D
import taichi as ti
import numpy as np
import matplotlib.cm as cm
import matplotlib
from pathlib import Path
import sys
from abc import ABC, abstractmethod
import random

from filelock import FileLock

import fsi_simulator as fsi


class BaseEnv(ABC):
    def __init__(self, cfg_path):
        self.cfg = json.load(open(cfg_path))
        self.dim = self.cfg["dim"]
        self.model_name = self.cfg["model_name"]
        self.experiment_name = self.cfg["experiment_name"]
        self.task_type = self.cfg["task_type"]
        self.data_dir = Path(__file__).parent.parent / "output" / \
            self.experiment_name / self.model_name
        create_folder(self.data_dir, exist_ok=True)

        self.config_path = self.data_dir / "config.json"
        self.action_size = self.cfg["action_size"]
        # if not self.config_path.exists():
        lock_path = str(self.config_path) + ".lock"
        lock = FileLock(lock_path)
        with lock:
            self.make_config()
        # else:
        #     self.config = json.load(open(self.config_path))
        #     self.nx = self.config["fluid_nx"]
        #     self.ny = self.config["fluid_ny"]
        #     self.dx = self.config["fluid_dx"]
        #     self.dt = self.config["dt"]

        if self.dim == 2:
            config_loader = fsi.Config2D()

            config_loader.load(str(self.config_path))
            self.params = config_loader.get_params()
            self.simulator = fsi.Simulator2D(self.params)
            self.simulator.initialize()
        elif self.dim == 3:
            config_loader = fsi.Config3D()
            config_loader.load(str(self.config_path))
            self.params = config_loader.get_params()
            self.simulator = fsi.Simulator3D(self.params)
            self.simulator.initialize()
            self.nz = self.config["fluid_nz"]
        else:
            print("Dimension should be 2 or 3.")
            sys.exit(1)

        self.r_smooth = self.cfg["r_smooth"]
        self.r_reg = self.cfg["r_reg"]
        self.r_energy = self.cfg["r_energy"]
        self.interval = self.cfg["interval"]
        self.itrnum_per_step = int(self.interval / self.dt)

        self.action_range = np.zeros(self.action_size)

        self.sample_points_idx = to_integer_array(
            self.simulator.getControlPointIdx())
        pos = to_real_array(self.simulator.getVertices())
        self.sample_points_origin = pos[self.sample_points_idx, :]
        self.init_diagnal = np.linalg.norm(
            np.max(pos, axis=0) - np.min(pos, axis=0))
        self.center_origin = np.mean(self.sample_points_origin, axis=0)
        self.sample_points_origin -= self.center_origin[np.newaxis, :]

        self.generate_target()
        self.distance_to_target = np.linalg.norm(
            self.target - self.center_origin)
        self.last_actions = np.zeros(self.action_size)
        self.numofframe = 0
        self.diverge_punish = -0.2

    def make_config(self):
        self.config = json.load(open(self.cfg["config_template_path"]))
        self.config["output_path"] = str(self.data_dir / "render_data")

        mesh_path = str(Path(__file__).parent.parent /
                        "assets" / "mesh" / f"{self.model_name}.mesh")
        if not Path(mesh_path).exists():
            print(f"Mesh file {mesh_path} does not exist.")
            sys.exit(1)

        self.nx = self.config["fluid_nx"]
        self.ny = self.config["fluid_ny"]
        self.dx = self.config["fluid_dx"]
        self.dt = self.config["dt"]
        if self.task_type == "flow_resistance":
            translate = [self.nx * self.dx * 0.6, self.ny * self.dx * 0.5]
        else:
            translate = [self.nx * self.dx * 0.2, self.ny * self.dx * 0.5]
        if self.dim == 3:
            self.nz = self.config["fluid_nz"]
            translate.append(self.nz * self.dx * 0.5)

        self.config["solids"][0]["mesh_path"] = mesh_path
        self.config["solids"][0]["translate"] = translate
        self.config["solids"][0]["scale"] = self.cfg["scale"]
        self.config["solids"][0]["rotate"] = self.cfg["rotate"]

        if self.task_type == "flow_resistance":
            vin = self.cfg.get("vin", 0.1)
            self.config["boundary_velocities"] = [vin] * self.dim * 2
            for bc in self.config["boundaries"]:
                if bc["type"] == "OutletRight":
                    bc["type"] = "InletRight"

        # customize parameters based on the task and control methods.
        self.transfer_params()

        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    @abstractmethod
    def transfer_params(self):
        pass

    def reset(self):
        self.simulator.reset()
        self.distance_to_target = np.linalg.norm(
            self.target - self.center_origin)
        self.last_actions = np.zeros(self.action_size)
        self.numofframe = 0
        state, _ = self.get_state()
        return state

    def generate_target(self):
        if self.task_type == "forward":
            if self.dim == 2:
                self.target = np.array(
                    [self.nx * self.dx * 0.9, self.ny * self.dx / 2])
            else:
                self.target = np.array(
                    [self.nx * self.dx * 0.9, self.ny * self.dx / 2, self.nz * self.dx / 2])
        elif self.task_type == "chasing":
            l = random.uniform(1.5, 2.0)
            theta = random.uniform(-np.pi / 8, np.pi / 8)
            self.target = self.center_origin + l * \
                np.array([np.cos(theta), np.sin(theta)])
        elif self.task_type == "flow_resistance":
            self.target = self.center_origin

    @abstractmethod
    def process_action(self, action):
        pass

    def step(self, action):
        self.process_action(action)

        self.step_energy = 0.0
        for i in range(self.itrnum_per_step):
            self.simulator.step()
            velocity = to_real_array(self.simulator.getVelocity())
            max_vel = np.max(np.linalg.norm(velocity, axis=1))
            if max_vel > 10.0:
                print("Velocity is too large.")
                return None, self.diverge_punish, 2, None

            force = to_real_array(self.simulator.getForce())
            power = np.sum(-force * velocity)
            if power > 0.0:
                self.step_energy += power * self.dt

        state, done = self.get_state()
        if done == 2:
            return state, self.diverge_punish, 2, None

        reward, info = self.get_reward()
        self.last_actions = self.actions.copy()

        return state, reward, done, info

    def get_state(self):
        # check state.
        done = 0

        pos = to_real_array(self.simulator.getVertices())
        vel = to_real_array(self.simulator.getVelocity())
        mean_vel = np.mean(vel, axis=0)
        state_raw = np.concatenate(
            (pos[self.sample_points_idx, :], vel[self.sample_points_idx, :]), axis=1)

        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)) or np.any(np.isnan(vel)) or np.any(np.isinf(vel)):
            print("state is nan or inf.")
            state = np.zeros_like(state_raw)
            done = 2
            return state, done

        diagnal = np.linalg.norm(
            np.max(pos, axis=0) - np.min(pos, axis=0))
        if diagnal > self.init_diagnal * 3. or np.isnan(diagnal):
            print("Exploded!")
            done = 2
            return state, done

        # check if outside the boundary.
        if np.any(state_raw[:, 0] < 0) or np.any(state_raw[:, 0] >= self.nx * self.dx) \
                or np.any(state_raw[:, 1] < 0) or np.any(state_raw[:, 1] >= self.ny * self.dx) \
                or (self.dim == 3 and (np.any(state_raw[:, 2] < 0) or np.any(state_raw[:, 2] >= self.nz * self.dx))):
            print("object is outside the boundary.")
            done = 2

        # icp.
        sample_points_deformed = state_raw[:, :self.dim]
        center_deformed = np.mean(sample_points_deformed, axis=0)
        H = self.sample_points_origin.T @ (
            sample_points_deformed - center_deformed[np.newaxis, :])
        U, S, Vh = np.linalg.svd(H)
        self.R = Vh.T @ U.T
        self.t = center_deformed
        state_pos_local = (sample_points_deformed -
                           self.t[np.newaxis, :]) @ self.R
        state_vel_local = state_raw[:, self.dim:] @ self.R
        mean_vel_local = mean_vel @ self.R
        state_local = np.concatenate(
            (state_pos_local, state_vel_local), axis=1).reshape(-1)

        self.fish_vel_local = np.mean(state_vel_local, axis=0)

        local_target = (self.target - self.t) @ self.R
        self.target_ditance = np.linalg.norm(local_target)
        if self.target_ditance > 0.0:
            self.local_target_direction = local_target / self.target_ditance
        else:
            self.local_target_direction = np.zeros(self.dim)
        target_distance_exp = np.exp(- self.target_ditance / 1.0)

        # check if the fish reached the target.
        if self.task_type == "forward":
            if self.target_ditance < 0.1:
                print("fish reached the target.")
                done = 1
        if self.task_type == "chasing":
            if self.target_ditance < 0.1:
                print("fish reached the target.")
                done = 1
            elif self.fish_vel_local[0] < 0.0 and np.linalg.norm(self.fish_vel_local) > 0.2:
                print("fish is passing the target.")
                done = 2
        elif self.task_type == "flow_resistance":
            if self.target_ditance > 0.4 and self.fish_vel_local[0] < 0.0:
                print("fish is far away from the target.")
                done = 2

        state = np.concatenate((
            state_local,
            mean_vel_local,
            self.local_target_direction,
            [target_distance_exp],
            self.last_actions
        ), axis=0)

        return state, done

    def get_reward(self):
        distance_to_target = np.linalg.norm(self.target - self.t)
        fish_vel_towards_target = (
            self.distance_to_target - distance_to_target) / self.interval
        action_distance2 = np.mean((self.actions - self.last_actions) ** 2)
        action_squared_mean = np.mean(self.actions ** 2)
        reward = 1.0 * fish_vel_towards_target \
            - self.r_smooth * action_distance2 \
            - self.r_reg * action_squared_mean \
            - self.r_energy * self.step_energy

        alive_bonus = 0.01
        if self.task_type == "flow_resistance":
            reward += alive_bonus

        self.distance_to_target = distance_to_target

        info = {
            "fish_vel_towards_target": fish_vel_towards_target,
            "action_distance2": action_distance2,
            "action_squared_mean": action_squared_mean,
            "distance_to_target": self.distance_to_target,
            "step_energy": self.step_energy
        }

        # print("reward: ", fish_vel_towards_target, "reward_0: ", self.reward_1 * action_distance2, "reward_1: ", self.reward_2 * action_squared_mean)
        return reward, info

    def vis2d(self, f, type="magnitude"):
        fMom1 = self.simulator.get_fluid_moments().transpose(1, 0, 2)
        flags = to_integer_array(self.simulator.fillSolid())
        if type == "magnitude":
            vel = (fMom1[:, :, 1] ** 2 + fMom1[:, :, 2] ** 2) ** 0.5
            colors = [
                (255/255, 255/255, 255/255),
                (151/255, 139/255, 229/255),
            ]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "my_cmap", colors)
            vel_img = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(
                vmin=0, vmax=0.1), cmap=my_cmap).to_rgba(vel)
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
        if not hasattr(self, "gui"):
            self.gui = ti.GUI(f"Fluid Simulation", (self.nx, self.ny))

        self.gui.set_image(vel_img)

        if self.task_type == "chasing":
            self.gui.circle(pos=[self.target[0]/self.nx/self.dx,
                            self.target[1]/self.ny/self.dx], radius=20, color=0xED553B)

        self.gui.show(str(self.data_dir / "render_data" / f"{f:03d}.png"))

    def render(self):
        if self.dim == 3:
            self.simulator.save_frame_data(
                self.numofframe, self.cfg["save_fluid"], self.cfg["save_solid"])
        else:
            self.vis2d(self.numofframe, type="vorticity")
        self.numofframe += 1

    @staticmethod
    @abstractmethod
    def get_state_dim(action_dim):
        pass


class LBSEnv(BaseEnv):
    def __init__(self, cfg_path):
        super().__init__(cfg_path)
        self.action_range = np.zeros(self.action_size)
        ranges = self.cfg.get("action_range", {
            "translate": 0.2,
            "rotate": 0.2,
        })
        for i in range(self.action_size):
            if i % (2 * self.dim) < self.dim:
                self.action_range[i] = ranges["translate"]
            else:
                self.action_range[i] = ranges["rotate"]

        self.shift = np.zeros((self.cnum, self.dim))
        self.rotation = np.zeros(
            (self.cnum, 3, 3)) if self.dim == 3 else np.zeros((self.cnum))

    def transfer_params(self):
        if self.dim == 2:
            assert self.action_size % 3 == 0, "action_size should be divisible by 3 for 2D LBS."
            self.cnum = self.action_size // 3
        elif self.dim == 3:
            assert self.action_size % 6 == 0, "action_size should be divisible by 6 for 3D LBS."
            self.cnum = self.action_size // 6

        self.config["solids"][0]["lbs_control_config"]["cnum"] = self.cnum
        self.config["solids"][0]["lbs_control_config"]["omega"] = self.cfg.get(
            "omega", self.config["solids"][0]["lbs_control_config"]["omega"])
        self.config["solids"][0]["lbs_control_config"]["stiffness"] = self.cfg.get(
            "stiffness", self.config["solids"][0]["lbs_control_config"]["stiffness"])

    def process_action(self, action):
        self.actions = action.copy() * self.action_range
        for i in range(self.cnum):
            if self.dim == 2:
                self.shift[i] = self.actions[3*i:3*i+2]
                self.rotation[i] = self.actions[3*i+2]
            else:
                self.shift[i] = self.actions[6*i:6*i+3]
                self.rotation[i] = eulerAnglesToRotationMatrix3D(
                    self.actions[6*i+3:6*(i+1)])

        self.simulator.apply_lbs_control(0, self.shift, self.rotation)

    @staticmethod
    def get_state_dim(dim, action_dim):
        if dim == 2:
            cnum = action_dim // 3
            return 4 * cnum + 2 + 2 + 1 + action_dim
        elif dim == 3:
            cnum = action_dim // 6
            return 6 * cnum + 3 + 3 + 1 + action_dim


class KMeansEnv(BaseEnv):
    def __init__(self, cfg_path):
        super().__init__(cfg_path)
        act_range = self.cfg.get("action_range", 0.1)
        self.action_range = np.ones(self.action_size) * act_range

        if self.dim == 2:
            eles = to_integer_array(
                self.simulator.getTriangles()).reshape(-1, 3)
        elif self.dim == 3:
            eles = to_integer_array(
                self.simulator.getTetrahedra()).reshape(-1, 4)
        eles_centers = np.mean(to_real_array(
            self.simulator.getVertices())[eles], axis=1)
        self.ele_num = eles.shape[0]

        from sklearn.cluster import KMeans
        # k means clustering.
        kmeans = KMeans(n_clusters=self.kmeans_num,
                        random_state=4).fit(eles_centers)
        self.kmeans_labels = kmeans.labels_
        # pca for each cluster.
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.dim)
        centers_of_clusters = []
        pca_of_clusters = []
        act_dirs = []

        for i in range(self.knum):
            centers_i = eles_centers[self.kmeans_labels == i]
            pca.fit_transform(centers_i)
            centers_of_clusters.append(centers_i)
            pca_of_clusters.append(pca.components_)
            act_dir = pca.components_[0]
            act_dirs.append(act_dir)

    def transfer_params(self):
        self.knum = self.action_size
        # just for farthest point sampling
        self.config["solids"][0]["lbs_control_config"]["cnum"] = self.knum

    def process_action(self, action):
        self.actions = action.copy() * self.action_range + self.action_range
        for i in range(self.ele_num):
            act_dir = self.act_dirs[self.kmeans_labels[i]]
            action = self.actions[self.kmeans_labels[i]]
            self.simulator.apply_active_strain(0, i, act_dir, 1 - action)

    @staticmethod
    def get_state_dim(dim, action_dim):
        if dim == 2:
            return 4 * action_dim + 2 + 2 + 1 + action_dim
        elif dim == 3:
            return 6 * action_dim + 3 + 3 + 1 + action_dim
