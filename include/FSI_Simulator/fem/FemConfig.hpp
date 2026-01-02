// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/common/Types.hpp" // For fsi::real
#include <nlohmann/json.hpp>
#include <string>
#include <optional>

namespace fsi
{

    namespace fem
    {

        // This struct defines the configuration of the FEM solver.
        struct FemSolverOptions
        {
            // "newton" for Newton's method, could be extended to "lbfgs", "gradient_descent" etc.
            std::string optimizer_type = "newton";

            int iterations = 10;

            // --- Line Search Options---
            std::string line_search_method = "backtracking";
            int ls_max_iter = 10; // line search max iterations
            real ls_beta = 0.5;   // step size reduction factor (0 < beta < 1)
            real ls_alpha = 1e-4; // Armijo condition parameter (0 < alpha < 0.5)

            // --- Linear Solver Option ---
            // for solving Hessian system H*p = -g
            // "eigen_ldlt", "cholmod_ldlt", "cuda_qr", "cuda_lu"
            std::string linear_solver_type = "eigen_ldlt";
            real force_density_abs_tol = 1e-3;
            bool grad_check = false;
            int thread_ct = 8;

            // --- VBD Solver Options ---
            int substeps = 3;
            real omega = 0.8;
            int vbd_iterations = 30;
        };

        inline void from_json(const nlohmann::json &j, FemSolverOptions &opts)
        {
            // 使用 .value("key", default_value) 来实现可选字段
            // default_value 就是 opts 对象中已经存在的默认成员值
            opts.optimizer_type = j.value("optimizer_type", opts.optimizer_type);
            opts.iterations = j.value("iterations", opts.iterations);
            opts.line_search_method = j.value("line_search_method", opts.line_search_method);
            opts.ls_max_iter = j.value("ls_max_iter", opts.ls_max_iter);
            opts.ls_beta = j.value("ls_beta", opts.ls_beta);
            opts.ls_alpha = j.value("ls_alpha", opts.ls_alpha);
            opts.linear_solver_type = j.value("linear_solver_type", opts.linear_solver_type);
            opts.substeps = j.value("substeps", opts.substeps);
            opts.omega = j.value("omega", opts.omega);
            opts.vbd_iterations = j.value("vbd_iterations", opts.vbd_iterations);
            opts.force_density_abs_tol = j.value("force_density_abs_tol", opts.force_density_abs_tol);
            opts.grad_check = j.value("grad_check", opts.grad_check);
            opts.thread_ct = j.value("thread_ct", opts.thread_ct);
        }

        // --- 单个可变形体的完整配置 ---
        // 整合了物理属性、几何信息和求解器选项

        struct LBSControlConfig
        {
            int cnum = 2;
            std::string lbs_distance_type = "geodesic";
            bool random_first = false;
            real omega = 0.5;
            real stiffness = 10.0;
        };

        inline void from_json(const nlohmann::json &j, LBSControlConfig &config)
        {
            config.cnum = j.value("cnum", config.cnum);
            config.lbs_distance_type = j.value("lbs_distance_type", config.lbs_distance_type);
            config.random_first = j.value("random_first", config.random_first);
            config.omega = j.value("omega", config.omega);
            config.stiffness = j.value("stiffness", config.stiffness);
        }

        struct SolidBodyConfig2D
        {
            // --- 几何和物理属性 ---
            std::string mesh_path;
            real density = 1000.0;
            real youngs_modulus = 1.0e6;
            real poisson_ratio = 0.45;

            // 初始状态
            std::array<real, 2> translate = {0.0, 0.0};
            real rotate = 0.0;
            std::array<real, 2> scale = {1.0, 1.0};
            std::array<real, 2> initial_velocity = {0.0, 0.0};

            LBSControlConfig lbs_control_config;
        };

        inline void from_json(const nlohmann::json &j, SolidBodyConfig2D &config)
        {
            config.mesh_path = j.value("mesh_path", config.mesh_path);
            config.density = j.value("density", config.density);
            config.youngs_modulus = j.value("youngs_modulus", config.youngs_modulus);
            config.poisson_ratio = j.value("poisson_ratio", config.poisson_ratio);
            config.translate = j.value("translate", config.translate);
            config.rotate = j.value("rotate", config.rotate);
            config.scale = j.value("scale", config.scale);
            config.initial_velocity = j.value("initial_velocity", config.initial_velocity);
            config.lbs_control_config = j.value("lbs_control_config", config.lbs_control_config);
        }

        struct SolidBodyConfig3D
        {
            // --- 几何和物理属性 ---
            std::string mesh_path;
            real density = 1000.0;
            real youngs_modulus = 1.0e6;
            real poisson_ratio = 0.45;

            // 初始状态
            std::array<real, 3> translate = {0.0, 0.0, 0.0};
            std::array<real, 3> rotate = {0.0, 0.0, 0.0};
            std::array<real, 3> scale = {1.0, 1.0, 1.0};
            std::array<real, 3> initial_velocity = {0.0, 0.0, 0.0};

            // LBS控制
            LBSControlConfig lbs_control_config;
        };

        inline void from_json(const nlohmann::json &j, SolidBodyConfig3D &config)
        {
            config.mesh_path = j.value("mesh_path", config.mesh_path);
            config.density = j.value("density", config.density);
            config.youngs_modulus = j.value("youngs_modulus", config.youngs_modulus);
            config.poisson_ratio = j.value("poisson_ratio", config.poisson_ratio);
            config.translate = j.value("translate", config.translate);
            config.rotate = j.value("rotate", config.rotate);
            config.scale = j.value("scale", config.scale);
            config.initial_velocity = j.value("initial_velocity", config.initial_velocity);
            config.lbs_control_config = j.value("lbs_control_config", config.lbs_control_config);
        }

    } // namespace fem

} // namespace fsi