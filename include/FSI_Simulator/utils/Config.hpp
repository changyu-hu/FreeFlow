// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <nlohmann/json.hpp>

#include "FSI_Simulator/utils/Logger.hpp"

#include "FSI_Simulator/lbm/LbmDataTypes.cuh"

#include "FSI_Simulator/fem/FemConfig.hpp"

namespace fsi
{
    struct BoundaryCondition
    {
        std::vector<int> position;
        lbm::LbmNodeFlag flag;
    };

    inline void from_json(const nlohmann::json &j, BoundaryCondition &bc)
    {
        j.at("pos").get_to(bc.position);
        std::string flag_str = j.at("type").get<std::string>();

        if (flag_str == "Wall")
            bc.flag = lbm::LbmNodeFlag::Wall;
        else if (flag_str == "InletLeft")
            bc.flag = lbm::LbmNodeFlag::InletLeft;
        else if (flag_str == "InletRight")
            bc.flag = lbm::LbmNodeFlag::InletRight;
        else if (flag_str == "InletUp")
            bc.flag = lbm::LbmNodeFlag::InletUp;
        else if (flag_str == "InletDown")
            bc.flag = lbm::LbmNodeFlag::InletDown;
        else if (flag_str == "InletFront")
            bc.flag = lbm::LbmNodeFlag::InletFront;
        else if (flag_str == "InletBack")
            bc.flag = lbm::LbmNodeFlag::InletBack;
        else if (flag_str == "OutletLeft")
            bc.flag = lbm::LbmNodeFlag::OutletLeft;
        else if (flag_str == "OutletRight")
            bc.flag = lbm::LbmNodeFlag::OutletRight;
        else if (flag_str == "OutletUp")
            bc.flag = lbm::LbmNodeFlag::OutletUp;
        else if (flag_str == "OutletDown")
            bc.flag = lbm::LbmNodeFlag::OutletDown;
        else if (flag_str == "OutletFront")
            bc.flag = lbm::LbmNodeFlag::OutletFront;
        else if (flag_str == "OutletBack")
            bc.flag = lbm::LbmNodeFlag::OutletBack;
        else
            bc.flag = lbm::LbmNodeFlag::Invalid; // or throw an error
    }

    struct SimulationParameters2D
    {
        int dimension = 2;

        // fluid parameters
        double fluid_viscosity = 0.001;
        double fluid_density = 1000.0;
        int fluid_nx = 400;
        int fluid_ny = 400;
        double fluid_dx = 1.0;

        // simulation control
        std::string solid_solver_type = "static"; // "static" or "vbd"
        double total_time = 10.0;
        double dt = 0.001;
        int output_frequency = 100;
        std::string output_path = "./output";

        std::string log_level = "info";          // default log level "info"
        std::string log_file = "simulation.log"; // default "simulation.log"

        bool use_newton = true;

        std::vector<BoundaryCondition> boundaries;
        std::vector<float> boundary_velocities; // [vx_left, vy_down, vx_right, vy_up]

        // if a solid does not define its own options, use this global one
        fem::FemSolverOptions global_fem_options;

        // --- solids config ---
        std::vector<fem::SolidBodyConfig2D> solids;

        float getCf() const
        {
            float cm = fluid_density * fluid_dx * fluid_dx;
            return cm * fluid_dx / (dt * dt);
        }
    };

    inline void from_json(const nlohmann::json &j, SimulationParameters2D &params)
    {
        params.dimension = j.value("dimension", 2);
        params.fluid_viscosity = j.value("fluid_viscosity", 0.001);
        params.fluid_density = j.value("fluid_density", 1000.0);
        params.fluid_nx = j.value("fluid_nx", 400);
        params.fluid_ny = j.value("fluid_ny", 400);
        params.fluid_dx = j.value("fluid_dx", 1.0);
        params.solid_solver_type = j.value("solid_solver_type", "static");
        params.total_time = j.value("total_time", 10.0);
        params.dt = j.value("dt", 0.001);
        params.output_frequency = j.value("output_frequency", 100);
        params.output_path = j.value("output_path", "./output");
        params.log_level = j.value("log_level", "info");
        params.log_file = j.value("log_file", "simulation.log");
        params.use_newton = j.value("use_newton", params.use_newton);
        params.boundaries = j.value("boundaries", std::vector<BoundaryCondition>());
        params.boundary_velocities = j.value("boundary_velocities", std::vector<float>(4, 0.0));
        params.global_fem_options = j.value("global_fem_options", fem::FemSolverOptions());
        params.solids = j.value("solids", std::vector<fem::SolidBodyConfig2D>());
    }

    struct SimulationParameters3D
    {
        int dimension = 3;

        double fluid_viscosity = 0.001;
        double fluid_density = 1000.0;
        int fluid_nx = 128;
        int fluid_ny = 128;
        int fluid_nz = 128;
        double fluid_dx = 1.0;

        std::string solid_solver_type = "static"; // "static" or "vbd"
        double total_time = 10.0;
        double dt = 0.001;
        int output_frequency = 100;
        std::string output_path = "./output";

        std::string log_level = "info";
        std::string log_file = "simulation.log";

        std::vector<BoundaryCondition> boundaries;
        std::vector<float> boundary_velocities; // [vx_left, vy_down, vx_right, vy_up]

        bool use_newton = true;

        fem::FemSolverOptions global_fem_options;

        std::vector<fem::SolidBodyConfig3D> solids;

        float getCf() const
        {
            float cm = fluid_density * fluid_dx * fluid_dx * fluid_dx;
            return cm * fluid_dx / (dt * dt);
        }
    };

    inline void from_json(const nlohmann::json &j, SimulationParameters3D &params)
    {
        try
        {
            params.dimension = j.value("dimension", params.dimension);
            params.fluid_viscosity = j.value("fluid_viscosity", params.fluid_viscosity);
            params.fluid_density = j.value("fluid_density", params.fluid_density);
            params.fluid_nx = j.value("fluid_nx", params.fluid_nx);
            params.fluid_ny = j.value("fluid_ny", params.fluid_ny);
            params.fluid_nz = j.value("fluid_nz", params.fluid_nz);
            params.fluid_dx = j.value("fluid_dx", 1.0);
            params.solid_solver_type = j.value("solid_solver_type", params.solid_solver_type);
            params.total_time = j.value("total_time", params.total_time);
            params.dt = j.value("dt", params.dt);
            params.output_path = j.value("output_path", params.output_path);
            params.log_level = j.value("log_level", params.log_level);
            params.log_file = j.value("log_file", params.log_file);
            params.use_newton = j.value("use_newton", params.use_newton);
            params.boundary_velocities = j.value("boundary_velocities", params.boundary_velocities);

            if (j.contains("global_fem_options") && j.at("global_fem_options").is_object())
            {
                j.at("global_fem_options").get_to(params.global_fem_options);
            }

            if (j.contains("boundaries") && j.at("boundaries").is_array())
            {
                const auto &boundaries_json = j.at("boundaries");
                params.boundaries.reserve(boundaries_json.size());
                for (const auto &bc_json : boundaries_json)
                {
                    if (bc_json.is_object())
                    {
                        params.boundaries.push_back(bc_json.get<BoundaryCondition>());
                    }
                    else
                    {
                        std::cerr << "[JSON PARSE WARN] Item in 'boundaries' array is not a JSON object. Skipping." << std::endl;
                    }
                }
            }

            if (j.contains("solids") && j.at("solids").is_array())
            {
                const auto &solids_json = j.at("solids");
                params.solids.reserve(solids_json.size());
                for (const auto &solid_json : solids_json)
                {
                    if (solid_json.is_object())
                    {
                        params.solids.push_back(solid_json.get<fem::SolidBodyConfig3D>());
                    }
                    else
                    {
                        std::cerr << "[JSON PARSE WARN] Item in 'solids' array is not a JSON object. Skipping." << std::endl;
                    }
                }
            }
        }
        catch (const nlohmann::json::exception &e)
        {
            std::cerr << "[JSON PARSE FATAL] A fatal error occurred during JSON deserialization: " << e.what() << std::endl;
            throw;
        }
    }

    class ConfigBase
    {
    public:
        bool loadBase(const std::string &filepath, nlohmann::json &data);
    };

    class Config2D : public ConfigBase
    {
    public:
        bool load(const std::string &filepath);
        const SimulationParameters2D &getParams() const { return m_params; }

    private:
        void postProcess();
        SimulationParameters2D m_params;
    };

    class Config3D : public ConfigBase
    {
    public:
        bool load(const std::string &filepath);
        const SimulationParameters3D &getParams() const { return m_params; }

    private:
        void postProcess();
        SimulationParameters3D m_params;
    };

} // namespace fsi