// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/core/Simulator3D.hpp"
#include "FSI_Simulator/lbm/LbmInitializer.hpp"
#include "FSI_Simulator/core/FsiCoupling.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include "FSI_Simulator/utils/Profiler.hpp"
#include "FSI_Simulator/fem/VBDSolver.hpp"
#include "FSI_Simulator/fem/StaticBodySolver.hpp"

namespace fsi
{
    Simulator3D::Simulator3D(const SimulationParameters3D &params)
        : m_params(params)
    {
        cudaStreamCreate(&m_stream);
    }

    void Simulator3D::initialize()
    {
        PROFILE_FUNCTION();
        m_flow_field = std::make_unique<lbm::LbmFlowField3D>(m_params);
        lbm::LbmInitializer::initializeState3D(*m_flow_field, m_params);

        if (m_params.solid_solver_type == "static")
        {
            m_solid_solver = std::make_unique<fem::StaticBodySolver3D>();
        }
        else if (m_params.solid_solver_type == "vbd")
        {
            m_solid_solver = std::make_unique<fem::VbdSolver3D>(m_stream);
        }
        else
        {
            LOG_CRITICAL("Unknown solid_solver_type: '{}'. Supported types are 'static' and 'vbd'.", m_params.solid_solver_type);
            throw std::runtime_error("Invalid solid_solver_type in SimulationParameters3D.");
        }
        m_solid_solver->initialize(m_params);

        if (m_params.output_frequency > 0 && !m_params.output_path.empty())
        {
            try
            {
                std::filesystem::path out_path(m_params.output_path);

                if (std::filesystem::create_directories(out_path))
                {
                    LOG_INFO("Output directory created successfully: '{}'", m_params.output_path);
                }
                else
                {
                    LOG_INFO("Output directory already exists: '{}'", m_params.output_path);
                }
            }
            catch (const std::filesystem::filesystem_error &e)
            {
                LOG_CRITICAL("Failed to create or access output directory '{}'. Error: {}", m_params.output_path, e.what());
            }
        }
        else
        {
            LOG_WARN("Output is disabled (output_frequency <= 0 or output_path is empty).");
        }

        LOG_INFO("Simulator initialized successfully.");
    }

    void Simulator3D::reset()
    {
        PROFILE_FUNCTION();
        if (m_flow_field)
        {
            m_flow_field->reset();
            lbm::LbmInitializer::initializeState3D(*m_flow_field, m_params);
        }
        if (m_solid_solver)
        {
            m_solid_solver->reset();
        }
    }

    void Simulator3D::step()
    {
        PROFILE_FUNCTION();

        if (m_fluid_solver_enabled)
        {
            {
                PROFILE_CUDA_SCOPE("LBM & FSI Coupling", m_stream);
                auto solid_proxy = m_solid_solver->getSolidGeometryProxy();
                lbm::fsi_coupling::solveLbmAndFsiStep3D(*m_flow_field, solid_proxy, m_params, m_stream);
            }

            {
                PROFILE_CUDA_SCOPE("Update Solid Forces", m_stream);
                m_solid_solver->applyForces();
            }
        }

        {
            PROFILE_CUDA_SCOPE("Solid Solver", m_stream);
            m_solid_solver->advanceTimeStep();
        }

        // {
        //     auto positions = m_solid_solver->getPositions();
        //     auto velocities = m_solid_solver->getVelocities();
        //     vec3_t mean_velocity(0.0f), mean_position(0.0f);
        //     for (int i = 0; i < positions.size(); i++) {
        //         mean_velocity += velocities[i];
        //         mean_position += positions[i];
        //     }
        //     mean_velocity /= velocities.size();
        //     mean_position /= positions.size();
        //     LOG_INFO("Mean velocity: {}, Mean position: {}", mean_velocity, mean_position);

        // }
    }

    void Simulator3D::applyLBSControl(int mesh_id, const std::vector<vec3_t> &lbs_shift, const std::vector<mat3_t> &lbs_rotation)
    {
        m_solid_solver->applyLBSControl(mesh_id, lbs_shift, lbs_rotation);
    }

    void Simulator3D::applyActiveStrain(int mesh_id, int tet_id, vec3_t dir, real magnitude)
    {
        m_solid_solver->applyActiveStrain(mesh_id, tet_id, dir, magnitude);
    }

    void Simulator3D::applyKinematicControl(int mesh_id, const std::vector<vec3_t> &target_pos, real vel)
    {
        m_solid_solver->applyKinematicControl(mesh_id, target_pos, vel);
    }

    void Simulator3D::saveFrameData(int frame_index, bool save_fluid, bool save_solid) const
    {
        PROFILE_FUNCTION();
        if (save_fluid)
        {
            std::string filepath = m_params.output_path + "/fluid_frame_" + std::to_string(frame_index) + ".vtk";
            m_flow_field->saveFrameData(filepath);
        }

        if (save_solid)
        {
            std::string solid_filepath = m_params.output_path + "/solid_frame_" + std::to_string(frame_index) + ".vtk";
            m_solid_solver->saveFrameData(solid_filepath);
        }
    }

} // namespace fsi