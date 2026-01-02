// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/core/Simulator2D.hpp"
#include "FSI_Simulator/lbm/LbmInitializer.hpp"
#include "FSI_Simulator/core/FsiCoupling.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include "FSI_Simulator/utils/Profiler.hpp"
#include "FSI_Simulator/fem/VBDSolver.hpp"
#include "FSI_Simulator/fem/StaticBodySolver.hpp"

namespace fsi
{
    Simulator2D::Simulator2D(const SimulationParameters2D &params)
        : m_params(params)
    {
        cudaStreamCreate(&m_stream);
    }

    void Simulator2D::initialize()
    {
        PROFILE_FUNCTION();
        // 1. initialize LBM flow field
        m_flow_field = std::make_unique<lbm::LbmFlowField2D>(m_params);
        lbm::LbmInitializer::initializeState2D(*m_flow_field, m_params);

        // 2. initialize solid solver
        if (m_params.solid_solver_type == "static")
        {
            m_solid_solver = std::make_unique<fem::StaticBodySolver2D>();
        }
        else if (m_params.solid_solver_type == "vbd")
        {
            m_solid_solver = std::make_unique<fem::vbdSolver2D>(m_stream);
        }
        else
        {
            LOG_CRITICAL("Unknown solid_solver_type: '{}'. Supported types are 'static' and 'vbd'.", m_params.solid_solver_type);
            throw std::runtime_error("Invalid solid_solver_type in SimulationParameters2D.");
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

    void Simulator2D::reset()
    {
        PROFILE_FUNCTION();
        if (m_flow_field)
        {
            m_flow_field->reset();
            lbm::LbmInitializer::initializeState2D(*m_flow_field, m_params);
        }
        if (m_solid_solver)
        {
            m_solid_solver->reset();
        }
    }

    void Simulator2D::step()
    {
        PROFILE_FUNCTION();

        if (m_fluid_solver_enabled)
        {

            // --- 1. LBM + FSI coupling ---
            {
                PROFILE_CUDA_SCOPE("LBM & FSI Coupling", m_stream);
                // prepare solid geometry proxy
                auto solid_proxy = m_solid_solver->getSolidGeometryProxy();
                lbm::fsi_coupling::solveLbmAndFsiStep2D(*m_flow_field, solid_proxy, m_params, m_stream);
            }

            // --- 2. update solid forces ---
            {
                PROFILE_CUDA_SCOPE("Update Solid Forces", m_stream);
                m_solid_solver->applyForces();
                // printf("After applying solid forces\n"); // --- DEBUG ---
            }
        }

        // --- 3. solid solver step ---
        {
            PROFILE_CUDA_SCOPE("Solid Solver", m_stream);
            m_solid_solver->advanceTimeStep();
            // printf("After solid solver step\n"); // --- DEBUG ---
        }

        // {
        //     auto positions = m_solid_solver->getPositions();
        //     auto velocities = m_solid_solver->getVelocities();
        //     vec2_t mean_velocity(0.0f), mean_position(0.0f);
        //     for (int i = 0; i < positions.size(); i++) {
        //         mean_velocity += velocities[i];
        //         mean_position += positions[i];
        //     }
        //     mean_velocity /= velocities.size();
        //     mean_position /= positions.size();
        //     LOG_INFO("Mean velocity: {}, Mean position: {}", mean_velocity, mean_position);
        // }
    }

    std::vector<std::array<int, 2>> Simulator2D::fillSolid()
    {
        PROFILE_FUNCTION();

        auto solid_proxy = m_solid_solver->getSolidGeometryProxy();
        fsi::lbm::fsi_coupling::fillSolidFlags2D(*m_flow_field, solid_proxy, m_params, m_stream);
        auto flags = m_flow_field->getFlags().download();
        std::vector<std::array<int, 2>> solid_indices;
        for (int i = 0; i < flags.size(); i++)
        {
            if (flags[i] == lbm::LbmNodeFlag::SolidDynamic)
            {
                int x = i % m_params.fluid_nx;
                int y = i / m_params.fluid_nx;
                solid_indices.push_back({x, y});
            }
        }

        return solid_indices;
    }

    void Simulator2D::applyLBSControl(int mesh_id, const std::vector<vec2_t> &lbs_shift, const std::vector<real> &lbs_rotation)
    {
        if (m_solid_solver)
        {
            m_solid_solver->applyLBSControl(mesh_id, lbs_shift, lbs_rotation);
        }
        else
        {
            LOG_WARN("Solid solver not initialized. Cannot apply LBS control.");
        }
    }

    void Simulator2D::applyKinematicControl(int mesh_id, const std::vector<vec2_t> &target_pos, real vel)
    {
        if (m_solid_solver)
        {
            m_solid_solver->applyKinematicControl(mesh_id, target_pos, vel);
        }
        else
        {
            LOG_WARN("Solid solver not initialized. Cannot apply kinematic control.");
        }
    }

    void Simulator2D::applyActiveStrain(int mesh_id, int tri_id, vec2_t dir, real magnitude)
    {
        if (m_solid_solver)
        {
            m_solid_solver->applyActiveStrain(mesh_id, tri_id, dir, magnitude);
        }
        else
        {
            LOG_WARN("Solid solver not initialized. Cannot apply active strain.");
        }
    }

    void Simulator2D::saveFrameData(int frame_index) const
    {
        if (m_params.output_frequency <= 0 || m_params.output_path.empty())
        {
            return;
        }
        PROFILE_FUNCTION();
        std::string filepath = m_params.output_path + "/solid_frame_" + std::to_string(frame_index) + ".vtk";
        m_solid_solver->saveFrameData(filepath);
    }

}