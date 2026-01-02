// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/fem/StaticBodySolver.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include "FSI_Simulator/fem/StaticBodyKernels.cuh"

namespace fsi
{
    namespace fem
    {

        void StaticBodySolver3D::initialize(const SimulationParameters3D &params)
        {
            ISolidSolver3D::initialize(params);

            LOG_INFO("Initializing StaticBodySolver3D...");

            m_dt = params.dt;
            target_pos = {};
            vel = 0.0f;
            current_target_index = -1;

            LOG_INFO("StaticBodySolver3D initialized with {} meshes.", m_scene->getNumMeshes());
        }

        void StaticBodySolver3D::reset()
        {
            ISolidSolver3D::reset();

            target_pos.clear();
            vel = 0.0f;
            current_target_index = -1;
        }

        void StaticBodySolver3D::advanceTimeStep()
        {
            if (target_pos.empty() || vel <= 0.0f)
            {
                return;
            }

            auto positions = m_scene->getGpuData().positions.download();
            vec3_t current_target, current_marker;
            current_marker = positions[0];

            if (current_target_index < 0)
            {
                current_target_index = 0;
            }

            current_target = target_pos[current_target_index];

            if (glm::length(current_target - current_marker) < m_dt * vel)
            {
                current_target_index = (current_target_index + 1) % target_pos.size();
                current_target = target_pos[current_target_index];
            }

            auto dir = glm::normalize(current_target - current_marker);
            auto current_vel = vel * dir;
            auto shift = current_vel * m_dt;
            set_vel(m_scene->getGpuData(), current_vel);
            add_pos(m_scene->getGpuData(), shift);
        }

        void StaticBodySolver3D::applyForces()
        {
            LOG_TRACE("Ignoring received forces for static body.");
        }

        void StaticBodySolver3D::applyKinematicControl(int mesh_id, const std::vector<vec3_t> &target_pos, real vel)
        {
            if (mesh_id < 0 || mesh_id >= m_scene->getNumMeshes())
            {
                LOG_WARN("Invalid mesh_id {} for kinematic control.", mesh_id);
                return;
            }
            if (target_pos.size() < 2)
            {
                LOG_WARN("At least two target positions are required for kinematic control.");
                return;
            }
            if (vel <= 0.0f)
            {
                LOG_WARN("Velocity must be positive for kinematic control.");
                return;
            }

            this->target_pos = target_pos;
            this->vel = vel;
            this->current_target_index = -1;
            LOG_INFO("Kinematic control applied to mesh {} with {} target positions and velocity {}.", mesh_id, target_pos.size(), vel);
        }

        void StaticBodySolver2D::initialize(const SimulationParameters2D &params)
        {
            LOG_INFO("Initializing StaticBodySolver2D...");

            ISolidSolver2D::initialize(params);

            m_dt = params.dt;
            target_pos = {};
            vel = 0.0f;
            current_target_index = -1;

            LOG_INFO("StaticBodySolver2D initialized with {} meshes.", m_scene->getNumMeshes());
        }

        void StaticBodySolver2D::reset()
        {
            ISolidSolver2D::reset();

            target_pos.clear();
            vel = 0.0f;
            current_target_index = -1;
        }

        void StaticBodySolver2D::advanceTimeStep()
        {
            if (target_pos.empty() || vel <= 0.0f)
            {
                return;
            }

            auto positions = m_scene->getGpuData().positions.download();
            vec2_t current_target, current_marker;
            current_marker = positions[0];

            if (current_target_index < 0)
            {
                current_target_index = 0;
            }

            current_target = target_pos[current_target_index];

            if (glm::length(current_target - current_marker) < m_dt * vel)
            {
                current_target_index = (current_target_index + 1) % target_pos.size();
                current_target = target_pos[current_target_index];
            }

            auto dir = glm::normalize(current_target - current_marker);
            auto current_vel = vel * dir;
            auto shift = current_vel * m_dt;
            set_vel(m_scene->getGpuData(), current_vel);
            add_pos(m_scene->getGpuData(), shift);
        }

        void StaticBodySolver2D::applyForces()
        {
            LOG_TRACE("Ignoring received forces for static body.");
        }

        void StaticBodySolver2D::applyKinematicControl(int mesh_id, const std::vector<vec2_t> &target_pos, real vel)
        {
            if (mesh_id < 0 || mesh_id >= m_scene->getNumMeshes())
            {
                LOG_WARN("Invalid mesh_id {} for kinematic control.", mesh_id);
                return;
            }
            if (target_pos.size() < 2)
            {
                LOG_WARN("At least two target positions are required for kinematic control.");
                return;
            }
            if (vel <= 0.0f)
            {
                LOG_WARN("Velocity must be positive for kinematic control.");
                return;
            }

            this->target_pos = target_pos;
            this->vel = vel;
            this->current_target_index = -1;
            LOG_INFO("Kinematic control applied to mesh {} with {} target positions and velocity {}.", mesh_id, target_pos.size(), vel);
        }

    } // namespace fem
} // namespace fsi