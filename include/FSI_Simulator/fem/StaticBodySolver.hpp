// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/fem/ISolidSolver.hpp"
#include "FSI_Simulator/utils/Config.hpp"
#include <memory>
#include <vector>

namespace fsi
{
    namespace fem
    {
        /**
         * @brief represents a static body that can be kinematically controlled.
         */
        class StaticBodySolver3D : public ISolidSolver3D
        {
        public:
            StaticBodySolver3D() = default;
            ~StaticBodySolver3D() override = default;

            void initialize(const SimulationParameters3D &params) override;
            void reset() override;
            void advanceTimeStep() override;
            void applyForces() override;
            void applyKinematicControl(int mesh_id, const std::vector<vec3_t> &target_pos, real vel) override;

        private:
            real m_dt;
            std::vector<vec3_t> target_pos;
            int current_target_index;
            real vel;
        };

        class StaticBodySolver2D : public ISolidSolver2D
        {
        public:
            StaticBodySolver2D() = default;
            ~StaticBodySolver2D() override = default;

            void initialize(const SimulationParameters2D &params) override;
            void reset() override;
            void advanceTimeStep() override;
            void applyForces() override;
            void applyKinematicControl(int mesh_id, const std::vector<vec2_t> &target_pos, real vel) override;

        private:
            real m_dt;
            std::vector<vec2_t> target_pos;
            int current_target_index;
            real vel;
        };

    } // namespace fem
} // namespace fsi