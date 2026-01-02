// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/fem/ISolidSolver.hpp"
#include "FSI_Simulator/fem/FemScene.hpp"
#include "FSI_Simulator/utils/Config.hpp" // For SimulationParameters3D
#include <memory>
#include <cuda_runtime.h> // For cudaStream_t

namespace fsi
{
    namespace fem
    {

        class VbdSolver3D : public ISolidSolver3D
        {
        public:
            VbdSolver3D(cudaStream_t stream);

            void initialize(const SimulationParameters3D &params) override;

            void reset() override;

            virtual std::vector<vec3_t> getLBSVertices() const override
            {
                ASSERT(false, "Not implemented");
                return m_scene->getLBSData().position_target.download();
            }

            void applyLBSControl(int mesh_id, const std::vector<vec3_t> &lbs_shift, const std::vector<mat3_t> &lbs_rotation) override;

            void applyActiveStrain(int mesh_id, int tet_id, vec3_t dir, real magnitude) override;

            void advanceTimeStep() override;

            void applyForces() override;

        private:
            void solveVbdConstraints();

            void LBSDynamicCorrection();

            real m_dt;
            int m_iterations;
            int m_substeps;
            real m_omega;
            bool m_use_newton;
            FemSolverOptions m_solver_options;

            bool m_lazy_lbs_control_updated;

            bool m_lazy_active_strain_updated;

        public:
            cudaStream_t m_stream;
        };

        class vbdSolver2D : public ISolidSolver2D
        {
        public:
            vbdSolver2D(cudaStream_t stream);

            void initialize(const SimulationParameters2D &params) override;
            void reset() override;

            virtual std::vector<vec2_t> getLBSVertices() const override
            {
                ASSERT(false, "Not implemented");
                return m_scene->getLBSData().position_target.download();
            }

            void applyLBSControl(int mesh_id, const std::vector<vec2_t> &lbs_shift, const std::vector<real> &lbs_rotation) override;
            void applyActiveStrain(int mesh_id, int tri_id, vec2_t dir, real magnitude) override;
            void advanceTimeStep() override;
            void applyForces() override;

        private:
            void solveVbdConstraints();
            void LBSDynamicCorrection();
            real m_dt;
            int m_iterations;
            int m_substeps;
            real m_omega;
            bool m_use_newton;
            FemSolverOptions m_solver_options;
            bool m_lazy_lbs_control_updated;
            bool m_lazy_active_strain_updated;

        public:
            cudaStream_t m_stream;
        };

    } // namespace fem
} // namespace fsi