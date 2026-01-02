// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/fem/VBDSolver.hpp"
#include "FSI_Simulator/fem/DeformableMesh.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include "FSI_Simulator/fem/VBDKernels.cuh"
#include "FSI_Simulator/utils/Profiler.hpp"
#include "FSI_Simulator/fsi/FsiCouplingUtils.cuh"
#include "FSI_Simulator/fem/NewtonSolver.hpp"

namespace fsi
{
    namespace fem
    {

        VbdSolver3D::VbdSolver3D(cudaStream_t stream) : m_stream(stream)
        {
            // CUDA_CHECK(cudaStreamCreate(&m_stream));
        }

        void VbdSolver3D::initialize(const SimulationParameters3D &params)
        {
            ISolidSolver3D::initialize(params);

            LOG_INFO("Initializing VbdSolver3D...");

            m_dt = params.dt;
            m_iterations = params.global_fem_options.iterations;
            m_use_newton = params.use_newton;
            m_solver_options = params.global_fem_options;

            m_substeps = params.global_fem_options.substeps;
            m_omega = params.global_fem_options.omega;
            m_iterations = params.global_fem_options.vbd_iterations;
            m_dt = params.dt / m_substeps;

            m_lazy_lbs_control_updated = true;
            m_lazy_active_strain_updated = true;

            LOG_INFO("VbdSolver3D initialized with {} meshes.", m_scene->getNumMeshes());
        }

        void VbdSolver3D::reset()
        {
            ISolidSolver3D::reset();

            m_lazy_lbs_control_updated = true;
            m_lazy_active_strain_updated = true;
        }

        void VbdSolver3D::advanceTimeStep()
        {
            PROFILE_CUDA_SCOPE("VbdSolver3D::advanceTimeStep", m_stream);
            ASSERT(m_scene, "VbdSolver3D must be initialized before advancing time step.");

            if (!m_lazy_lbs_control_updated)
            {
                LBSDynamicCorrection();
                m_scene->updateLBSControl(m_stream);
                m_lazy_lbs_control_updated = true;
            }

            if (!m_lazy_active_strain_updated)
            {
                m_scene->updateActiveStrain(m_stream);
                m_lazy_active_strain_updated = true;
            }

            for (int i = 0; i < m_substeps; ++i)
            {
                // // --- 1. (Advection/Prediction) ---
                predictPositions(
                    m_scene->getGpuData(),
                    m_dt,
                    m_stream);

                // // --- 2. Solve VBD constraints ---
                solveVbdConstraints();

                // // --- 3. Update final positions and velocities ---
                updateVelocitiesAndPositions(
                    m_scene->getGpuData(),
                    m_dt,
                    m_stream);
            }

            // coupling::updateSurfaceStatesFromVolume(
            //     m_scene->getGpuData(),
            //     m_scene->getCouplingData(),
            //     m_scene->getTotalSurfaceVertices(),
            //     m_stream
            // );

            CUDA_CHECK(cudaStreamSynchronize(m_stream));
        }

        void VbdSolver3D::applyForces()
        {
            PROFILE_CUDA_SCOPE("VbdSolver3D::applyForces", m_stream);
            ASSERT(m_scene, "Solver not initialized.");
            m_scene->getGpuData().forces.setZeroAsync(m_stream);
            coupling::scatterSurfaceForcesToVolume(
                m_scene->getGpuData(),
                m_scene->getCouplingData(),
                m_scene->getTotalSurfaceVertices(),
                m_stream);
            m_scene->getCouplingData().surface_forces.setZeroAsync(m_stream);

            cudaStreamSynchronize(m_stream);
            // auto forces = m_scene->getGpuData().forces.download();
            // vec3_t max_force(0.0f);
            // vec3_t min_force(0.0f);
            // vec3_t mean_force(0.0f);

            // auto mass = m_scene->getGpuData().vertex_masses.download();

            // for (int i = 0; i < forces.size(); i++) {
            //     auto force = forces[i] / mass[i];
            //     if (glm::length(force) > glm::length(max_force)) {
            //         max_force = force;
            //     }
            //     if (glm::length(force) < glm::length(min_force)) {
            //         min_force = force;
            //     }
            //     mean_force += force;
            // }
            // mean_force /= forces.size();
            // LOG_INFO("Max acc: {}, Min acc: {}, Mean acc: {}", max_force, min_force, mean_force);
        }

        void VbdSolver3D::applyLBSControl(int mesh_id, const std::vector<vec3_t> &lbs_shift, const std::vector<mat3_t> &lbs_rotation)
        {
            m_scene->lazyApplyLBSControl(mesh_id, lbs_shift, lbs_rotation, m_stream);
            m_lazy_lbs_control_updated = false;
        }

        void VbdSolver3D::applyActiveStrain(int mesh_id, int tet_id, vec3_t dir, real magnitude)
        {
            m_scene->lazyApplyActiveStrain(mesh_id, tet_id, dir, magnitude, m_stream);
            m_lazy_active_strain_updated = false;
        }

        void VbdSolver3D::solveVbdConstraints()
        {
            PROFILE_CUDA_SCOPE("VBD Constraint Solver", m_stream);
            real itr_omega = 2.0 / (2.0 - m_omega * m_omega);
            for (int i = 1; i <= m_iterations; ++i)
            {
                solveTetrahedronConstraints(
                    m_scene->getGpuData(),
                    m_dt,
                    i,
                    itr_omega,
                    m_stream);

                if (i > 1)
                {
                    itr_omega = 4.0 / (4.0 - m_omega * m_omega * itr_omega);
                }
            }
        }

        void VbdSolver3D::LBSDynamicCorrection()
        {
            if (m_use_newton)
            {
                // solveNewtonDynamicCorrectionStep(m_scene->getCpuData(), m_scene->getLBSData(), m_solver_options);
                solveNewtonDynamicCorrectionStepGpu(m_scene->getCpuData(), m_scene->getGpuData(), m_scene->getLBSData(), m_solver_options, m_stream);
            }
            else
            {
                PROFILE_CUDA_SCOPE("LBS Dynamic Correction", m_stream);

                m_scene->getLBSData().position_target.copyFromAsync(m_scene->getLBSData().position_rest, m_stream);

                real itr_omega = 2.0 / (2.0 - m_omega * m_omega);
                for (size_t i = 1; i <= m_iterations * 10; i++)
                {
                    solveLBSDynamicCorrection(m_scene->getGpuData(), m_scene->getLBSData(), i, itr_omega, m_stream);

                    if (i > 1)
                    {
                        itr_omega = 4.0 / (4.0 - m_omega * m_omega * itr_omega);
                    }
                }
            }
        }

        // VBD 2d

        vbdSolver2D::vbdSolver2D(cudaStream_t stream) : m_stream(stream) {}

        void vbdSolver2D::initialize(const SimulationParameters2D &params)
        {
            ISolidSolver2D::initialize(params);

            LOG_INFO("Initializing vbdSolver2D...");

            m_dt = params.dt;
            m_iterations = params.global_fem_options.iterations;
            m_use_newton = params.use_newton;
            m_solver_options = params.global_fem_options;

            m_substeps = params.global_fem_options.substeps;
            m_omega = params.global_fem_options.omega;
            m_iterations = params.global_fem_options.vbd_iterations;
            m_dt = params.dt / m_substeps;

            m_lazy_lbs_control_updated = true;
            m_lazy_active_strain_updated = true;

            LOG_INFO("vbdSolver2D initialized with {} meshes.", m_scene->getNumMeshes());
        }

        void vbdSolver2D::reset()
        {
            ISolidSolver2D::reset();

            m_lazy_lbs_control_updated = true;
            m_lazy_active_strain_updated = true;
        }

        void vbdSolver2D::advanceTimeStep()
        {
            PROFILE_CUDA_SCOPE("vbdSolver2D::advanceTimeStep", m_stream);
            ASSERT(m_scene, "vbdSolver2D must be initialized before advancing time step.");

            if (!m_lazy_lbs_control_updated)
            {
                LBSDynamicCorrection();
                m_scene->updateLBSControl(m_stream);
                m_lazy_lbs_control_updated = true;
            }

            if (!m_lazy_active_strain_updated)
            {
                m_scene->updateActiveStrain(m_stream);
                m_lazy_active_strain_updated = true;
            }

            // auto forces = m_scene->getGpuData().forces.download();
            // vec2_t mean_force(0.0f);
            // for (const auto& force : forces) {
            //     mean_force += force;
            // }
            // mean_force /= forces.size();
            // LOG_INFO("Before Step: Mean force: {}", mean_force);

            for (int i = 0; i < m_substeps; ++i)
            {
                // // --- 1. Advection & Prediction ---
                predictPositions(
                    m_scene->getGpuData(),
                    m_dt,
                    m_stream);

                // // --- 2. Solve VBD constraints ---
                solveVbdConstraints();

                // // --- 3. Update final positions and velocities ---
                updateVelocitiesAndPositions(
                    m_scene->getGpuData(),
                    m_dt,
                    m_stream);
            }

            // // --- 4. Update surface states ---
            // coupling::updateSurfaceStatesFromVolume(
            //     m_scene->getGpuData(),
            //     m_scene->getCouplingData(),
            //     m_scene->getTotalSurfaceVertices(),
            //     m_stream
            // );

            CUDA_CHECK(cudaStreamSynchronize(m_stream));
        }

        void vbdSolver2D::applyForces()
        {
            PROFILE_CUDA_SCOPE("vbdSolver2D::applyForces", m_stream);
            ASSERT(m_scene, "Solver not initialized.");
            m_scene->getGpuData().forces.setZeroAsync(m_stream);
            coupling::scatterSurfaceForcesToVolume(
                m_scene->getGpuData(),
                m_scene->getCouplingData(),
                m_scene->getTotalSurfaceVertices(),
                m_stream);
            m_scene->getCouplingData().surface_forces.setZeroAsync(m_stream);

            // std::vector<vec2_t> forces(m_scene->getTotalVertices(), vec2_t(0.1f, 0.0f));
            // m_scene->getGpuData().forces.uploadAsync(forces, m_stream);

            cudaStreamSynchronize(m_stream);
            // auto forces = m_scene->getGpuData().forces.download();
            // vec2_t max_force(0.0f);
            // vec2_t min_force(0.0f);
            // vec2_t mean_force(0.0f);
            // for (const auto& force : forces) {
            //     if (glm::length(force) > glm::length(max_force)) {
            //         max_force = force;
            //     }
            //     if (glm::length(force) < glm::length(min_force)) {
            //         min_force = force;
            //     }
            //     mean_force += force;
            // }
            // mean_force /= forces.size();
            // LOG_INFO("Max force: {}, Min force: {}, Mean force: {}", max_force, min_force, mean_force);
        }

        void vbdSolver2D::applyLBSControl(int mesh_id, const std::vector<vec2_t> &lbs_shift, const std::vector<real> &lbs_rotation)
        {
            m_scene->lazyApplyLBSControl(mesh_id, lbs_shift, lbs_rotation, m_stream);
            m_lazy_lbs_control_updated = false;
        }

        void vbdSolver2D::applyActiveStrain(int mesh_id, int tri_id, vec2_t dir, real magnitude)
        {
            m_scene->lazyApplyActiveStrain(mesh_id, tri_id, dir, magnitude, m_stream);
            m_lazy_active_strain_updated = false;
        }

        void vbdSolver2D::solveVbdConstraints()
        {
            PROFILE_CUDA_SCOPE("VBD Constraint Solver", m_stream);
            real itr_omega = 2.0 / (2.0 - m_omega * m_omega);
            for (int i = 1; i <= m_iterations; ++i)
            {
                solveConstraints(
                    m_scene->getGpuData(),
                    m_dt,
                    i,
                    itr_omega,
                    m_stream);

                if (i > 1)
                {
                    itr_omega = 4.0 / (4.0 - m_omega * m_omega * itr_omega);
                }
            }
        }

        void vbdSolver2D::LBSDynamicCorrection()
        {
            if (m_use_newton)
            {
                // solveNewtonDynamicCorrectionStep(m_scene->getCpuData(), m_scene->getLBSData(), m_solver_options);
                solveNewtonDynamicCorrectionStepGpu(m_scene->getCpuData(), m_scene->getGpuData(), m_scene->getLBSData(), m_solver_options, m_stream);
            }
            else
            {
                PROFILE_CUDA_SCOPE("LBS Dynamic Correction", m_stream);

                m_scene->getLBSData().position_target.copyFromAsync(m_scene->getLBSData().position_rest, m_stream);

                real itr_omega = 2.0 / (2.0 - m_omega * m_omega);
                for (size_t i = 1; i <= m_iterations * 10; i++)
                {
                    solveLBSDynamicCorrection(m_scene->getGpuData(), m_scene->getLBSData(), i, itr_omega, m_stream);

                    if (i > 1)
                    {
                        itr_omega = 4.0 / (4.0 - m_omega * m_omega * itr_omega);
                    }
                }
            }
        }

    } // namespace fem
} // namespace fsi