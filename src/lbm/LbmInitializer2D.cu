// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/lbm/LbmInitializer.hpp"
#include "FSI_Simulator/lbm/LbmFlowField2D.hpp"
#include "FSI_Simulator/lbm/LbmConstants.hpp"
#include "FSI_Simulator/utils/CudaErrorCheck.cuh"
#include "FSI_Simulator/utils/Logger.hpp"
#include <vector>

namespace fsi
{

    namespace lbm
    {

        namespace LbmD2Q9
        {
            extern __constant__ float c_w[Q];
            extern __constant__ float c_ex[Q];
            extern __constant__ float c_ey[Q];
            extern __constant__ float c_cs2;
        }

        namespace LbmInitializer
        {

            // --- 2D CUDA initialization kernel ---
            __global__ void initialize_state_kernel_2d(
                float *m_pre,
                float *m_post,
                const LbmNodeFlag *flags,
                int nx, int ny,
                float initial_density,
                const float *bc_velocities)
            {
                long i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= (long)nx * ny)
                    return;

                LbmNodeFlag flag = flags[i];
                float ux = 0.0f, uy = 0.0f;
                float rho = 1.0f;

                switch (flag)
                {
                case LbmNodeFlag::InletLeft:
                    ux = bc_velocities[0];
                    break;
                case LbmNodeFlag::InletDown:
                    uy = bc_velocities[1];
                    break;
                case LbmNodeFlag::InletRight:
                    ux = -bc_velocities[2];
                    break;
                case LbmNodeFlag::InletUp:
                    uy = -bc_velocities[3];
                    break;
                default:
                    break;
                }

                // --- 2D (D2Q9) equaliibrium distribution ---
                constexpr int Q = LbmD2Q9::Q;

                float u_sq = ux * ux + uy * uy;
                float feq[Q];
                for (int k = 0; k < Q; ++k)
                {
                    float c_dot_u = LbmD2Q9::c_ex[k] * ux + LbmD2Q9::c_ey[k] * uy;
                    feq[k] = LbmD2Q9::c_w[k] * rho * (1.0f + 3.0f * c_dot_u + 4.5f * c_dot_u * c_dot_u - 1.5f * u_sq);
                }

                // compute moments from f_eq
                float invRho = 1 / rho;
                float pixx = feq[1] + feq[3] + feq[5] + feq[6] + feq[7] + feq[8];
                float piyy = feq[2] + feq[4] + feq[5] + feq[6] + feq[7] + feq[8];
                float pixy = feq[5] - feq[6] + feq[7] - feq[8];
                pixx = 1 * (pixx * invRho - LbmD2Q9::c_cs2);
                piyy = 1 * (piyy * invRho - LbmD2Q9::c_cs2);
                pixy = 1 * (pixy * invRho);

                // write 2D moments
                constexpr int NM = 6;
                size_t moment_idx = (size_t)i * NM;
                m_pre[moment_idx + 0] = m_post[moment_idx + 0] = rho;
                m_pre[moment_idx + 1] = m_post[moment_idx + 1] = ux;
                m_pre[moment_idx + 2] = m_post[moment_idx + 2] = uy;
                m_pre[moment_idx + 3] = m_post[moment_idx + 3] = pixx;
                m_pre[moment_idx + 4] = m_post[moment_idx + 4] = piyy;
                m_pre[moment_idx + 5] = m_post[moment_idx + 5] = pixy;
            }

            void initializeState2D(LbmFlowField2D &flow_field, const SimulationParameters2D &params)
            {
                long num_nodes = flow_field.getNumNodes();
                int nx = flow_field.getNx();
                int ny = flow_field.getNy();

                std::vector<LbmNodeFlag> h_flags(num_nodes, LbmNodeFlag::Fluid); // default is fluid

                LOG_INFO("Applying {} boundary conditions from config...", params.boundaries.size());
                for (const auto &bc : params.boundaries)
                {
                    // get position from config
                    int x = bc.position[0];
                    int y = bc.position[1];

                    // check if boundary is within grid
                    if (x >= 0 && x < nx && y >= 0 && y < flow_field.getNy())
                    {
                        long idx = (long)y * nx + x;
                        h_flags[idx] = bc.flag;
                    }
                    else
                    {
                        LOG_WARN("Boundary condition at [{}, {}] is outside the grid and will be ignored.", x, y);
                    }
                }

                flow_field.getFlags().upload(h_flags);

                // prepare BC velocity array
                std::vector<float> h_bc_velocities = params.boundary_velocities;
                if (h_bc_velocities.size() != 4)
                {
                    LOG_WARN("Boundary velocity array size is not 4, resetting to zeros.");
                    h_bc_velocities.assign(4, 0.0f); // safe fallback
                }
                CudaArray<float> d_bc_velocities(h_bc_velocities);

                // launch initialization kernel
                dim3 block(256);
                dim3 grid((num_nodes + block.x - 1) / block.x);

                initialize_state_kernel_2d<<<grid, block>>>(
                    flow_field.m_fMom.data(),
                    flow_field.m_fMomPost.data(),
                    flow_field.getFlags().data(),
                    nx, ny,
                    params.fluid_density,
                    d_bc_velocities.data());

                CUDA_CHECK_KERNEL();
                LOG_INFO("2D LBM flow field state initialized on GPU.");
            }

        } // namespace LbmInitializer

    } // namespace lbm

} // namespace fsi