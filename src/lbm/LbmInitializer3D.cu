// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/lbm/LbmInitializer.hpp"
#include "FSI_Simulator/lbm/LbmFlowField3D.hpp"
#include "FSI_Simulator/lbm/LbmConstants.hpp"
#include "FSI_Simulator/utils/CudaErrorCheck.cuh"
#include "FSI_Simulator/utils/Logger.hpp"
#include <vector>

namespace fsi
{

    namespace lbm
    {

        namespace LbmD3Q27
        {
            extern __constant__ float c_w[Q];
            extern __constant__ float c_ex[Q];
            extern __constant__ float c_ey[Q];
            extern __constant__ float c_ez[Q];
            extern __constant__ float c_cs2;
        }

        namespace LbmInitializer
        {

            // --- 3D CUDA initialization kernel ---
            __global__ void initialize_state_kernel_3d(
                float *m_pre,
                float *m_post,
                const LbmNodeFlag *flags,
                int nx, int ny, int nz,
                float initial_density,
                const float *bc_velocities)
            {
                long i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= (long)nx * ny * nz)
                    return;

                LbmNodeFlag flag = flags[i];
                float ux = 0.0f, uy = 0.0f, uz = 0.0f;
                float rho = 1.0f;

                // Set initial velocity based on flag
                switch (flag)
                {
                case LbmNodeFlag::InletLeft:
                    ux = bc_velocities[0];
                    break;
                case LbmNodeFlag::InletDown:
                    uz = bc_velocities[1];
                    break;
                case LbmNodeFlag::InletRight:
                    ux = -bc_velocities[2];
                    break;
                case LbmNodeFlag::InletUp:
                    uz = -bc_velocities[3];
                    break;
                case LbmNodeFlag::InletFront:
                    uy = bc_velocities[4];
                    break;
                case LbmNodeFlag::InletBack:
                    uy = -bc_velocities[5];
                    break;
                default:
                    break;
                }

                // --- 3D (D3Q27) equilibrium moments calculation ---
                constexpr int Q = LbmD3Q27::Q;

                float u_sq = ux * ux + uy * uy + uz * uz;
                float feq[Q];
                for (int k = 0; k < Q; ++k)
                {
                    float c_dot_u = LbmD3Q27::c_ex[k] * ux + LbmD3Q27::c_ey[k] * uy + LbmD3Q27::c_ez[k] * uz;
                    feq[k] = LbmD3Q27::c_w[k] * rho * (1.0f + 3.0f * c_dot_u + 4.5f * c_dot_u * c_dot_u - 1.5f * u_sq);
                }

                float pixx = ((feq[1] + feq[2] + feq[7] + feq[8] + feq[9] + feq[10] + feq[13] + feq[14] + feq[15] + feq[16] + feq[19] + feq[20] + feq[21] + feq[22] + feq[23] + feq[24] + feq[25] + feq[26]));
                float piyy = ((feq[3] + feq[4] + feq[7] + feq[8] + feq[11] + feq[12] + feq[13] + feq[14] + feq[17] + feq[18] + feq[19] + feq[20] + feq[21] + feq[22] + feq[23] + feq[24] + feq[25] + feq[26]));
                float pizz = ((feq[5] + feq[6] + feq[9] + feq[10] + feq[11] + feq[12] + feq[15] + feq[16] + feq[17] + feq[18] + feq[19] + feq[20] + feq[21] + feq[22] + feq[23] + feq[24] + feq[25] + feq[26]));
                float pixy = (((feq[7] + feq[8] + feq[19] + feq[20] + feq[21] + feq[22]) - (feq[13] + feq[14] + feq[23] + feq[24] + feq[25] + feq[26])));
                float piyz = (((feq[11] + feq[12] + feq[19] + feq[20] + feq[25] + feq[26]) - (feq[17] + feq[18] + feq[21] + feq[22] + feq[23] + feq[24])));
                float pixz = (((feq[9] + feq[10] + feq[19] + feq[20] + feq[23] + feq[24]) - (feq[15] + feq[16] + feq[21] + feq[22] + feq[25] + feq[26])));

                float invRho = 1 / rho;
                float cs2 = LbmD3Q27::c_cs2;
                pixx = 1 * (pixx * invRho - 1.0 * cs2);
                piyy = 1 * (piyy * invRho - 1.0 * cs2);
                pizz = 1 * (pizz * invRho - 1.0 * cs2);
                pixy = 1 * (pixy * invRho);
                piyz = 1 * (piyz * invRho);
                pixz = 1 * (pixz * invRho);

                constexpr int NM = 10;
                size_t moment_idx = (size_t)i * NM;
                m_pre[moment_idx + 0] = m_post[moment_idx + 0] = rho;
                m_pre[moment_idx + 1] = m_post[moment_idx + 1] = ux;
                m_pre[moment_idx + 2] = m_post[moment_idx + 2] = uy;
                m_pre[moment_idx + 3] = m_post[moment_idx + 3] = uz;
                m_pre[moment_idx + 4] = m_post[moment_idx + 4] = pixx;
                m_pre[moment_idx + 5] = m_post[moment_idx + 5] = piyy;
                m_pre[moment_idx + 6] = m_post[moment_idx + 6] = pizz;
                m_pre[moment_idx + 7] = m_post[moment_idx + 7] = pixy;
                m_pre[moment_idx + 8] = m_post[moment_idx + 8] = piyz;
                m_pre[moment_idx + 9] = m_post[moment_idx + 9] = pixz;
            }

            void initializeState3D(LbmFlowField3D &flow_field, const SimulationParameters3D &params)
            {
                long num_nodes = flow_field.getNumNodes();
                int nx = flow_field.getNx();
                int ny = flow_field.getNy();
                int nz = flow_field.getNz();

                std::vector<LbmNodeFlag> h_flags(num_nodes, LbmNodeFlag::Fluid);

                LOG_INFO("Applying {} boundary conditions from config...", params.boundaries.size());
                for (const auto &bc : params.boundaries)
                {
                    int x = bc.position[0];
                    int y = bc.position[1];
                    int z = bc.position[2];

                    if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz)
                    {
                        long idx = (long)z * ny * nx + (long)y * nx + x;
                        h_flags[idx] = bc.flag;
                    }
                    else
                    {
                        LOG_WARN("Boundary condition at [{}, {}, {}] is outside the grid and will be ignored.", x, y, z);
                    }
                }

                flow_field.getFlags().upload(h_flags);

                std::vector<float> h_bc_velocities = params.boundary_velocities;
                if (h_bc_velocities.size() != 6)
                {
                    LOG_WARN("Boundary velocity array size is not 6, resetting to zeros.");
                    h_bc_velocities.assign(6, 0.0f);
                }
                CudaArray<float> d_bc_velocities(h_bc_velocities);

                dim3 block(256);
                dim3 grid((num_nodes + block.x - 1) / block.x);

                initialize_state_kernel_3d<<<grid, block>>>(
                    flow_field.m_moments_pre.data(),
                    flow_field.m_moments_post.data(),
                    flow_field.getFlags().data(),
                    nx, ny, nz,
                    params.fluid_density,
                    d_bc_velocities.data());

                CUDA_CHECK_KERNEL();
                LOG_INFO("3D LBM flow field state initialized on GPU.");
            }

        } // namespace LbmInitializer

    } // namespace lbm

} // namespace fsi