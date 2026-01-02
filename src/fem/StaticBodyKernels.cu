// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/fem/StaticBodyKernels.cuh"

namespace fsi
{
    namespace fem
    {

        __global__ void set_vel_kernel(vec3_t *d_velocities, vec3_t vel, int num_vertices)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < num_vertices)
            {
                d_velocities[idx] = vel;
            }
        }

        __global__ void add_pos_kernel(vec3_t *d_positions, vec3_t shift, int num_vertices)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < num_vertices)
            {
                d_positions[idx] += shift;
            }
        }

        __global__ void set_vel_kernel(vec2_t *d_velocities, vec2_t vel, int num_vertices)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < num_vertices)
            {
                d_velocities[idx] = vel;
            }
        }

        __global__ void add_pos_kernel(vec2_t *d_positions, vec2_t shift, int num_vertices)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < num_vertices)
            {
                d_positions[idx] += shift;
            }
        }

        void set_vel(VbdSceneDataGpu3D &data, vec3_t vel, cudaStream_t stream)
        {
            int vnum = data.vertex_num;
            dim3 block(KernelConfig::BLOCK_SIZE_1D, 1, 1);
            dim3 grid((vnum + KernelConfig::BLOCK_SIZE_1D - 1) / KernelConfig::BLOCK_SIZE_1D, 1, 1);
            set_vel_kernel<<<grid, block, 0, stream>>>(data.velocities.data(), vel, vnum);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        void add_pos(VbdSceneDataGpu3D &data, vec3_t shift, cudaStream_t stream)
        {
            int vnum = data.vertex_num;
            dim3 block(KernelConfig::BLOCK_SIZE_1D, 1, 1);
            dim3 grid((vnum + KernelConfig::BLOCK_SIZE_1D - 1) / KernelConfig::BLOCK_SIZE_1D, 1, 1);
            add_pos_kernel<<<grid, block, 0, stream>>>(data.positions.data(), shift, vnum);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        void set_vel(VbdSceneDataGpu2D &data, vec2_t vel, cudaStream_t stream)
        {
            int vnum = data.vertex_num;
            dim3 block(KernelConfig::BLOCK_SIZE_1D, 1, 1);
            dim3 grid((vnum + KernelConfig::BLOCK_SIZE_1D - 1) / KernelConfig::BLOCK_SIZE_1D, 1, 1);
            set_vel_kernel<<<grid, block, 0, stream>>>(data.velocities.data(), vel, vnum);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        void add_pos(VbdSceneDataGpu2D &data, vec2_t shift, cudaStream_t stream)
        {
            int vnum = data.vertex_num;
            dim3 block(KernelConfig::BLOCK_SIZE_1D, 1, 1);
            dim3 grid((vnum + KernelConfig::BLOCK_SIZE_1D - 1) / KernelConfig::BLOCK_SIZE_1D, 1, 1);
            add_pos_kernel<<<grid, block, 0, stream>>>(data.positions.data(), shift, vnum);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

    }
}