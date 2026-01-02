// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/fsi/FsiCouplingUtils.cuh"
#include "FSI_Simulator/common/CudaCommon.cuh"

namespace fsi
{
    namespace coupling
    {
        __global__ void update_surface_pos_kernel(
            float *d_surface_pos,
            float *d_surface_vel,
            const vec3_t *d_volume_pos,
            const vec3_t *d_volume_vel,
            const unsigned int *d_surface_to_volume_map,
            int num_surface_vertices)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= num_surface_vertices)
                return;

            int volume_idx = d_surface_to_volume_map[i];

            // (Gather)
            d_surface_pos[i * 3 + 0] = d_volume_pos[volume_idx][0];
            d_surface_pos[i * 3 + 1] = d_volume_pos[volume_idx][1];
            d_surface_pos[i * 3 + 2] = d_volume_pos[volume_idx][2];

            d_surface_vel[i * 3 + 0] = d_volume_vel[volume_idx][0];
            d_surface_vel[i * 3 + 1] = d_volume_vel[volume_idx][1];
            d_surface_vel[i * 3 + 2] = d_volume_vel[volume_idx][2];
        }

        __global__ void update_surface_pos_kernel2d(
            float *d_surface_pos,
            float *d_surface_vel,
            const vec2_t *d_volume_pos,
            const vec2_t *d_volume_vel,
            const unsigned int *d_surface_to_volume_map,
            int num_surface_vertices)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= num_surface_vertices)
                return;

            int volume_idx = d_surface_to_volume_map[i];

            // (Gather)
            d_surface_pos[i * 2 + 0] = d_volume_pos[volume_idx][0];
            d_surface_pos[i * 2 + 1] = d_volume_pos[volume_idx][1];
            d_surface_vel[i * 2 + 0] = d_volume_vel[volume_idx][0];
            d_surface_vel[i * 2 + 1] = d_volume_vel[volume_idx][1];
            // printf("surf pos %d: %f, %f\n", i, d_surface_pos[i * 2 + 0], d_surface_pos[i * 2 + 1]);
        }

        __global__ void scatter_forces_kernel(
            vec3_t *d_volume_forces,
            const float *d_surface_forces,
            const unsigned int *d_surface_to_volume_map,
            int num_surface_vertices)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= num_surface_vertices)
                return;

            int volume_idx = d_surface_to_volume_map[i];

            d_volume_forces[volume_idx][0] = d_surface_forces[i * 3 + 0];
            d_volume_forces[volume_idx][1] = d_surface_forces[i * 3 + 1];
            d_volume_forces[volume_idx][2] = d_surface_forces[i * 3 + 2];
        }

        __global__ void scatter_forces_kernel2d(
            vec2_t *d_volume_forces,
            const float *d_surface_forces,
            const unsigned int *d_surface_to_volume_map,
            int num_surface_vertices)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= num_surface_vertices)
                return;

            int volume_idx = d_surface_to_volume_map[i];

            d_volume_forces[volume_idx][0] = d_surface_forces[i * 2 + 0];
            d_volume_forces[volume_idx][1] = d_surface_forces[i * 2 + 1];
        }

        void updateSurfaceStatesFromVolume(
            fem::VbdSceneDataGpu3D &scene_data,
            FsiCouplingDataGpu &coupling_data,
            size_t num_surface_vertices,
            cudaStream_t stream)
        {
            if (num_surface_vertices == 0)
                return;

            const int block_size = KernelConfig::BLOCK_SIZE_1D;
            const int grid_size = (num_surface_vertices + block_size - 1) / block_size;

            update_surface_pos_kernel<<<grid_size, block_size, 0, stream>>>(
                coupling_data.surface_positions.data(),
                coupling_data.surface_velocities.data(),
                scene_data.positions.data(),
                scene_data.velocities.data(),
                coupling_data.d_surface_to_volume_map.data(),
                num_surface_vertices);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        void updateSurfaceStatesFromVolume(
            fem::VbdSceneDataGpu2D &scene_data,
            FsiCouplingDataGpu &coupling_data,
            size_t num_surface_vertices,
            cudaStream_t stream)
        {
            if (num_surface_vertices == 0)
                return;

            const int block_size = KernelConfig::BLOCK_SIZE_1D;
            const int grid_size = (num_surface_vertices + block_size - 1) / block_size;

            update_surface_pos_kernel2d<<<grid_size, block_size, 0, stream>>>(
                coupling_data.surface_positions.data(),
                coupling_data.surface_velocities.data(),
                scene_data.positions.data(),
                scene_data.velocities.data(),
                coupling_data.d_surface_to_volume_map.data(),
                num_surface_vertices);
            CUDA_CHECK_KERNEL();

            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        void scatterSurfaceForcesToVolume(
            fem::VbdSceneDataGpu3D &scene_data,
            FsiCouplingDataGpu &coupling_data,
            size_t num_surface_vertices,
            cudaStream_t stream)
        {
            if (num_surface_vertices == 0)
                return;

            scene_data.forces.setZeroAsync(stream);

            const int block_size = KernelConfig::BLOCK_SIZE_1D;
            const int grid_size = (num_surface_vertices + block_size - 1) / block_size;

            scatter_forces_kernel<<<grid_size, block_size, 0, stream>>>(
                scene_data.forces.data(),
                coupling_data.surface_forces.data(),
                coupling_data.d_surface_to_volume_map.data(),
                num_surface_vertices);
            CUDA_CHECK_KERNEL();
        }

        void scatterSurfaceForcesToVolume(
            fem::VbdSceneDataGpu2D &scene_data,
            FsiCouplingDataGpu &coupling_data,
            size_t num_surface_vertices,
            cudaStream_t stream)
        {
            if (num_surface_vertices == 0)
                return;

            scene_data.forces.setZeroAsync(stream);

            const int block_size = KernelConfig::BLOCK_SIZE_1D;
            const int grid_size = (num_surface_vertices + block_size - 1) / block_size;

            scatter_forces_kernel2d<<<grid_size, block_size, 0, stream>>>(
                scene_data.forces.data(),
                coupling_data.surface_forces.data(),
                coupling_data.d_surface_to_volume_map.data(),
                num_surface_vertices);
            CUDA_CHECK_KERNEL();
        }

    } // namespace coupling
} // namespace fsi