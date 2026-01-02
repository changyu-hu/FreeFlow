// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/control/LBSControlUtils.cuh"

namespace fsi
{
    namespace control
    {

        __global__ void computeTetDmInv(vec3_t *vpos, int tetnum, unsigned int *tetvIdx_dev, mat3_t *tetDmInvFaInv_dev)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid < tetnum)
            {
                vec3_t v0 = vpos[tetvIdx_dev[4 * tid]];
                vec3_t v1 = vpos[tetvIdx_dev[4 * tid + 1]];
                vec3_t v2 = vpos[tetvIdx_dev[4 * tid + 2]];
                vec3_t v3 = vpos[tetvIdx_dev[4 * tid + 3]];

                mat3_t *DmInv = tetDmInvFaInv_dev + tid;
                mat3_t Dm = glm::transpose(mat3_t(v1 - v0, v2 - v0, v3 - v0));
                real det = glm::determinant(Dm);
                if (det == 0)
                {
                    printf("error: det = 0\n");
                }
                else
                {
                    *DmInv = glm::inverse(Dm);
                }
            }
        }

        __global__ void computeLBSPosition(
            int vnum, vec3_t *vpos,
            int cnum, vec3_t *v_lbs,
            int offset,
            real *lbs_weight,
            vec3_t *lbs_shift,
            mat3_t *lbs_rotation,
            vec3_t center)
        {
            int vid = blockIdx.x * blockDim.x + threadIdx.x;
            vec3_t v = vpos[vid];

            if (vid < vnum)
            {
                v_lbs[offset + vid] = center;
                real *w = lbs_weight + cnum * vid;
                for (size_t i = 0; i < cnum; i++)
                {
                    vec3_t shift = lbs_shift[i];
                    mat3_t rotation = lbs_rotation[i];
                    v_lbs[offset + vid] += w[i] * (rotation * (v - center) + shift);
                }
            }
        }

        __global__ void computeLBSWeight(
            int vnum,
            int cnum,
            real *lbs_weight,
            real *lbs_dist,
            real omega)
        {
            int vid = blockIdx.x * blockDim.x + threadIdx.x;
            if (vid < vnum)
            {
                real *w = lbs_weight + cnum * vid;
                real *d = lbs_dist + cnum * vid;
                real w_sum = 0.0;
                for (size_t i = 0; i < cnum; i++)
                {
                    real dist = d[i];
                    w[i] = exp(-dist * dist / (2.0 * omega * omega));
                    w_sum += w[i];
                }
                for (size_t i = 0; i < cnum; i++)
                {
                    w[i] = w[i] / w_sum;
                }
            }
        }

        void compute_lbs_weight(
            int vnum,
            int cnum,
            real *lbs_weight,
            real *lbs_dist,
            real omega,
            cudaStream_t stream)
        {
            const int block_size = KernelConfig::BLOCK_SIZE_1D;
            const int grid_size = (vnum + block_size - 1) / block_size;
            computeLBSWeight<<<grid_size, block_size, 0, stream>>>(
                vnum, cnum, lbs_weight, lbs_dist, omega);
            CUDA_CHECK_KERNEL();
        }

        void update_target_position(
            int vnum, vec3_t *vpos,
            int tetnum, unsigned int *tetvIdx_dev, mat3_t *tetDmInv,
            cudaStream_t stream)
        {
            const int block_size = KernelConfig::BLOCK_SIZE_1D;
            const int grid_size = (tetnum + block_size - 1) / block_size;

            computeTetDmInv<<<grid_size, block_size, 0, stream>>>(
                vpos, tetnum, tetvIdx_dev, tetDmInv);
            CUDA_CHECK_KERNEL();
        }

        void compute_lbs_position(
            int vnum, vec3_t *vpos,
            int cnum, vec3_t *v_lbs,
            int offset,
            real *lbs_weight,
            vec3_t *lbs_shift,
            mat3_t *lbs_rotation,
            vec3_t center,
            cudaStream_t stream)
        {
            const int block_size = KernelConfig::BLOCK_SIZE_1D;
            const int grid_size = (vnum + block_size - 1) / block_size;

            computeLBSPosition<<<grid_size, block_size, 0, stream>>>(
                vnum, vpos, cnum, v_lbs, offset, lbs_weight, lbs_shift, lbs_rotation, center);
            CUDA_CHECK_KERNEL();
        }

        __global__ void computeTriDmInv(vec2_t *vpos, int tetnum, unsigned int *tetvIdx_dev, mat2_t *tetDmInvFaInv_dev)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid < tetnum)
            {
                vec2_t v0 = vpos[tetvIdx_dev[3 * tid]];
                vec2_t v1 = vpos[tetvIdx_dev[3 * tid + 1]];
                vec2_t v2 = vpos[tetvIdx_dev[3 * tid + 2]];

                mat2_t *DmInv = tetDmInvFaInv_dev + tid;
                vec2_t e1 = v0 - v2;
                vec2_t e2 = v1 - v2;
                mat2_t Dm(e1[0], e2[0],
                          e1[1], e2[1]);
                real det = glm::determinant(Dm);
                if (det == 0)
                {
                    printf("error: det = 0\n");
                }
                else
                {
                    *DmInv = glm::inverse(Dm);
                }
            }
        }

        __global__ void computeLBSPosition(
            int vnum, vec2_t *vpos,
            int cnum, vec2_t *v_lbs,
            int offset,
            real *lbs_weight,
            vec2_t *lbs_shift,
            real *lbs_rotation,
            vec2_t center)
        {
            int vid = blockIdx.x * blockDim.x + threadIdx.x;
            vec2_t v = vpos[vid];
            int curind = offset + vid;

            if (vid < vnum)
            {
                v_lbs[curind] = center;
                real *w = lbs_weight + cnum * vid;
                for (size_t i = 0; i < cnum; i++)
                {
                    vec2_t shift = lbs_shift[i];
                    real angle = lbs_rotation[i];
                    mat2_t rotation(
                        cos(angle), -sin(angle),
                        sin(angle), cos(angle));
                    v_lbs[curind] += w[i] * (rotation * (v - center) + shift);
                }
            }
        }

        void update_target_position(
            int vnum, vec2_t *vpos,
            int trinum, unsigned int *trivIdx_dev, mat2_t *triDmInv_dev,
            cudaStream_t stream)
        {
            const int block_size = KernelConfig::BLOCK_SIZE_1D;
            const int grid_size = (trinum + block_size - 1) / block_size;

            computeTriDmInv<<<grid_size, block_size, 0, stream>>>(
                vpos, trinum, trivIdx_dev, triDmInv_dev);
            CUDA_CHECK_KERNEL();
        }

        void compute_lbs_position(
            int vnum, vec2_t *vpos,
            int cnum, vec2_t *v_lbs,
            int offset,
            real *lbs_weight,
            vec2_t *lbs_shift,
            real *lbs_rotation,
            vec2_t center,
            cudaStream_t stream)
        {
            const int block_size = KernelConfig::BLOCK_SIZE_1D;
            const int grid_size = (vnum + block_size - 1) / block_size;

            computeLBSPosition<<<grid_size, block_size, 0, stream>>>(
                vnum, vpos, cnum, v_lbs, offset, lbs_weight, lbs_shift, lbs_rotation, center);
            CUDA_CHECK_KERNEL();
        }

    }
}