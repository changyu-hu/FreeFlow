// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/fem/VBDKernels.cuh"
#include "FSI_Simulator/utils/NumericUtils.cuh"

namespace fsi
{
    namespace fem
    {

        __global__ void kernel_initguess3d_ip(
            int vnum,
            vec3_t *vpos,
            vec3_t *vprepos,
            vec3_t *vitrprepos,
            vec3_t *vitrpreprepos,
            vec3_t *vinertia,
            vec3_t *vvel,
            vec3_t *vprevel,
            vec3_t *vforce,
            real *vmass,
            real dt)
        {
            int vidx = threadIdx.x + blockIdx.x * blockDim.x;
            if (vidx >= vnum)
                return;
            vec3_t at = (vvel[vidx] - vprevel[vidx]) / dt;
            vec3_t e = vforce[vidx] / vmass[vidx];
            // if (glm::length(e) > 10.0)
            //     printf("vidx: %d, vforce[vidx]: %f %f %f, acc[vidx]: %f %f %f\n", vidx, vforce[vidx][0], vforce[vidx][1], vforce[vidx][2], e[0], e[1], e[2]);
            real aextnorm = glm::length(e);
            if (aextnorm == 0.0)
            {
                e = vec3_t(0.0);
            }
            else
            {
                e = e / aextnorm;
            }
            real adot;
            adot = glm::dot(at, e);
            if (adot < 0.0)
            {
                adot = 0.0;
            }
            else if (adot > aextnorm)
            {
                adot = aextnorm;
            }

            vinertia[vidx] = vpos[vidx] + dt * vvel[vidx] + dt * dt * vforce[vidx] / vmass[vidx];
            vprepos[vidx] = vpos[vidx];
            vpos[vidx] = vpos[vidx] + dt * vvel[vidx] + dt * dt * adot * e;
            vitrprepos[vidx] = vpos[vidx];
            vitrpreprepos[vidx] = vpos[vidx];
        }

        __device__ void assembleVertexVForceAndHessian(const real *dE_dF, const real d2E_dF_dF[9][9], real m1, real m2, real m3,
                                                       real *force, real *h)
        {
            // force is the negative of gradient
            force[0] -= dE_dF[0] * m1 + dE_dF[3] * m2 + dE_dF[6] * m3;
            force[1] -= dE_dF[1] * m1 + dE_dF[4] * m2 + dE_dF[7] * m3;
            force[2] -= dE_dF[2] * m1 + dE_dF[5] * m2 + dE_dF[8] * m3;

            real HL_Row1[9];
            real HL_Row2[9];
            real HL_Row3[9];

            real *HL[3] = {HL_Row1, HL_Row2, HL_Row3};

            for (int32_t iCol = 0; iCol < 9; iCol++)
            {
                HL_Row1[iCol] = d2E_dF_dF[0][iCol] * m1;
                HL_Row1[iCol] += d2E_dF_dF[3][iCol] * m2;
                HL_Row1[iCol] += d2E_dF_dF[6][iCol] * m3;

                HL_Row2[iCol] = d2E_dF_dF[1][iCol] * m1;
                HL_Row2[iCol] += d2E_dF_dF[4][iCol] * m2;
                HL_Row2[iCol] += d2E_dF_dF[7][iCol] * m3;

                HL_Row3[iCol] = d2E_dF_dF[2][iCol] * m1;
                HL_Row3[iCol] += d2E_dF_dF[5][iCol] * m2;
                HL_Row3[iCol] += d2E_dF_dF[8][iCol] * m3;
            }

            for (int32_t iRow = 0; iRow < 3; iRow++)
            {
                h[iRow] += HL[iRow][0] * m1 + HL[iRow][3] * m2 + HL[iRow][6] * m3;
                h[iRow + 3] += HL[iRow][1] * m1 + HL[iRow][4] * m2 + HL[iRow][7] * m3;
                h[iRow + 6] += HL[iRow][2] * m1 + HL[iRow][5] * m2 + HL[iRow][8] * m3;
            }
        }

        // neohookean.
        __device__ inline void compute_neohookean_derivatives(
            real mu, real lambda, real tet_volume, int vidxintet,
            const mat3_t &DmInvFaInv,
            const vec3_t &v0, const vec3_t &v1, const vec3_t &v2, const vec3_t &v3,
            // 输出参数
            real *force_contrib_out,
            real *hessian_contrib_out)
        {
            // 1. 计算变形梯度 F = Ds * DmInv
            mat3_t Ds;
            for (size_t i = 0; i < 3; i++)
            {
                Ds[i][0] = v1[i] - v0[i];
                Ds[i][1] = v2[i] - v0[i];
                Ds[i][2] = v3[i] - v0[i];
            }

            const mat3_t F = DmInvFaInv * Ds;

            // 2. 计算能量对F的梯度 P = dE/dF (Piola-Kirchhoff I Stress)
            const real detF = glm::determinant(F);
            const real Ic = glm::dot(F[0], F[0]) + glm::dot(F[1], F[1]) + glm::dot(F[2], F[2]);
            const real a = 1.0 + 0.75 * mu / lambda;
            const real k = detF - a;
            const real ddetF_dF[9] = {
                F[1][1] * F[2][2] - F[1][2] * F[2][1],
                F[0][2] * F[2][1] - F[0][1] * F[2][2],
                F[0][1] * F[1][2] - F[0][2] * F[1][1],
                F[1][2] * F[2][0] - F[1][0] * F[2][2],
                F[0][0] * F[2][2] - F[0][2] * F[2][0],
                F[0][2] * F[1][0] - F[0][0] * F[1][2],
                F[1][0] * F[2][1] - F[1][1] * F[2][0],
                F[0][1] * F[2][0] - F[0][0] * F[2][1],
                F[0][0] * F[1][1] - F[0][1] * F[1][0]};

            const real C = (1.0 - 1.0 / (Ic + 1.0)) * mu;

            // 3. 计算能量对F的Hessian d2E/dF2 (9x9矩阵)
            real d2E_dF_dF[9][9] = {0};
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    d2E_dF_dF[i][j] = ddetF_dF[i] * ddetF_dF[j];
                }
            }

            d2E_dF_dF[0][4] += k * F[2][2];
            d2E_dF_dF[4][0] += k * F[2][2];
            d2E_dF_dF[0][5] += k * -F[1][2];
            d2E_dF_dF[5][0] += k * -F[1][2];
            d2E_dF_dF[0][7] += k * -F[2][1];
            d2E_dF_dF[7][0] += k * -F[2][1];
            d2E_dF_dF[0][8] += k * F[1][1];
            d2E_dF_dF[8][0] += k * F[1][1];

            d2E_dF_dF[1][3] += k * -F[2][2];
            d2E_dF_dF[3][1] += k * -F[2][2];
            d2E_dF_dF[1][5] += k * F[0][2];
            d2E_dF_dF[5][1] += k * F[0][2];
            d2E_dF_dF[1][6] += k * F[2][1];
            d2E_dF_dF[6][1] += k * F[2][1];
            d2E_dF_dF[1][8] += k * -F[0][1];
            d2E_dF_dF[8][1] += k * -F[0][1];

            d2E_dF_dF[2][3] += k * F[1][2];
            d2E_dF_dF[3][2] += k * F[1][2];
            d2E_dF_dF[2][4] += k * -F[0][2];
            d2E_dF_dF[4][2] += k * -F[0][2];
            d2E_dF_dF[2][6] += k * -F[1][1];
            d2E_dF_dF[6][2] += k * -F[1][1];
            d2E_dF_dF[2][7] += k * F[0][1];
            d2E_dF_dF[7][2] += k * F[0][1];

            d2E_dF_dF[3][7] += k * F[2][0];
            d2E_dF_dF[7][3] += k * F[2][0];
            d2E_dF_dF[3][8] += k * -F[1][0];
            d2E_dF_dF[8][3] += k * -F[1][0];

            d2E_dF_dF[4][6] += k * -F[2][0];
            d2E_dF_dF[6][4] += k * -F[2][0];
            d2E_dF_dF[4][8] += k * F[0][0];
            d2E_dF_dF[8][4] += k * F[0][0];

            d2E_dF_dF[5][6] += k * F[1][0];
            d2E_dF_dF[6][5] += k * F[1][0];
            d2E_dF_dF[5][7] += k * -F[0][0];
            d2E_dF_dF[7][5] += k * -F[0][0];

            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    d2E_dF_dF[i][j] *= lambda;
                }
            }

            for (int i = 0; i < 9; i++)
            {
                d2E_dF_dF[i][i] += C;
            }

            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    d2E_dF_dF[i][j] *= tet_volume;
                }
            }

            real dE_dF[9] = {0};
            for (int i = 0; i < 9; i++)
            {
                dE_dF[i] = (ddetF_dF[i] * lambda * k + F[i % 3][i / 3] * C) * tet_volume;
            }

            const real ms[4][3] = {
                {-DmInvFaInv[0][0] - DmInvFaInv[1][0] - DmInvFaInv[2][0], -DmInvFaInv[0][1] - DmInvFaInv[1][1] - DmInvFaInv[2][1], -DmInvFaInv[0][2] - DmInvFaInv[1][2] - DmInvFaInv[2][2]},
                {DmInvFaInv[0][0], DmInvFaInv[0][1], DmInvFaInv[0][2]},
                {DmInvFaInv[1][0], DmInvFaInv[1][1], DmInvFaInv[1][2]},
                {DmInvFaInv[2][0], DmInvFaInv[2][1], DmInvFaInv[2][2]}};

            const real m1 = ms[vidxintet][0], m2 = ms[vidxintet][1], m3 = ms[vidxintet][2];
            assembleVertexVForceAndHessian(dE_dF, d2E_dF_dF, m1, m2, m3, force_contrib_out, hessian_contrib_out);
        }

        __device__ inline void compute_lbs_derivatives(
            real stiffness,
            vec3_t vpos_lbs,
            vec3_t vpos_target,
            real *force_contrib_out,
            real *hessian_contrib_out)
        {
            force_contrib_out[0] += 2 * stiffness * (vpos_lbs[0] - vpos_target[0]);
            force_contrib_out[1] += 2 * stiffness * (vpos_lbs[1] - vpos_target[1]);
            force_contrib_out[2] += 2 * stiffness * (vpos_lbs[2] - vpos_target[2]);

            hessian_contrib_out[0] += 2 * stiffness;
            hessian_contrib_out[4] += 2 * stiffness;
            hessian_contrib_out[8] += 2 * stiffness;
        }

        __global__ void kernel_iterationonce3d_ip(
            int itr_idx, real itr_omega,
            unsigned int *vindices, int vnum,
            unsigned int *tetvIdx_dev, unsigned int *neitetNum_dev, unsigned int *neitetNumstart_dev, unsigned int *neitetIdx_dev, unsigned int *Idxinneitet_dev,
            real *vMass_dev,
            vec3_t *vpos,
            vec3_t *vadvectPos_dev,
            vec3_t *vitrprePos_dev,
            vec3_t *vitrpreprePos_dev,
            real dt,
            real *tet_mu_dev, real *tet_lambda_dev, real *tetVolume_dev, real *tet_kd_dev, mat3_t *tetDmInv_dev, mat3_t *tetFaInv_dev)
        {
            // shared memory for H and f.
            __shared__ __builtin_align__(16) real H_and_f_shared[KernelConfig::VBD3D_THREAD_DIM_FORTET * 12];
            // every block <-> 1 vertex; every theard <-> 1 neitet.
            int bidx = blockIdx.x;
            int tidx = threadIdx.x;

            if (bidx >= vnum)
                return;

            int vidx = vindices[bidx];

            real *Handf = H_and_f_shared + 12 * tidx;
            real *H = Handf;
            real *f = H + 9;
            // set H and f of this idxingroup to 0.
            for (size_t i = 0; i < 12; i++)
            {
                Handf[i] = 0.0;
            }

            // VBD3D_THREAD_DIM_FORTET tet each group, in sequence.
            int neitetnum = neitetNum_dev[vidx];
            if (tidx < neitetnum)
            {
                int tetidx = neitetIdx_dev[neitetNumstart_dev[vidx] + tidx];
                int vidxintet = Idxinneitet_dev[neitetNumstart_dev[vidx] + tidx];
                compute_neohookean_derivatives(
                    tet_mu_dev[tetidx], tet_lambda_dev[tetidx], tetVolume_dev[tetidx], vidxintet,
                    tetFaInv_dev[tetidx] * tetDmInv_dev[tetidx],
                    vpos[tetvIdx_dev[4 * tetidx]], vpos[tetvIdx_dev[4 * tetidx + 1]], vpos[tetvIdx_dev[4 * tetidx + 2]], vpos[tetvIdx_dev[4 * tetidx + 3]],
                    f, H);
            }
            if (neitetnum > KernelConfig::VBD3D_THREAD_DIM_FORTET)
            {
                // printf("vidx: %d, neitetNum: %d.\n", vidx, neitetnum);
                int extraSize = neitetnum - KernelConfig::VBD3D_THREAD_DIM_FORTET;
                for (int i = 0; i < extraSize; i += KernelConfig::VBD3D_THREAD_DIM_FORTET)
                {
                    int index = tidx + KernelConfig::VBD3D_THREAD_DIM_FORTET + i;
                    if (index < neitetnum)
                    {
                        int tetidx = neitetIdx_dev[neitetNumstart_dev[vidx] + index];
                        int vidxintet = Idxinneitet_dev[neitetNumstart_dev[vidx] + index];
                        compute_neohookean_derivatives(
                            tet_mu_dev[tetidx], tet_lambda_dev[tetidx], tetVolume_dev[tetidx], vidxintet,
                            tetFaInv_dev[tetidx] * tetDmInv_dev[tetidx],
                            vpos[tetvIdx_dev[4 * tetidx]], vpos[tetvIdx_dev[4 * tetidx + 1]], vpos[tetvIdx_dev[4 * tetidx + 2]], vpos[tetvIdx_dev[4 * tetidx + 3]],
                            f, H);
                    }
                }
            }

            __syncthreads();
            // log(N) add.
            for (unsigned int j = KernelConfig::VBD3D_THREAD_DIM_FORTET / 2; j > 0; j >>= 1)
            {
                if (tidx < j)
                {
                    real *Handf_other = H_and_f_shared + 12 * (tidx + j);
                    for (size_t i = 0; i < 12; i++)
                    {
                        Handf[i] = Handf[i] + Handf_other[i];
                    }
                }
                __syncthreads();
            }
            if (tidx == 0)
            {
                // inertia.
                real factor = vMass_dev[vidx] / dt / dt;
                mat3_t Htmp(H[0] + factor, H[3], H[6],
                            H[1], H[4] + factor, H[7],
                            H[2], H[5], H[8] + factor);
                vec3_t ftmp(f[0], f[1], f[2]);
                ftmp += -factor * (vpos[vidx] - vadvectPos_dev[vidx]);

                // update pos, prepos and preprepos.
                vitrpreprePos_dev[vidx] = vitrprePos_dev[vidx];
                vitrprePos_dev[vidx] = vpos[vidx];

                if (NumericUtils::isApproxGreater(ftmp[0] * ftmp[0] + ftmp[1] * ftmp[1] + ftmp[2] * ftmp[2], 0.0))
                {
                    vec3_t descentDirection;
                    real stepSize = 1e-3;
                    if (NumericUtils::isApproxZero(glm::determinant(Htmp)))
                    {
                        descentDirection = stepSize * ftmp;
                    }
                    else
                    {
                        descentDirection = glm::inverse(Htmp) * ftmp;
                        // printf("vidx: %d, f = [%f, %f, %f] ,H = [[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]\n", vidx, ftmp[0], ftmp[1], ftmp[2], Htmp[0][0], Htmp[0][1], Htmp[0][2],
                        //         Htmp[1][0], Htmp[1][1], Htmp[1][2],
                        //         Htmp[2][0], Htmp[2][1], Htmp[2][2]);
                    }

                    vpos[vidx] += descentDirection;
                }

                if (itr_idx > 1)
                {
                    vpos[vidx] = itr_omega * (vpos[vidx] - vitrpreprePos_dev[vidx]) + vitrpreprePos_dev[vidx];
                }
            }
        }

        __global__ void kernel_updatevel3d(
            int vnum,
            vec3_t *vpos,
            vec3_t *vprepos,
            vec3_t *vvel,
            vec3_t *vprevel,
            real dt)
        {
            int vidx = threadIdx.x + blockIdx.x * blockDim.x;
            if (vidx >= vnum)
                return;
            vprevel[vidx] = vvel[vidx];
            vvel[vidx] = (vpos[vidx] - vprepos[vidx]) / dt;
        }

        __global__ void kernel_solveLBSDynamicCorrection(
            int itr_idx, real itr_omega,
            unsigned int *vindices, int vnum,
            unsigned int *tetvIdx_dev, unsigned int *neitetNum_dev, unsigned int *neitetNumstart_dev, unsigned int *neitetIdx_dev, unsigned int *Idxinneitet_dev,
            real *vMass_dev,
            vec3_t *vpos_target,
            vec3_t *vpos_lbs,
            vec3_t *vitrprePos_dev,
            vec3_t *vitrpreprePos_dev,
            real *stiffness_dev,
            real *tet_mu_dev, real *tet_lambda_dev, real *tetVolume_dev, real *tet_kd_dev, mat3_t *tetDmInv_dev)
        {
            // shared memory for H and f.
            __shared__ __builtin_align__(16) real H_and_f_shared[KernelConfig::VBD3D_THREAD_DIM_FORTET * 12];
            // every block <-> 1 vertex; every theard <-> 1 neitet.
            int bidx = blockIdx.x;
            int tidx = threadIdx.x;

            if (bidx >= vnum)
                return;

            int vidx = vindices[bidx];

            real *Handf = H_and_f_shared + 12 * tidx;
            real *H = Handf;
            real *f = H + 9;
            // set H and f of this idxingroup to 0.
            for (size_t i = 0; i < 12; i++)
            {
                Handf[i] = 0.0;
            }

            // if(vidx == 80)
            // {
            //     printf("itr: %d, vidx: %d, f = [%f %f %f], H = [[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]\n",
            //         itr_idx, vidx, f[0], f[1], f[2], H[0], H[3], H[6], H[1], H[4], H[7], H[2], H[5], H[8]);
            // }

            // VBD3D_THREAD_DIM_FORTET tet each group, in sequence.
            int neitetnum = neitetNum_dev[vidx];
            if (tidx < neitetnum)
            {
                int tetidx = neitetIdx_dev[neitetNumstart_dev[vidx] + tidx];
                int vidxintet = Idxinneitet_dev[neitetNumstart_dev[vidx] + tidx];
                compute_neohookean_derivatives(
                    tet_mu_dev[tetidx], tet_lambda_dev[tetidx], tetVolume_dev[tetidx], vidxintet,
                    tetDmInv_dev[tetidx],
                    vpos_target[tetvIdx_dev[4 * tetidx]], vpos_target[tetvIdx_dev[4 * tetidx + 1]], vpos_target[tetvIdx_dev[4 * tetidx + 2]], vpos_target[tetvIdx_dev[4 * tetidx + 3]],
                    f, H);
                // if(vidx == 510)
                // {
                //     printf("vidx: %d, f = [%f %f %f], H = [[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]\n",
                //         vidx, f[0], f[1], f[2], H[0], H[3], H[6], H[1], H[4], H[7], H[2], H[5], H[8]);
                // }
            }
            if (neitetnum > KernelConfig::VBD3D_THREAD_DIM_FORTET)
            {
                // printf("vidx: %d, neitetNum: %d.\n", vidx, neitetnum);
                int extraSize = neitetnum - KernelConfig::VBD3D_THREAD_DIM_FORTET;
                for (int i = 0; i < extraSize; i += KernelConfig::VBD3D_THREAD_DIM_FORTET)
                {
                    int index = tidx + KernelConfig::VBD3D_THREAD_DIM_FORTET + i;
                    if (index < neitetnum)
                    {
                        int tetidx = neitetIdx_dev[neitetNumstart_dev[vidx] + index];
                        int vidxintet = Idxinneitet_dev[neitetNumstart_dev[vidx] + index];
                        compute_neohookean_derivatives(
                            tet_mu_dev[tetidx], tet_lambda_dev[tetidx], tetVolume_dev[tetidx], vidxintet,
                            tetDmInv_dev[tetidx],
                            vpos_target[tetvIdx_dev[4 * tetidx]], vpos_target[tetvIdx_dev[4 * tetidx + 1]], vpos_target[tetvIdx_dev[4 * tetidx + 2]], vpos_target[tetvIdx_dev[4 * tetidx + 3]],
                            f, H);
                    }
                }
            }

            __syncthreads();
            // log(N) add.
            for (unsigned int j = KernelConfig::VBD3D_THREAD_DIM_FORTET / 2; j > 0; j >>= 1)
            {
                if (tidx < j)
                {
                    real *Handf_other = H_and_f_shared + 12 * (tidx + j);
                    Handf[0] = Handf[0] + Handf_other[0];
                    Handf[1] = Handf[1] + Handf_other[1];
                    Handf[2] = Handf[2] + Handf_other[2];
                    Handf[3] = Handf[3] + Handf_other[3];
                    Handf[4] = Handf[4] + Handf_other[4];
                    Handf[5] = Handf[5] + Handf_other[5];
                    Handf[6] = Handf[6] + Handf_other[6];
                    Handf[7] = Handf[7] + Handf_other[7];
                    Handf[8] = Handf[8] + Handf_other[8];
                    Handf[9] = Handf[9] + Handf_other[9];
                    Handf[10] = Handf[10] + Handf_other[10];
                    Handf[11] = Handf[11] + Handf_other[11];
                }
                __syncthreads();
            }
            if (tidx == 0)
            {
                compute_lbs_derivatives(stiffness_dev[vidx], vpos_lbs[vidx], vpos_target[vidx], f, H);

                mat3_t Htmp(H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8]);
                vec3_t ftmp(f[0], f[1], f[2]);

                // update pos, prepos and preprepos.
                vitrpreprePos_dev[vidx] = vitrprePos_dev[vidx];
                vitrprePos_dev[vidx] = vpos_target[vidx];
                vec3_t vNew = vpos_target[vidx];

                if (NumericUtils::isApproxGreater(f[0] * f[0] + f[1] * f[1] + f[2] * f[2], 0.0))
                {
                    vec3_t descentDirection;
                    real stepSize = 1e-3;
                    if (NumericUtils::isApproxZero(glm::determinant(Htmp)))
                    {
                        descentDirection = ftmp;
                    }
                    else
                    {
                        descentDirection = stepSize * glm::inverse(Htmp) * ftmp;
                    }

                    vNew += descentDirection;
                }

                vpos_target[vidx] = vNew;
                if (itr_idx > 1)
                {
                    vpos_target[vidx] = itr_omega * (vpos_target[vidx] - vitrpreprePos_dev[vidx]) + vitrpreprePos_dev[vidx];
                }
            }
        }

        void predictPositions(VbdSceneDataGpu3D &data, real dt, cudaStream_t stream)
        {
            int vnum = data.vertex_num;
            dim3 block(KernelConfig::BLOCK_SIZE_1D, 1, 1);
            dim3 grid((vnum + KernelConfig::BLOCK_SIZE_1D - 1) / KernelConfig::BLOCK_SIZE_1D, 1, 1);
            kernel_initguess3d_ip<<<grid, block, 0, stream>>>(
                vnum,
                data.positions.data(),
                data.pre_positions.data(),
                data.itr_pre_positions.data(),
                data.itr_pre_pre_positions.data(),
                data.inertia.data(),
                data.velocities.data(),
                data.pre_velocities.data(),
                data.forces.data(),
                data.vertex_masses.data(),
                dt);
        }

        void solveTetrahedronConstraints(VbdSceneDataGpu3D &data, real dt, int itr_idx, real itr_omega, cudaStream_t stream)
        {
            for (int i = 0; i < data.color_num; i++)
            {
                int vnum = data.color_vertex_nums[i];
                dim3 block(KernelConfig::VBD3D_THREAD_DIM_FORTET, 1, 1);
                dim3 grid(vnum, 1, 1);
                kernel_iterationonce3d_ip<<<grid, block, 0, stream>>>(
                    itr_idx, itr_omega,
                    data.color_vertex_indices[i].data(), vnum,
                    data.tetrahedra_indices.data(), data.neibour_tetrahedra_nums.data(), data.neibour_tetrahedra_start_indices.data(), data.neibour_tetrahedra_indices.data(), data.vertex_indices_in_neibour_tetrahedra.data(),
                    data.vertex_masses.data(),
                    data.positions.data(), data.inertia.data(), data.itr_pre_positions.data(), data.itr_pre_pre_positions.data(),
                    dt,
                    data.tet_mu.data(), data.tet_lambda.data(), data.tet_volumes.data(), data.tet_kd.data(), data.tet_DmInv.data(), data.tet_FaInv.data());
            }
        }

        void updateVelocitiesAndPositions(VbdSceneDataGpu3D &data, real dt, cudaStream_t stream)
        {
            int vnum = data.vertex_num;
            dim3 block(KernelConfig::BLOCK_SIZE_1D, 1, 1);
            dim3 grid((vnum + KernelConfig::BLOCK_SIZE_1D - 1) / KernelConfig::BLOCK_SIZE_1D, 1, 1);
            kernel_updatevel3d<<<grid, block, 0, stream>>>(
                vnum,
                data.positions.data(),
                data.pre_positions.data(),
                data.velocities.data(),
                data.pre_velocities.data(),
                dt);
        }

        void solveLBSDynamicCorrection(VbdSceneDataGpu3D &data, control::LBSDataGpu3D &lbs_data, int itr_idx, real itr_omega, cudaStream_t stream)
        {
            for (size_t i = 0; i < data.color_num; i++)
            {
                int vnum = data.color_vertex_nums[i];
                dim3 block(KernelConfig::VBD3D_THREAD_DIM_FORTET, 1, 1);
                dim3 grid(vnum, 1, 1);
                kernel_solveLBSDynamicCorrection<<<grid, block, 0, stream>>>(
                    itr_idx, itr_omega,
                    data.color_vertex_indices[i].data(), vnum,
                    data.tetrahedra_indices.data(), data.neibour_tetrahedra_nums.data(), data.neibour_tetrahedra_start_indices.data(), data.neibour_tetrahedra_indices.data(), data.vertex_indices_in_neibour_tetrahedra.data(),
                    data.vertex_masses.data(),
                    lbs_data.position_target.data(), lbs_data.position_lbs.data(),
                    data.itr_pre_positions.data(), data.itr_pre_pre_positions.data(),
                    lbs_data.stiffness.data(),
                    data.tet_mu.data(), data.tet_lambda.data(), data.tet_volumes.data(), data.tet_kd.data(), data.tet_DmInv.data());
            }
        }
    }
}