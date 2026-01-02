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
        __global__ void kernel_initguess2d_ip(
            int vnum,
            vec2_t *vpos,
            vec2_t *vprepos,
            vec2_t *vitrprepos,
            vec2_t *vitrpreprepos,
            vec2_t *vinertia,
            vec2_t *vvel,
            vec2_t *vprevel,
            vec2_t *vforce,
            real *vmass,
            real dt)
        {
            int vidx = threadIdx.x + blockIdx.x * blockDim.x;
            if (vidx >= vnum)
                return;
            vec2_t at = (vvel[vidx] - vprevel[vidx]) / dt;
            vec2_t e = vforce[vidx] / vmass[vidx];
            // if (glm::length(e) > 10.0)
            //     printf("vidx: %d, vforce[vidx]: %f %f %f, acc[vidx]: %f %f %f\n", vidx, vforce[vidx][0], vforce[vidx][1], vforce[vidx][2], e[0], e[1], e[2]);
            real aextnorm = glm::length(e);
            if (aextnorm == 0.0)
            {
                e = vec2_t(0.0);
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

        __device__ void neohookean_H_and_f(
            real mu, real lambda, real kd, int vidxintri,
            real tri_area, const mat2_t DmInv_FaInv,
            const vec2_t &v0, const vec2_t &v1, const vec2_t &v2,
            real *H, real *f)
        {
            real dim = 2.0;
            real t = mu / lambda;
            real a = 2 * t + 1.0, b = 2 * t * (dim - 1.0) + dim, c = -t * dim;
            real delta = (-b + sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
            real alpha = (1.0 - 1.0 / (dim + delta)) * mu / lambda + 1.0;
            real Ds[4];
            Ds[0] = v0[0] - v2[0];
            Ds[2] = v1[0] - v2[0];
            Ds[1] = v0[1] - v2[1];
            Ds[3] = v1[1] - v2[1];

            real DmInvFaInv[4] = {
                DmInv_FaInv[0][0], DmInv_FaInv[1][0],
                DmInv_FaInv[0][1], DmInv_FaInv[1][1]};

            real F[4];
            // F = Ds * DmInvFaInv.
            F[0] = Ds[0] * DmInvFaInv[0] + Ds[2] * DmInvFaInv[1];
            F[2] = Ds[0] * DmInvFaInv[2] + Ds[2] * DmInvFaInv[3];
            F[1] = Ds[1] * DmInvFaInv[0] + Ds[3] * DmInvFaInv[1];
            F[3] = Ds[1] * DmInvFaInv[2] + Ds[3] * DmInvFaInv[3];

            real det_F = F[0] * F[3] - F[2] * F[1];
            real trace_FTF = F[0] * F[0] + F[1] * F[1] + F[2] * F[2] + F[3] * F[3];

            real dPhi_dF[4];
            real trace_FTF_add_delta = trace_FTF + delta;
            dPhi_dF[0] = mu * F[0] + lambda * (det_F - alpha) * F[3] - mu / trace_FTF_add_delta * F[0];
            dPhi_dF[1] = mu * F[1] + lambda * (det_F - alpha) * (-F[2]) - mu / trace_FTF_add_delta * F[1];
            dPhi_dF[2] = mu * F[2] + lambda * (det_F - alpha) * (-F[1]) - mu / trace_FTF_add_delta * F[2];
            dPhi_dF[3] = mu * F[3] + lambda * (det_F - alpha) * F[0] - mu / trace_FTF_add_delta * F[3];
            real d2Phi_dF2[16];
            // col1.
            d2Phi_dF2[0] = mu * (1.0 - 1.0 / trace_FTF_add_delta) + lambda * F[3] * F[3] + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[0] * F[0];
            d2Phi_dF2[1] = lambda * (-F[2]) * F[3] + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[1] * F[0];
            d2Phi_dF2[2] = lambda * (-F[1]) * F[3] + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[2] * F[0];
            d2Phi_dF2[3] = lambda * F[0] * F[3] + lambda * (det_F - alpha) + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[3] * F[0];
            // col2.
            d2Phi_dF2[4] = d2Phi_dF2[1];
            d2Phi_dF2[5] = mu * (1.0 - 1.0 / trace_FTF_add_delta) + lambda * (-F[2]) * (-F[2]) + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[1] * F[1];
            d2Phi_dF2[6] = lambda * (-F[1]) * (-F[2]) - lambda * (det_F - alpha) + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[2] * F[1];
            d2Phi_dF2[7] = lambda * F[0] * (-F[2]) + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[3] * F[1];
            // col3.
            d2Phi_dF2[8] = d2Phi_dF2[2];
            d2Phi_dF2[9] = d2Phi_dF2[6];
            d2Phi_dF2[10] = mu * (1.0 - 1.0 / trace_FTF_add_delta) + lambda * (-F[1]) * (-F[1]) + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[2] * F[2];
            d2Phi_dF2[11] = lambda * F[0] * (-F[1]) + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[3] * F[2];
            // col4.
            d2Phi_dF2[12] = d2Phi_dF2[3];
            d2Phi_dF2[13] = d2Phi_dF2[7];
            d2Phi_dF2[14] = d2Phi_dF2[11];
            d2Phi_dF2[15] = mu * (1.0 - 1.0 / trace_FTF_add_delta) + lambda * F[0] * F[0] + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[3] * F[3];

            real ms[3][2] = {
                {DmInvFaInv[0], DmInvFaInv[2]},
                {DmInvFaInv[1], DmInvFaInv[3]},
                {-DmInvFaInv[0] - DmInvFaInv[1], -DmInvFaInv[2] - DmInvFaInv[3]}};
            real m1 = ms[vidxintri][0], m2 = ms[vidxintri][1];
            // cal dE_dr and d2E_dr2.
            real dE_dr[2];
            dE_dr[0] = tri_area * (dPhi_dF[0] * m1 + dPhi_dF[2] * m2);
            dE_dr[1] = tri_area * (dPhi_dF[1] * m1 + dPhi_dF[3] * m2);

            real d2E_dr2[4];
            // col1.
            d2E_dr2[0] = tri_area * (d2Phi_dF2[0] * m1 * m1 + (d2Phi_dF2[2] + d2Phi_dF2[8]) * m1 * m2 + d2Phi_dF2[10] * m2 * m2);
            d2E_dr2[1] = tri_area * (d2Phi_dF2[4] * m1 * m1 + (d2Phi_dF2[6] + d2Phi_dF2[12]) * m1 * m2 + d2Phi_dF2[14] * m2 * m2);
            // col2.
            d2E_dr2[2] = d2E_dr2[1];
            d2E_dr2[3] = tri_area * (d2Phi_dF2[5] * m1 * m1 + (d2Phi_dF2[7] + d2Phi_dF2[13]) * m1 * m2 + d2Phi_dF2[15] * m2 * m2);

            // real damping = kd / dt;
            real damping = 0.0;
            H[0] += (1.0 + damping) * d2E_dr2[0];
            H[1] += (1.0 + damping) * d2E_dr2[1];
            H[2] += (1.0 + damping) * d2E_dr2[2];
            H[3] += (1.0 + damping) * d2E_dr2[3];
            // f[0] += -dE_dr[0] - damping * (d2E_dr2[0] * (vPos_dev[2 * vidx] - vprePos_dev[2 * vidx]) + d2E_dr2[2] * (vPos_dev[2 * vidx + 1] - vprePos_dev[2 * vidx + 1]));
            // f[1] += -dE_dr[1] - damping * (d2E_dr2[1] * (vPos_dev[2 * vidx] - vprePos_dev[2 * vidx]) + d2E_dr2[3] * (vPos_dev[2 * vidx + 1] - vprePos_dev[2 * vidx + 1]));
            f[0] += -dE_dr[0];
            f[1] += -dE_dr[1];
        }

        __device__ void neohookean_H_and_f_ref(
            real mu, real lambda, real kd,
            real triAera_dev, real *triDmInvFaInv_dev,
            int vidxintri,
            real *vpos_1, real *vpos_2, real *vpos_3,
            real *H, real *f)
        {
            real dim = 2.0;
            real t = mu / lambda;
            real a = 2 * t + 1.0, b = 2 * t * (dim - 1.0) + dim, c = -t * dim;
            real delta = (-b + sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
            real alpha = (1.0 - 1.0 / (dim + delta)) * mu / lambda + 1.0;

            real Ds[4];
            Ds[0] = vpos_1[0] - vpos_3[0];
            Ds[2] = vpos_2[0] - vpos_3[0];
            Ds[1] = vpos_1[1] - vpos_3[1];
            Ds[3] = vpos_2[1] - vpos_3[1];

            real *DmInvFaInv = triDmInvFaInv_dev;
            real F[4];
            // F = Ds * DmInvFaInv.
            F[0] = Ds[0] * DmInvFaInv[0] + Ds[2] * DmInvFaInv[1];
            F[2] = Ds[0] * DmInvFaInv[2] + Ds[2] * DmInvFaInv[3];
            F[1] = Ds[1] * DmInvFaInv[0] + Ds[3] * DmInvFaInv[1];
            F[3] = Ds[1] * DmInvFaInv[2] + Ds[3] * DmInvFaInv[3];

            real det_F = F[0] * F[3] - F[2] * F[1];
            real trace_FTF = F[0] * F[0] + F[1] * F[1] + F[2] * F[2] + F[3] * F[3];

            real dPhi_dF[4];
            real trace_FTF_add_delta = trace_FTF + delta;
            dPhi_dF[0] = mu * F[0] + lambda * (det_F - alpha) * F[3] - mu / trace_FTF_add_delta * F[0];
            dPhi_dF[1] = mu * F[1] + lambda * (det_F - alpha) * (-F[2]) - mu / trace_FTF_add_delta * F[1];
            dPhi_dF[2] = mu * F[2] + lambda * (det_F - alpha) * (-F[1]) - mu / trace_FTF_add_delta * F[2];
            dPhi_dF[3] = mu * F[3] + lambda * (det_F - alpha) * F[0] - mu / trace_FTF_add_delta * F[3];
            real d2Phi_dF2[16];
            // col1.
            d2Phi_dF2[0] = mu * (1.0 - 1.0 / trace_FTF_add_delta) + lambda * F[3] * F[3] + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[0] * F[0];
            d2Phi_dF2[1] = lambda * (-F[2]) * F[3] + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[1] * F[0];
            d2Phi_dF2[2] = lambda * (-F[1]) * F[3] + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[2] * F[0];
            d2Phi_dF2[3] = lambda * F[0] * F[3] + lambda * (det_F - alpha) + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[3] * F[0];
            // col2.
            d2Phi_dF2[4] = d2Phi_dF2[1];
            d2Phi_dF2[5] = mu * (1.0 - 1.0 / trace_FTF_add_delta) + lambda * (-F[2]) * (-F[2]) + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[1] * F[1];
            d2Phi_dF2[6] = lambda * (-F[1]) * (-F[2]) - lambda * (det_F - alpha) + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[2] * F[1];
            d2Phi_dF2[7] = lambda * F[0] * (-F[2]) + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[3] * F[1];
            // col3.
            d2Phi_dF2[8] = d2Phi_dF2[2];
            d2Phi_dF2[9] = d2Phi_dF2[6];
            d2Phi_dF2[10] = mu * (1.0 - 1.0 / trace_FTF_add_delta) + lambda * (-F[1]) * (-F[1]) + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[2] * F[2];
            d2Phi_dF2[11] = lambda * F[0] * (-F[1]) + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[3] * F[2];
            // col4.
            d2Phi_dF2[12] = d2Phi_dF2[3];
            d2Phi_dF2[13] = d2Phi_dF2[7];
            d2Phi_dF2[14] = d2Phi_dF2[11];
            d2Phi_dF2[15] = mu * (1.0 - 1.0 / trace_FTF_add_delta) + lambda * F[0] * F[0] + 2.0 * mu / (trace_FTF_add_delta * trace_FTF_add_delta) * F[3] * F[3];

            real ms[3][2] = {
                {DmInvFaInv[0], DmInvFaInv[2]},
                {DmInvFaInv[1], DmInvFaInv[3]},
                {-DmInvFaInv[0] - DmInvFaInv[1], -DmInvFaInv[2] - DmInvFaInv[3]}};
            real m1 = ms[vidxintri][0], m2 = ms[vidxintri][1];
            // cal dE_dr and d2E_dr2.
            real dE_dr[2];
            dE_dr[0] = triAera_dev * (dPhi_dF[0] * m1 + dPhi_dF[2] * m2);
            dE_dr[1] = triAera_dev * (dPhi_dF[1] * m1 + dPhi_dF[3] * m2);
            real d2E_dr2[4];
            // col1.
            d2E_dr2[0] = triAera_dev * (d2Phi_dF2[0] * m1 * m1 + (d2Phi_dF2[2] + d2Phi_dF2[8]) * m1 * m2 + d2Phi_dF2[10] * m2 * m2);
            d2E_dr2[1] = triAera_dev * (d2Phi_dF2[4] * m1 * m1 + (d2Phi_dF2[6] + d2Phi_dF2[12]) * m1 * m2 + d2Phi_dF2[14] * m2 * m2);
            // col2.
            d2E_dr2[2] = d2E_dr2[1];
            d2E_dr2[3] = triAera_dev * (d2Phi_dF2[5] * m1 * m1 + (d2Phi_dF2[7] + d2Phi_dF2[13]) * m1 * m2 + d2Phi_dF2[15] * m2 * m2);
            // cal contribution to H and f.
            H[0] += d2E_dr2[0];
            H[1] += d2E_dr2[1];
            H[2] += d2E_dr2[2];
            H[3] += d2E_dr2[3];
            f[0] += -dE_dr[0];
            f[1] += -dE_dr[1];
        }

        __global__ void test_neohookean_H_and_f()
        {
            real mu = 1.01;
            real lambda = 3.043;
            real kd = 0.0;
            real triAera_dev = 0.137;
            vec2_t vpos_1(0.073, 0.041);
            vec2_t vpos_2(0.123, 0.0);
            vec2_t vpos_3(0.0, 0.1111);
            real DmInv[4] = {-10.011, 5.025, 9.035, -6.011};
            mat2_t DmInv_mat = mat2_t(
                DmInv[0], DmInv[2],
                DmInv[1], DmInv[3]);
            for (int i = 0; i < 3; i++)
            {
                int vidxintri = i;
                real H[4] = {0.0, 0.0, 0.0, 0.0};
                real f[2] = {0.0, 0.0};
                neohookean_H_and_f_ref(
                    mu, lambda, kd,
                    triAera_dev, DmInv,
                    vidxintri,
                    glm::value_ptr(vpos_1), glm::value_ptr(vpos_2), glm::value_ptr(vpos_3),
                    H, f);

                real H2[4] = {0.0, 0.0, 0.0, 0.0};
                real f2[2] = {0.0, 0.0};
                neohookean_H_and_f(
                    mu, lambda, kd, vidxintri,
                    triAera_dev, DmInv_mat,
                    vpos_1, vpos_2, vpos_3,
                    H2, f2);

                printf("vidxintri: %d\n", vidxintri);
                printf("H: %f %f %f %f\n", H[0], H[1], H[2], H[3]);
                printf("H2: %f %f %f %f\n", H2[0], H2[1], H2[2], H2[3]);
                printf("f: %f %f\n", f[0], f[1]);
                printf("f2: %f %f\n", f2[0], f2[1]);
                printf("\n");
            }
        }

        __global__ void kernel_iterationonce2d_ip(
            int itr_idx, real itr_omega,
            unsigned int *vindices, int vnum,
            unsigned int *trivIdx_dev, unsigned int *neitriNum_dev, unsigned int *neitriNumstart_dev, unsigned int *neitriIdx_dev, unsigned int *Idxinneitri_dev,
            real *vMass_dev,
            vec2_t *vpos,
            vec2_t *vadvectPos_dev,
            vec2_t *vitrprePos_dev,
            vec2_t *vitrpreprePos_dev,
            real dt,
            real *tri_mu_dev, real *tri_lambda_dev, real *tri_area, real *tri_kd_dev, mat2_t *tri_DmInv_dev, mat2_t *tri_FaInv_dev)
        {
            // shared memory for H and f.
            __shared__ __builtin_align__(16) real H_and_f_shared[KernelConfig::VBD2D_THREAD_DIM_FORTRI * 6];
            // every block <-> 1 vertex; every theard <-> 1 neitri.
            int bidx = blockIdx.x;
            int tidx = threadIdx.x;

            if (bidx >= vnum)
                return;

            int vidx = vindices[bidx];

            real *Handf = H_and_f_shared + 6 * tidx;
            real *H = Handf;
            real *f = H + 4;
            // set H and f of this idxingroup to 0.
            H[0] = 0.0;
            H[2] = 0.0;
            H[1] = 0.0;
            H[3] = 0.0;
            f[0] = 0.0;
            f[1] = 0.0;
            // VBD2D_THREAD_DIM_FORTRI tri each group, in sequence.
            int neitrinum = neitriNum_dev[vidx];
            if (tidx < neitrinum)
            {
                int triidx = neitriIdx_dev[neitriNumstart_dev[vidx] + tidx];
                int vidxintri = Idxinneitri_dev[neitriNumstart_dev[vidx] + tidx];
                neohookean_H_and_f(
                    tri_mu_dev[triidx], tri_lambda_dev[triidx], tri_kd_dev[triidx], vidxintri,
                    tri_area[triidx], tri_FaInv_dev[triidx] * tri_DmInv_dev[triidx],
                    vpos[trivIdx_dev[3 * triidx]], vpos[trivIdx_dev[3 * triidx + 1]], vpos[trivIdx_dev[3 * triidx + 2]],
                    H, f);
            }
            if (neitrinum > KernelConfig::VBD2D_THREAD_DIM_FORTRI)
            {
                // printf("vidx: %d, neitriNum: %d.\n", vidx, neitrinum);
                int extraSize = neitrinum - KernelConfig::VBD2D_THREAD_DIM_FORTRI;
                for (int i = 0; i < extraSize; i += KernelConfig::VBD2D_THREAD_DIM_FORTRI)
                {
                    int index = tidx + KernelConfig::VBD2D_THREAD_DIM_FORTRI + i;
                    if (index < neitrinum)
                    {
                        int triidx = neitriIdx_dev[neitriNumstart_dev[vidx] + index];
                        int vidxintri = Idxinneitri_dev[neitriNumstart_dev[vidx] + index];
                        neohookean_H_and_f(
                            tri_mu_dev[triidx], tri_lambda_dev[triidx], tri_kd_dev[triidx], vidxintri,
                            tri_area[triidx], tri_FaInv_dev[triidx] * tri_DmInv_dev[triidx],
                            vpos[trivIdx_dev[3 * triidx]], vpos[trivIdx_dev[3 * triidx + 1]], vpos[trivIdx_dev[3 * triidx + 2]],
                            H, f);
                    }
                }
            }

            __syncthreads();
            // log(N) add.
            for (unsigned int j = KernelConfig::VBD2D_THREAD_DIM_FORTRI / 2; j > 0; j >>= 1)
            {
                if (tidx < j)
                {
                    real *Handf_other = H_and_f_shared + 6 * (tidx + j);
                    Handf[0] = Handf[0] + Handf_other[0];
                    Handf[1] = Handf[1] + Handf_other[1];
                    Handf[2] = Handf[2] + Handf_other[2];
                    Handf[3] = Handf[3] + Handf_other[3];
                    Handf[4] = Handf[4] + Handf_other[4];
                    Handf[5] = Handf[5] + Handf_other[5];
                }
                __syncthreads();
            }
            if (tidx == 0)
            {
                real factor = vMass_dev[vidx] / dt / dt;
                mat2_t Htmp(H[0] + factor, H[1], H[2], H[3] + factor);
                vec2_t ftmp(f[0], f[1]);
                ftmp += -factor * (vpos[vidx] - vadvectPos_dev[vidx]);

                // update pos, prepos and preprepos.
                vitrpreprePos_dev[vidx] = vitrprePos_dev[vidx];
                vitrprePos_dev[vidx] = vpos[vidx];

                if (NumericUtils::isApproxGreater(ftmp[0] * ftmp[0] + ftmp[1] * ftmp[1], 0.0))
                {
                    vec2_t descentDirection;
                    real stepSize = 1e-3;
                    if (NumericUtils::isApproxZero(glm::determinant(Htmp)))
                    {
                        descentDirection = stepSize * ftmp;
                    }
                    else
                    {
                        descentDirection = glm::inverse(Htmp) * ftmp;
                        // if (glm::length(ftmp) > 10)
                        // {
                        //     printf("vidx: %d, H: %f %f %f %f, f: %f %f, descentDirection: %f %f\n", vidx, Htmp[0][0], Htmp[0][1], Htmp[1][0], Htmp[1][1], ftmp[0], ftmp[1], descentDirection[0], descentDirection[1]);
                        // }
                    }

                    vpos[vidx] += descentDirection;
                }

                if (itr_idx > 1)
                {
                    vpos[vidx] = itr_omega * (vpos[vidx] - vitrpreprePos_dev[vidx]) + vitrpreprePos_dev[vidx];
                }
            }
        }

        __global__ void kernel_updatevel2d(
            int vnum,
            vec2_t *vpos,
            vec2_t *vprepos,
            vec2_t *vvel,
            vec2_t *vprevel,
            real dt)
        {
            int vidx = threadIdx.x + blockIdx.x * blockDim.x;
            if (vidx >= vnum)
                return;
            vprevel[vidx] = vvel[vidx];
            vvel[vidx] = (vpos[vidx] - vprepos[vidx]) / dt;
        }

        __device__ inline void compute_lbs_derivatives2d(
            real stiffness,
            vec2_t vpos_lbs,
            vec2_t vpos_target,
            real *force_contrib_out,
            real *hessian_contrib_out)
        {
            force_contrib_out[0] += 2 * stiffness * (vpos_lbs[0] - vpos_target[0]);
            force_contrib_out[1] += 2 * stiffness * (vpos_lbs[1] - vpos_target[1]);

            hessian_contrib_out[0] += 2 * stiffness;
            hessian_contrib_out[2] += 2 * stiffness;
        }

        __global__ void kernel_solveLBSDynamicCorrection2d(
            int itr_idx, real itr_omega,
            unsigned int *vindices, int vnum,
            unsigned int *trivIdx_dev, unsigned int *neitriNum_dev, unsigned int *neitriNumstart_dev, unsigned int *neitriIdx_dev, unsigned int *Idxinneitri_dev,
            real *vMass_dev,
            vec2_t *vpos_target,
            vec2_t *vpos_lbs,
            vec2_t *vitrprePos_dev,
            vec2_t *vitrpreprePos_dev,
            real *stiffness_dev,
            real *tri_mu_dev, real *tri_lambda_dev, real *tri_area, real *tri_kd_dev, mat2_t *tri_DmInv)
        {
            __shared__ __builtin_align__(16) real H_and_f_shared[KernelConfig::VBD2D_THREAD_DIM_FORTRI * 6];
            // every block <-> 1 vertex; every theard <-> 1 neitri.
            int bidx = blockIdx.x;
            int tidx = threadIdx.x;

            if (bidx >= vnum)
                return;

            int vidx = vindices[bidx];

            real *Handf = H_and_f_shared + 6 * tidx;
            real *H = Handf;
            real *f = H + 4;
            // set H and f of this idxingroup to 0.
            H[0] = 0.0;
            H[2] = 0.0;
            H[1] = 0.0;
            H[3] = 0.0;
            f[0] = 0.0;
            f[1] = 0.0;
            // VBD2D_THREAD_DIM_FORTRI tri each group, in sequence.
            int neitrinum = neitriNum_dev[vidx];
            if (tidx < neitrinum)
            {
                int triidx = neitriIdx_dev[neitriNumstart_dev[vidx] + tidx];
                int vidxintri = Idxinneitri_dev[neitriNumstart_dev[vidx] + tidx];
                neohookean_H_and_f(
                    tri_mu_dev[triidx], tri_lambda_dev[triidx], tri_kd_dev[triidx], vidxintri,
                    tri_area[triidx], tri_DmInv[triidx],
                    vpos_target[trivIdx_dev[3 * triidx]], vpos_target[trivIdx_dev[3 * triidx + 1]], vpos_target[trivIdx_dev[3 * triidx + 2]],
                    H, f);
            }
            if (neitrinum > KernelConfig::VBD2D_THREAD_DIM_FORTRI)
            {
                // printf("vidx: %d, neitriNum: %d.\n", vidx, neitrinum);
                int extraSize = neitrinum - KernelConfig::VBD2D_THREAD_DIM_FORTRI;
                for (int i = 0; i < extraSize; i += KernelConfig::VBD2D_THREAD_DIM_FORTRI)
                {
                    int index = tidx + KernelConfig::VBD2D_THREAD_DIM_FORTRI + i;
                    if (index < neitrinum)
                    {
                        int triidx = neitriIdx_dev[neitriNumstart_dev[vidx] + index];
                        int vidxintri = Idxinneitri_dev[neitriNumstart_dev[vidx] + index];
                        neohookean_H_and_f(
                            tri_mu_dev[triidx], tri_lambda_dev[triidx], tri_kd_dev[triidx], vidxintri,
                            tri_area[triidx], tri_DmInv[triidx],
                            vpos_target[trivIdx_dev[3 * triidx]], vpos_target[trivIdx_dev[3 * triidx + 1]], vpos_target[trivIdx_dev[3 * triidx + 2]],
                            H, f);
                    }
                }
            }

            __syncthreads();
            // log(N) add.
            for (unsigned int j = KernelConfig::VBD2D_THREAD_DIM_FORTRI / 2; j > 0; j >>= 1)
            {
                if (tidx < j)
                {
                    real *Handf_other = H_and_f_shared + 6 * (tidx + j);
                    Handf[0] = Handf[0] + Handf_other[0];
                    Handf[1] = Handf[1] + Handf_other[1];
                    Handf[2] = Handf[2] + Handf_other[2];
                    Handf[3] = Handf[3] + Handf_other[3];
                    Handf[4] = Handf[4] + Handf_other[4];
                    Handf[5] = Handf[5] + Handf_other[5];
                }
                __syncthreads();
            }
            if (tidx == 0)
            {
                compute_lbs_derivatives2d(stiffness_dev[vidx], vpos_lbs[vidx], vpos_target[vidx], f, H);

                mat2_t Htmp(H[0], H[1], H[2], H[3]);
                vec2_t ftmp(f[0], f[1]);

                // update pos, prepos and preprepos.
                vitrpreprePos_dev[vidx] = vitrprePos_dev[vidx];
                vitrprePos_dev[vidx] = vpos_target[vidx];

                if (NumericUtils::isApproxGreater(ftmp[0] * ftmp[0] + ftmp[1] * ftmp[1], 0.0))
                {
                    vec2_t descentDirection;
                    real stepSize = 1e-3;
                    if (NumericUtils::isApproxZero(glm::determinant(Htmp)))
                    {
                        descentDirection = stepSize * ftmp;
                    }
                    else
                    {
                        descentDirection = glm::inverse(Htmp) * ftmp;
                    }

                    vpos_target[vidx] += descentDirection;
                }

                if (itr_idx > 1)
                {
                    vpos_target[vidx] = itr_omega * (vpos_target[vidx] - vitrpreprePos_dev[vidx]) + vitrpreprePos_dev[vidx];
                }
            }
        }

        void predictPositions(VbdSceneDataGpu2D &data, real dt, cudaStream_t stream)
        {
            int vnum = data.vertex_num;
            dim3 block(KernelConfig::BLOCK_SIZE_1D, 1, 1);
            dim3 grid((vnum + KernelConfig::BLOCK_SIZE_1D - 1) / KernelConfig::BLOCK_SIZE_1D, 1, 1);
            kernel_initguess2d_ip<<<grid, block, 0, stream>>>(
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
        void solveConstraints(VbdSceneDataGpu2D &data, real dt, int itr_idx, real itr_omega, cudaStream_t stream)
        {
            // test_neohookean_H_and_f<<<1,1,0,stream>>>();
            for (int i = 0; i < data.color_num; i++)
            {
                int vnum = data.color_vertex_nums[i];
                dim3 block(KernelConfig::VBD2D_THREAD_DIM_FORTRI, 1, 1);
                dim3 grid(vnum, 1, 1);
                kernel_iterationonce2d_ip<<<grid, block, 0, stream>>>(
                    itr_idx, itr_omega,
                    data.color_vertex_indices[i].data(), vnum,
                    data.triangles_indices.data(), data.neibour_triangles_nums.data(), data.neibour_triangles_start_indices.data(), data.neibour_triangles_indices.data(), data.vertex_indices_in_neibour_triangles.data(),
                    data.vertex_masses.data(),
                    data.positions.data(), data.inertia.data(), data.itr_pre_positions.data(), data.itr_pre_pre_positions.data(),
                    dt,
                    data.tri_mu.data(), data.tri_lambda.data(), data.tri_areas.data(), data.tri_kd.data(), data.tri_DmInv.data(), data.tri_FaInv.data());
            }
        }
        void updateVelocitiesAndPositions(VbdSceneDataGpu2D &data, real dt, cudaStream_t stream)
        {
            int vnum = data.vertex_num;
            dim3 block(KernelConfig::BLOCK_SIZE_1D, 1, 1);
            dim3 grid((vnum + KernelConfig::BLOCK_SIZE_1D - 1) / KernelConfig::BLOCK_SIZE_1D, 1, 1);
            kernel_updatevel2d<<<grid, block, 0, stream>>>(
                vnum,
                data.positions.data(),
                data.pre_positions.data(),
                data.velocities.data(),
                data.pre_velocities.data(),
                dt);
        }

        void solveLBSDynamicCorrection(VbdSceneDataGpu2D &data, control::LBSDataGpu2D &lbs_data, int itr_idx, real itr_omega, cudaStream_t stream)
        {
            for (size_t i = 0; i < data.color_num; i++)
            {
                int vnum = data.color_vertex_nums[i];
                dim3 block(KernelConfig::VBD2D_THREAD_DIM_FORTRI, 1, 1);
                dim3 grid(vnum, 1, 1);
                kernel_solveLBSDynamicCorrection2d<<<grid, block, 0, stream>>>(
                    itr_idx, itr_omega,
                    data.color_vertex_indices[i].data(), vnum,
                    data.triangles_indices.data(), data.neibour_triangles_nums.data(), data.neibour_triangles_start_indices.data(), data.neibour_triangles_indices.data(), data.vertex_indices_in_neibour_triangles.data(),
                    data.vertex_masses.data(),
                    lbs_data.position_target.data(), lbs_data.position_lbs.data(), data.itr_pre_positions.data(), data.itr_pre_pre_positions.data(),
                    lbs_data.stiffness.data(),
                    data.tri_mu.data(), data.tri_lambda.data(), data.tri_areas.data(), data.tri_kd.data(), data.tri_DmInv.data());
            }
        }
    }
}
