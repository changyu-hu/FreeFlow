// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/common/CudaCommon.cuh"
#include "FSI_Simulator/common/Types.hpp"

namespace fsi
{

    namespace control
    {

        void compute_lbs_weight(
            int vnum,
            int cnum,
            real *lbs_weight,
            real *lbs_dist,
            real stiffness,
            cudaStream_t stream);

        void compute_lbs_position(
            int vnum, vec3_t *vpos,
            int cnum, vec3_t *v_lbs,
            int offset,
            real *lbs_weight,
            vec3_t *lbs_shift,
            mat3_t *lbs_rotation,
            vec3_t center,
            cudaStream_t stream);

        void compute_lbs_position(
            int vnum, vec2_t *vpos,
            int cnum, vec2_t *v_lbs,
            int offset,
            real *lbs_weight,
            vec2_t *lbs_shift,
            real *lbs_rotation,
            vec2_t center,
            cudaStream_t stream);

        void update_target_position(
            int vnum, vec3_t *vpos,
            int tetnum, unsigned int *tetvIdx_dev, mat3_t *tetDmInvFaInv_dev,
            cudaStream_t stream);

        void update_target_position(
            int vnum, vec2_t *vpos,
            int tetnum, unsigned int *tetvIdx_dev, mat2_t *tetDmInvFaInv_dev,
            cudaStream_t stream);

        void iterationonce3d_quasistatic(
            int itr_idx, real itr_omega,
            unsigned int *vindices, int vnum,
            unsigned int *tetvIdx_dev, unsigned int *neitetNum_dev, unsigned int *neitetNumstart_dev, unsigned int *neitetIdx_dev, unsigned int *Idxinneitet_dev,
            vec3_t *vpos,
            vec3_t *vitrprePos_dev,
            vec3_t *vitrpreprePos_dev,
            real dt,
            real *mass,
            vec3_t *vlbs, real stiffness,
            real *tet_mu_dev, real *tet_lambda_dev, real *tetVolume_dev, real *tet_kd_dev, mat3_t *tetDmInvFaInv_dev,
            cudaStream_t stream);
    }

}