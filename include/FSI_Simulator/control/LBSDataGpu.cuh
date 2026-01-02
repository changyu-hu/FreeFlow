// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/common/CudaCommon.cuh"
#include "FSI_Simulator/common/Types.hpp"

#include "FSI_Simulator/utils/CudaArray.cuh"

namespace fsi
{
    namespace control
    {

        struct LBSDataGpu3D
        {
            CudaArray<vec3_t> position_rest;
            CudaArray<mat3_t> tet_DmInv;
            CudaArray<vec3_t> position_lbs;
            CudaArray<vec3_t> position_target;
            CudaArray<real> stiffness;

            CudaArray<real> elastic_energy_integral;
            CudaArray<real> lbs_energy_integral;
            CudaArray<real> elastic_gradient_integral;
            CudaArray<integer> elastic_gradient_map_begin_index;
            CudaArray<integer> elastic_gradient_map;
            CudaArray<real> elastic_gradient_value_ptr;

            CudaArray<real> elastic_hessian_integral;
            CudaArray<real> elastic_hessian_projection_integral;
            int elastic_hessian_nonzero_num_;
            CudaArray<integer> elastic_hessian_nonzero_map_begin_index;
            CudaArray<integer> elastic_hessian_nonzero_map;
            CudaArray<real> elastic_hessian_value_ptr;
            SparseMatrixXr elastic_hessian_, elastic_hessian_projection_, hessian_lbs_;
            CudaArray<integer> hessian_is_diagnal;
        };

        struct LBSDataGpu2D
        {
            CudaArray<vec2_t> position_rest;
            CudaArray<mat2_t> tri_DmInv;
            CudaArray<vec2_t> position_lbs;
            CudaArray<vec2_t> position_target;
            CudaArray<real> stiffness;

            CudaArray<real> elastic_energy_integral;
            CudaArray<real> lbs_energy_integral;
            CudaArray<real> elastic_gradient_integral;
            CudaArray<integer> elastic_gradient_map_begin_index;
            CudaArray<integer> elastic_gradient_map;
            CudaArray<real> elastic_gradient_value_ptr;

            CudaArray<real> elastic_hessian_integral;
            CudaArray<real> elastic_hessian_projection_integral;
            int elastic_hessian_nonzero_num_;
            CudaArray<integer> elastic_hessian_nonzero_map_begin_index;
            CudaArray<integer> elastic_hessian_nonzero_map;
            CudaArray<real> elastic_hessian_value_ptr;
            SparseMatrixXr elastic_hessian_, elastic_hessian_projection_, hessian_lbs_;
            CudaArray<integer> hessian_is_diagnal;
        };

    }
}