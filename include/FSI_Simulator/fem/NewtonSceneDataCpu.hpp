// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include <FSI_Simulator/common/Types.hpp>
#include <FSI_Simulator/utils/CudaArray.cuh>

namespace fsi
{
    namespace fem
    {

        struct NewtonSceneDataCpu
        {
            int vertex_num;
            int tetrahedron_num;
            int triangle_num;

            Matrix3Xr position;
            Matrix3Xr next_position;

            Matrix3Xr next_velocity;
            Matrix3Xr velocity;

            Matrix3Xr external_acceleration;

            Matrix4Xi tetrahedra_indices;
            Matrix3Xi triangles_indices;
            SparseMatrixXr mass_matrix;

            std::vector<real> tetMeshMu;
            std::vector<real> tetMeshLambda;
            std::vector<real> tetMeshKd;
            std::vector<real> tetVolume;
            std::vector<Matrix3r> tetDmInv;
            std::vector<Matrix12_9> tetAt;

            // CudaArray<real> tri_mu;
            // CudaArray<real> tri_lambda;
            // CudaArray<real> tri_areas;
        };

    }
}