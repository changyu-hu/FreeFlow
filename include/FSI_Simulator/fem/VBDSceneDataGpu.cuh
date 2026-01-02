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
    namespace fem
    {

        struct VbdSceneDataGpu3D
        {
            int vertex_num;
            int tetrahedron_num;
            int triangle_num;
            int color_num;
            std::vector<int> color_vertex_nums;
            std::vector<CudaArray<unsigned int>> color_vertex_indices;

            CudaArray<vec3_t> positions;
            CudaArray<vec3_t> pre_positions;
            CudaArray<vec3_t> itr_pre_positions;
            CudaArray<vec3_t> itr_pre_pre_positions;

            CudaArray<vec3_t> pre_velocities;
            CudaArray<vec3_t> velocities;

            CudaArray<vec3_t> inertia;
            CudaArray<vec3_t> forces;

            CudaArray<unsigned int> tetrahedra_indices;
            CudaArray<unsigned int> triangles_indices;
            CudaArray<real> vertex_masses;

            CudaArray<unsigned int> neibour_tetrahedra_nums;
            CudaArray<unsigned int> neibour_tetrahedra_start_indices;
            CudaArray<unsigned int> neibour_tetrahedra_indices;
            CudaArray<unsigned int> vertex_indices_in_neibour_tetrahedra;

            CudaArray<real> tet_mu;
            CudaArray<real> tet_lambda;
            CudaArray<real> tet_volumes;
            CudaArray<real> tet_kd;
            CudaArray<mat3_t> tet_DmInv;
            CudaArray<mat3_t> tet_FaInv;
            CudaArray<real> tet_At;

            CudaArray<real> tri_mu;
            CudaArray<real> tri_lambda;
            CudaArray<real> tri_areas;
        };

        struct VbdSceneDataGpu2D
        {
            int vertex_num;
            int triangle_num;
            int edge_num;
            int color_num;
            std::vector<int> color_vertex_nums;
            std::vector<CudaArray<unsigned int>> color_vertex_indices;

            CudaArray<vec2_t> positions;
            CudaArray<vec2_t> pre_positions;
            CudaArray<vec2_t> itr_pre_positions;
            CudaArray<vec2_t> itr_pre_pre_positions;

            CudaArray<vec2_t> pre_velocities;
            CudaArray<vec2_t> velocities;

            CudaArray<vec2_t> inertia;
            CudaArray<vec2_t> forces;

            CudaArray<unsigned int> triangles_indices;
            CudaArray<unsigned int> edges_indices;
            CudaArray<real> vertex_masses;

            CudaArray<unsigned int> neibour_triangles_nums;
            CudaArray<unsigned int> neibour_triangles_start_indices;
            CudaArray<unsigned int> neibour_triangles_indices;
            CudaArray<unsigned int> vertex_indices_in_neibour_triangles;

            CudaArray<real> tri_mu;
            CudaArray<real> tri_lambda;
            CudaArray<real> tri_areas;
            CudaArray<real> tri_kd;
            CudaArray<mat2_t> tri_DmInv;
            CudaArray<mat2_t> tri_FaInv;
            CudaArray<real> tri_At;

            CudaArray<real> edge_stiffness;
            CudaArray<real> edge_rest_lengths;
            CudaArray<real> edge_kd;
        };

    } // namespace fem
} // namespace fsi