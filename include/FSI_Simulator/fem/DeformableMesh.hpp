// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/common/Types.hpp"
#include "FSI_Simulator/fem/FemConfig.hpp"
#include "FSI_Simulator/io/MeshLoader.hpp"
#include "FSI_Simulator/utils/Logger.hpp"

#include <FSI_Simulator/utils/CudaArray.cuh>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <memory>

namespace fsi
{
    namespace fem
    {

        class Mesh3D
        {
        public:
            explicit Mesh3D(const SolidBodyConfig3D &config);

            ~Mesh3D() = default;

            Mesh3D(const Mesh3D &) = delete;
            Mesh3D &operator=(const Mesh3D &) = delete;
            Mesh3D(Mesh3D &&) = default;
            Mesh3D &operator=(Mesh3D &&) = default;

            // --- Public Getters ---
            const std::string &getId() const { return m_id; }
            int getNumVertices() const { return static_cast<int>(h_initial_positions.size()); }
            int getNumTetrahedra() const { return static_cast<int>(h_tetrahedra.size()) / 4; }
            int getNumTriangles() const { return static_cast<int>(h_triangles.size()) / 3; }
            int getNumControlPoint() const { return cnum; }
            std::vector<int> getControlPoint() const { return ctrl_idx; }

            // LBS control
            void applyLBSControl(const std::vector<vec3_t> &lbs_shift, const std::vector<mat3_t> &lbs_rotation, vec3_t *lbs_position, int offset, cudaStream_t stream = nullptr);

            std::vector<int> h_surface_triangles_local_indices; // surface triangles, indices are local to surface vertices
            std::vector<int> h_surface_to_volume_map_local;     // surface vertex -> this body's volume vertex map

            // --- id and physical properties ---
            std::string m_id;
            real m_density;
            real m_youngs_modulus;
            real m_poisson_ratio;
            real m_kd;

            // --- host-side geometry and topology data ---
            // these data are typically read-only after loading
            std::vector<vec3_t> h_initial_positions;
            std::vector<unsigned int> h_tetrahedra;
            std::vector<unsigned int> h_triangles;

            // LBS control
            bool is_lbs_control_enabled;
            int cnum;
            vec3_t center;
            real m_lbs_stiffness;
            std::vector<int> ctrl_idx;
            CudaArray<vec3_t> d_initial_positions;
            CudaArray<vec3_t> d_lbs_shift;
            CudaArray<mat3_t> d_lbs_rotation;
            CudaArray<real> d_lbs_weight;

        private:
            void loadFromFile(const std::string &filepath);

            void applyInitialTransform(const SolidBodyConfig3D &config);

            void extractSurface();
        };

        class Mesh2D
        {
        public:
            explicit Mesh2D(const SolidBodyConfig2D &config);
            ~Mesh2D() = default;

            Mesh2D(const Mesh2D &) = delete;
            Mesh2D &operator=(const Mesh2D &) = delete;
            Mesh2D(Mesh2D &&) = default;
            Mesh2D &operator=(Mesh2D &&) = default;

            const std::string &getId() const { return m_id; }
            int getNumVertices() const { return static_cast<int>(h_initial_positions.size()); }
            int getNumTriangles() const { return static_cast<int>(h_triangles.size()) / 3; }
            int getNumEdges() const { return static_cast<int>(h_edges.size()) / 2; }
            int getNumControlPoint() const { return cnum; }
            std::vector<int> getControlPoint() const { return ctrl_idx; }

            void applyLBSControl(const std::vector<vec2_t> &lbs_shift, const std::vector<real> &lbs_rotation, CudaArray<vec2_t> &lbs_position, int offset, cudaStream_t stream = nullptr);

            std::vector<int> h_surface_edges_local_indices;
            std::vector<int> h_surface_to_area_map_local;

            std::string m_id;
            real m_density;
            real m_youngs_modulus;
            real m_poisson_ratio;
            real m_kd;
            std::vector<vec2_t> h_initial_positions;
            std::vector<unsigned int> h_triangles;
            std::vector<unsigned int> h_edges;

            // LBS control
            bool is_lbs_control_enabled;
            int cnum;
            vec2_t center;
            real m_lbs_stiffness;
            std::vector<int> ctrl_idx;
            CudaArray<vec2_t> d_initial_positions;
            CudaArray<vec2_t> d_lbs_shift;
            CudaArray<real> d_lbs_rotation;
            CudaArray<real> d_lbs_weight;

        private:
            void loadFromFile(const std::string &filepath);
            void applyInitialTransform(const SolidBodyConfig2D &config);
            void extractEdges();
        };

    } // namespace fem
} // namespace fsi