// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/common/Types.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <string>

namespace fsi
{
    namespace io
    {

        class MeshLoader3D
        {
        public:
            MeshLoader3D() = default;
            bool load(const std::string &filepath);

            const std::vector<glm::dvec3> &getVertices() const { return m_vertices; }
            const std::vector<unsigned int> &getTriangles() const { return m_triangles; }
            const std::vector<unsigned int> &getTetrahedra() const { return m_tetrahedra; }
            int getDimension() const { return m_dimension; }

        private:
            void clear();

            int m_version = 0;
            int m_dimension = 0;
            std::vector<glm::dvec3> m_vertices;
            std::vector<unsigned int> m_triangles;
            std::vector<unsigned int> m_tetrahedra;
        };

        class MeshLoader2D
        {
        public:
            MeshLoader2D() = default;
            bool load(const std::string &filepath);
            const std::vector<glm::dvec2> &getVertices() const { return m_vertices; }
            const std::vector<unsigned int> &getEdges() const { return m_edges; }
            const std::vector<unsigned int> &getTriangles() const { return m_triangles; }
            int getDimension() const { return m_dimension; }

        private:
            void clear();

            int m_version = 0;
            int m_dimension = 0;
            std::vector<glm::dvec2> m_vertices;
            std::vector<unsigned int> m_edges;
            std::vector<unsigned int> m_triangles;
        };

    } // namespace io
} // namespace fsi