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

        class ObjLoader
        {
        public:
            ObjLoader() = default;

            bool load(const std::string &filepath);

            const std::vector<glm::vec3> &getVertices() const { return m_vertices; }
            const std::vector<glm::vec3> &getNormals() const { return m_normals; }
            const std::vector<glm::vec2> &getTexCoords() const { return m_texcoords; }
            const std::vector<unsigned int> &getIndices() const { return m_indices; }

        private:
            void clear();

            std::vector<glm::vec3> m_vertices;
            std::vector<glm::vec3> m_normals;
            std::vector<glm::vec2> m_texcoords;
            std::vector<unsigned int> m_indices;
        };

    } // namespace io
} // namespace fsi