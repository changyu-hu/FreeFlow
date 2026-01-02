// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/io/MeshLoader.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <stdexcept> // For std::stoi, std::stod

namespace fsi
{
    namespace io
    {

        bool MeshLoader3D::load(const std::string &filepath)
        {
            clear();

            if (!std::filesystem::exists(filepath))
            {
                LOG_ERROR("Mesh file not found: {}", filepath);
                return false;
            }

            std::ifstream file(filepath);
            if (!file.is_open())
            {
                LOG_ERROR("Failed to open mesh file: {}", filepath);
                return false;
            }

            LOG_INFO("Loading unified mesh file: {}", filepath);

            try
            {
                std::string line;

                // --- parse header ---
                std::getline(file, line);
                std::stringstream ss_version(line);
                std::string keyword;
                ss_version >> keyword;
                ASSERT(keyword == "MeshVersionFormatted", "Expected 'MeshVersionFormatted' keyword.");
                ss_version >> m_version;

                std::getline(file, line);
                std::stringstream ss_dim(line);
                ss_dim >> keyword;
                ASSERT(keyword == "Dimension", "Expected 'Dimension' keyword.");
                ss_dim >> m_dimension;

                ASSERT(m_dimension == 3, "Only 3D meshes are supported in MeshLoader3D.");

                LOG_INFO("  - Mesh format version: {}, Dimension: {}", m_version, m_dimension);

                // --- parse data blocks ---
                while (std::getline(file, line))
                {
                    // skip empty lines
                    if (line.empty() || line.find_first_not_of(" \t\n\v\f\r") == std::string::npos)
                    {
                        continue;
                    }

                    if (line == "Vertices")
                    {
                        std::getline(file, line); // read count line
                        int count = std::stoi(line);
                        LOG_INFO("  - Reading {} vertices...", count);
                        m_vertices.reserve(count);
                        for (int i = 0; i < count; ++i)
                        {
                            std::getline(file, line);
                            std::stringstream ss(line);
                            glm::dvec3 v;
                            ss >> v.x >> v.y >> v.z;
                            m_vertices.push_back(v);
                        }
                    }
                    else if (line == "Triangles")
                    {
                        std::getline(file, line); // read count line
                        int count = std::stoi(line);
                        LOG_INFO("  - Reading {} triangles...", count);
                        m_triangles.reserve(count * 3);
                        for (int i = 0; i < count; ++i)
                        {
                            std::getline(file, line);
                            std::stringstream ss(line);
                            unsigned int v0, v1, v2;
                            ss >> v0 >> v1 >> v2;
                            m_triangles.push_back(v0 - 1);
                            m_triangles.push_back(v1 - 1);
                            m_triangles.push_back(v2 - 1);
                        }
                    }
                    else if (line == "Tetrahedra")
                    {
                        std::getline(file, line); // read count line
                        int count = std::stoi(line);
                        LOG_INFO("  - Reading {} tetrahedra...", count);
                        m_tetrahedra.reserve(count * 4);
                        for (int i = 0; i < count; ++i)
                        {
                            std::getline(file, line);
                            std::stringstream ss(line);
                            unsigned int v0, v1, v2, v3;
                            ss >> v0 >> v1 >> v2 >> v3;
                            m_tetrahedra.push_back(v0 - 1);
                            m_tetrahedra.push_back(v1 - 1);
                            m_tetrahedra.push_back(v2 - 1);
                            m_tetrahedra.push_back(v3 - 1);
                        }
                    }
                    else
                    {
                        LOG_WARN("Unknown keyword in mesh file: '{}'. Skipping line.", line);
                    }
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("An error occurred while parsing mesh file '{}': {}", filepath, e.what());
                clear();
                return false;
            }

            LOG_INFO("Finished loading mesh file. Vertices: {}, Triangles: {}, Tetrahedra: {}",
                     m_vertices.size(), m_triangles.size() / 3, m_tetrahedra.size() / 4);

            return true;
        }

        void MeshLoader3D::clear()
        {
            m_version = 0;
            m_dimension = 0;
            m_vertices.clear();
            m_triangles.clear();
            m_tetrahedra.clear();
        }

        bool MeshLoader2D::load(const std::string &filepath)
        {
            clear();

            if (!std::filesystem::exists(filepath))
            {
                LOG_ERROR("Mesh file not found: {}", filepath);
                return false;
            }

            std::ifstream file(filepath);
            if (!file.is_open())
            {
                LOG_ERROR("Failed to open mesh file: {}", filepath);
                return false;
            }

            LOG_INFO("Loading unified mesh file: {}", filepath);

            try
            {
                std::string line;

                // --- parse header ---
                std::getline(file, line);
                std::stringstream ss_version(line);
                std::string keyword;
                ss_version >> keyword;
                ASSERT(keyword == "MeshVersionFormatted", "Expected 'MeshVersionFormatted' keyword.");
                ss_version >> m_version;

                std::getline(file, line);
                std::stringstream ss_dim(line);
                ss_dim >> keyword;
                ASSERT(keyword == "Dimension", "Expected 'Dimension' keyword.");
                ss_dim >> m_dimension;

                ASSERT(m_dimension == 2, "Only 2D meshes are supported in MeshLoader2D.");

                LOG_INFO("  - Mesh format version: {}, Dimension: {}", m_version, m_dimension);

                // --- parse data blocks ---
                while (std::getline(file, line))
                {
                    // skip empty lines
                    if (line.empty() || line.find_first_not_of(" \t\n\v\f\r") == std::string::npos)
                    {
                        continue;
                    }

                    if (line == "Vertices")
                    {
                        std::getline(file, line); // read count line
                        int count = std::stoi(line);
                        LOG_INFO("  - Reading {} vertices...", count);
                        m_vertices.reserve(count);
                        for (int i = 0; i < count; ++i)
                        {
                            std::getline(file, line);
                            std::stringstream ss(line);
                            glm::dvec2 v;
                            ss >> v.x >> v.y;
                            m_vertices.push_back(v);
                        }
                    }
                    else if (line == "Edges")
                    {
                        std::getline(file, line); // read count line
                        int count = std::stoi(line);
                        LOG_INFO("  - Reading {} edges...", count);
                        m_edges.reserve(count * 2);
                        for (int i = 0; i < count; ++i)
                        {
                            std::getline(file, line);
                            std::stringstream ss(line);
                            unsigned int v0, v1;
                            ss >> v0 >> v1;
                            m_edges.push_back(v0 - 1);
                            m_edges.push_back(v1 - 1);
                        }
                    }
                    else if (line == "Triangles")
                    {
                        std::getline(file, line); // read count line
                        int count = std::stoi(line);
                        LOG_INFO("  - Reading {} triangles...", count);
                        m_triangles.reserve(count * 3);
                        for (int i = 0; i < count; ++i)
                        {
                            std::getline(file, line);
                            std::stringstream ss(line);
                            unsigned int v0, v1, v2;
                            ss >> v0 >> v1 >> v2;
                            m_triangles.push_back(v0 - 1);
                            m_triangles.push_back(v1 - 1);
                            m_triangles.push_back(v2 - 1);
                        }
                    }
                    else
                    {
                        LOG_WARN("Unknown keyword in mesh file: '{}'. Skipping line.", line);
                    }
                }
            }
            catch (const std::exception &e)
            {
                LOG_ERROR("An error occurred while parsing mesh file '{}': {}", filepath, e.what());
                clear();
                return false;
            }
            LOG_INFO("Finished loading mesh file. Vertices: {}, Edges: {}, Triangles: {}",
                     m_vertices.size(), m_edges.size() / 2, m_triangles.size() / 3);
            return true;
        }

        void MeshLoader2D::clear()
        {
            m_version = 0;
            m_dimension = 0;
            m_vertices.clear();
            m_edges.clear();
            m_triangles.clear();
        }

    } // namespace io
} // namespace fsi