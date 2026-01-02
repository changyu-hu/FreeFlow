// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/io/ObjLoader.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include <filesystem>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

namespace fsi
{
    namespace io
    {

        bool ObjLoader::load(const std::string &filepath)
        {
            clear();

            if (!std::filesystem::exists(filepath))
            {
                LOG_ERROR("OBJ file not found: {}", filepath);
                return false;
            }

            tinyobj::ObjReaderConfig reader_config;
            reader_config.mtl_search_path = std::filesystem::path(filepath).parent_path().string();
            reader_config.triangulate = true;

            tinyobj::ObjReader reader;

            if (!reader.ParseFromFile(filepath, reader_config))
            {
                if (!reader.Error().empty())
                {
                    LOG_ERROR("TinyObjLoader failed to parse {}: {}", filepath, reader.Error());
                }
                return false;
            }

            if (!reader.Warning().empty())
            {
                LOG_WARN("TinyObjLoader warning while parsing {}: {}", filepath, reader.Warning());
            }

            const auto &attrib = reader.GetAttrib();
            const auto &shapes = reader.GetShapes();
            // const auto& materials = reader.GetMaterials();

            LOG_INFO("Loading OBJ file: {}", filepath);

            for (const auto &shape : shapes)
            {
                size_t index_offset = 0;
                for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
                {
                    size_t num_verts_in_face = shape.mesh.num_face_vertices[f];
                    ASSERT(num_verts_in_face == 3, "Expected triangulated mesh from tinyobjloader.");

                    for (size_t v = 0; v < num_verts_in_face; ++v)
                    {
                        tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                        m_vertices.emplace_back(
                            attrib.vertices[3 * idx.vertex_index + 0],
                            attrib.vertices[3 * idx.vertex_index + 1],
                            attrib.vertices[3 * idx.vertex_index + 2]);

                        if (idx.normal_index >= 0)
                        {
                            m_normals.emplace_back(
                                attrib.normals[3 * idx.normal_index + 0],
                                attrib.normals[3 * idx.normal_index + 1],
                                attrib.normals[3 * idx.normal_index + 2]);
                        }

                        if (idx.texcoord_index >= 0)
                        {
                            m_texcoords.emplace_back(
                                attrib.texcoords[2 * idx.texcoord_index + 0],
                                attrib.texcoords[2 * idx.texcoord_index + 1]);
                        }

                        m_indices.push_back(static_cast<unsigned int>(m_indices.size()));
                    }
                    index_offset += num_verts_in_face;
                }
            }

            LOG_INFO("Finished loading OBJ. Final mesh: {} vertices, {} triangles.", m_vertices.size(), m_indices.size() / 3);
            return true;
        }

        void ObjLoader::clear()
        {
            m_vertices.clear();
            m_normals.clear();
            m_texcoords.clear();
            m_indices.clear();
        }

    } // namespace io
} // namespace fsi