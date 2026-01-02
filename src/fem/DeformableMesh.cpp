// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/fem/DeformableMesh.hpp"
#include "FSI_Simulator/utils/CudaArray.cuh"
#include "FSI_Simulator/control/LBSSampling.hpp"
#include "FSI_Simulator/control/LBSControlUtils.cuh"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <filesystem>
#include <set>

namespace fsi
{
    namespace fem
    {

        Mesh3D::Mesh3D(const SolidBodyConfig3D &config)
        {
            // 1. set id and physical properties
            m_id = config.mesh_path.empty() ? "generated_body" : std::filesystem::path(config.mesh_path).stem().string();

            m_density = config.density;
            m_youngs_modulus = config.youngs_modulus;
            m_poisson_ratio = config.poisson_ratio;
            m_kd = 1.0;

            LOG_INFO("Creating DeformableMesh '{}'...", m_id);

            // 2. load geometry data from file
            if (!config.mesh_path.empty())
            {
                loadFromFile(config.mesh_path);
            }
            else
            {
                LOG_WARN("Mesh '{}' has an empty mesh_path. It must be populated manually.", m_id);
            }

            // 3. apply initial transform
            applyInitialTransform(config);
            d_initial_positions.resize(h_initial_positions.size());
            d_initial_positions.upload(h_initial_positions);

            // 4. extract surface
            extractSurface();

            // 5. initialize LBS control
            if (config.lbs_control_config.cnum > 0)
            {
                is_lbs_control_enabled = true;
                cnum = config.lbs_control_config.cnum;
                center = vec3_t(config.translate[0], config.translate[1], config.translate[2]);
                d_lbs_shift.resize(cnum);
                d_lbs_rotation.resize(cnum);
                d_lbs_weight.resize(getNumVertices() * cnum);

                std::vector<real> lbs_dist;
                control::farthest_point_sampling(
                    getNumVertices(), h_initial_positions, h_tetrahedra,
                    cnum, ctrl_idx, lbs_dist,
                    config.lbs_control_config.lbs_distance_type,
                    config.lbs_control_config.random_first);

                CudaArray<real> d_lbs_dist(lbs_dist.size());
                d_lbs_dist.upload(lbs_dist);
                // compute mesh extent
                real mesh_extent = 0.0;
                for (int i = 0; i < 3; i++)
                {
                    auto [min_it, max_it] = std::minmax_element(h_initial_positions.begin(), h_initial_positions.end(), [i](const vec3_t &a, const vec3_t &b)
                                                                { return a[i] < b[i]; });
                    mesh_extent = std::max(mesh_extent, h_initial_positions[max_it - h_initial_positions.begin()][i] - h_initial_positions[min_it - h_initial_positions.begin()][i]);
                }
                real omega = config.lbs_control_config.omega * mesh_extent;
                control::compute_lbs_weight(
                    getNumVertices(),
                    cnum,
                    d_lbs_weight.data(), d_lbs_dist.data(),
                    omega,
                    nullptr);

                m_lbs_stiffness = config.lbs_control_config.stiffness;
            }

            LOG_INFO("DeformableMesh '{}' created successfully: {} vertices, {} tetrahedra, {} triangles.",
                     m_id, getNumVertices(), getNumTetrahedra(), getNumTriangles());
        }

        void Mesh3D::applyLBSControl(const std::vector<vec3_t> &lbs_shift, const std::vector<mat3_t> &lbs_rotation, vec3_t *lbs_position, int offset, cudaStream_t stream)
        {
            if (!is_lbs_control_enabled)
                return;

            ASSERT(lbs_shift.size() == cnum && lbs_rotation.size() == cnum, "LBS shift and rotation must have the same size as cnum.");

            d_lbs_shift.uploadAsync(lbs_shift, stream);
            d_lbs_rotation.uploadAsync(lbs_rotation, stream);

            control::compute_lbs_position(
                getNumVertices(), d_initial_positions.data(),
                cnum, lbs_position, offset,
                d_lbs_weight.data(), d_lbs_shift.data(),
                d_lbs_rotation.data(),
                center,
                stream);

            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        void Mesh3D::loadFromFile(const std::string &filepath)
        {
            std::filesystem::path path(filepath);
            if (!std::filesystem::exists(path))
            {
                ASSERT(false, "Mesh file not found: {}", filepath);
                return;
            }

            std::string extension = path.extension().string();

            if (extension == ".mesh")
            {
                io::MeshLoader3D loader;
                if (!loader.load(filepath))
                {
                    ASSERT(false, "Failed to load .mesh file: {}", filepath);
                    return;
                }

                auto vertices = loader.getVertices();
                h_initial_positions.resize(vertices.size());
                for (int i = 0; i < vertices.size(); i++)
                {
                    h_initial_positions[i] = vec3_t(vertices[i].x, vertices[i].y, vertices[i].z);
                }
                h_triangles = loader.getTriangles();
                h_tetrahedra = loader.getTetrahedra();
            }
            else
            {
                LOG_WARN("Loading from legacy format '{}'. Consider converting to .mesh format.", extension);
                if (extension == ".obj")
                {
                    ASSERT(false, "Loading from .obj format is not implemented yet.");
                }
                else
                {
                    ASSERT(false, "Unsupported mesh file format: {}", extension);
                }
            }
        }

        void Mesh3D::applyInitialTransform(const SolidBodyConfig3D &config)
        {
            if (getNumVertices() == 0)
                return;

            LOG_INFO("Applying initial transform to mesh '{}'...", m_id);

            glm::mat4 scale_mat = glm::scale(glm::vec3(config.scale[0], config.scale[1], config.scale[2]));

            glm::mat4 rotation_mat = glm::eulerAngleYXZ(
                glm::radians(static_cast<float>(config.rotate[1])), // Yaw
                glm::radians(static_cast<float>(config.rotate[0])), // Pitch
                glm::radians(static_cast<float>(config.rotate[2]))  // Roll
            );

            glm::mat4 translate_mat = glm::translate(glm::vec3(config.translate[0], config.translate[1], config.translate[2]));

            // SRT: Scale -> Rotate -> Translate
            glm::mat4 transform_mat = translate_mat * rotation_mat * scale_mat;

            // 2. apply transform to vertices
            for (auto &vertex : h_initial_positions)
            {
                glm::vec4 v4(vertex, 1.0f); // convert to homogeneous coordinates
                v4 = transform_mat * v4;
                vertex = glm::vec3(v4);
            }
        }

        void Mesh3D::extractSurface()
        {
            LOG_INFO("Extracting surface for mesh '{}'...", m_id);

            h_surface_triangles_local_indices.clear();

            std::vector<int> extracted_surface_tris;

            if (getNumTetrahedra() > 0)
            {
                std::map<std::array<unsigned int, 3>, std::pair<int, std::array<unsigned int, 3>>> face_counts;

                for (int i = 0; i < getNumTetrahedra(); ++i)
                {
                    const unsigned int *tet_nodes = &h_tetrahedra[i * 4];

                    std::array<unsigned int, 3> faces_of_tet[4] = {
                        {tet_nodes[0], tet_nodes[2], tet_nodes[1]},
                        {tet_nodes[0], tet_nodes[3], tet_nodes[2]},
                        {tet_nodes[0], tet_nodes[1], tet_nodes[3]},
                        {tet_nodes[1], tet_nodes[2], tet_nodes[3]}};

                    for (const auto &face_orig : faces_of_tet)
                    {
                        std::array<unsigned int, 3> face_sorted = face_orig;
                        std::sort(face_sorted.begin(), face_sorted.end());
                        face_counts[face_sorted].first++;
                        face_counts[face_sorted].second = face_orig;
                    }
                }

                for (const auto &pair : face_counts)
                {
                    if (pair.second.first == 1)
                    {
                        const auto &face = pair.second.second;
                        extracted_surface_tris.push_back(face[0]);
                        extracted_surface_tris.push_back(face[1]);
                        extracted_surface_tris.push_back(face[2]);
                    }
                }
                LOG_INFO("  - Extracted {} triangles from tetrahedral volume.", extracted_surface_tris.size() / 3);
            }

            if (getNumTriangles() > 0)
            {
                LOG_INFO("  - Appending {} co-dimensional triangles.", getNumTriangles());
                h_surface_triangles_local_indices.insert(
                    h_surface_triangles_local_indices.end(),
                    h_triangles.begin(),
                    h_triangles.end());
            }

            if (!extracted_surface_tris.empty())
            {
                h_surface_triangles_local_indices.insert(
                    h_surface_triangles_local_indices.end(),
                    extracted_surface_tris.begin(),
                    extracted_surface_tris.end());
            }

            LOG_INFO("Surface extraction complete. Total surface triangles for '{}': {}.",
                     m_id, h_surface_triangles_local_indices.size() / 3);

            h_surface_to_volume_map_local.clear();
            std::vector<bool> is_surface_vertex(getNumVertices(), false);
            for (int vertex_index : h_surface_triangles_local_indices)
            {
                is_surface_vertex[vertex_index] = true;
            }

            for (int i = 0; i < getNumVertices(); ++i)
            {
                if (is_surface_vertex[i])
                {
                    h_surface_to_volume_map_local.push_back(i);
                }
            }
            LOG_INFO("  - Found {} unique surface vertices for this mesh.", h_surface_to_volume_map_local.size());
        }

        Mesh2D::Mesh2D(const SolidBodyConfig2D &config)
        {
            // set id and physical properties
            m_id = config.mesh_path.empty() ? "generated_body" : std::filesystem::path(config.mesh_path).stem().string();

            m_density = config.density;
            m_youngs_modulus = config.youngs_modulus;
            m_poisson_ratio = config.poisson_ratio;

            LOG_INFO("Creating DeformableMesh '{}'...", m_id);

            // 2. load geometry data from file
            if (!config.mesh_path.empty())
            {
                loadFromFile(config.mesh_path);
            }
            else
            {
                LOG_WARN("Mesh '{}' has an empty mesh_path. It must be populated manually.", m_id);
            }

            // 3. apply initial transform
            applyInitialTransform(config);

            d_initial_positions.resize(getNumVertices());
            d_initial_positions.upload(h_initial_positions);

            // 4. extract edges
            extractEdges();

            // 5. initialize LBS control
            if (config.lbs_control_config.cnum > 0)
            {
                is_lbs_control_enabled = true;
                cnum = config.lbs_control_config.cnum;
                center = vec2_t(config.translate[0], config.translate[1]);
                d_lbs_shift.resize(cnum);
                d_lbs_rotation.resize(cnum);
                d_lbs_weight.resize(getNumVertices() * cnum);

                std::vector<real> lbs_dist;
                control::farthest_point_sampling(
                    getNumVertices(), h_initial_positions, h_triangles,
                    cnum, ctrl_idx, lbs_dist,
                    config.lbs_control_config.lbs_distance_type,
                    config.lbs_control_config.random_first);

                CudaArray<real> d_lbs_dist(lbs_dist.size());
                d_lbs_dist.upload(lbs_dist);
                // compute mesh extent
                real mesh_extent = 0.0;
                for (int i = 0; i < 2; i++)
                {
                    auto [min_it, max_it] = std::minmax_element(h_initial_positions.begin(), h_initial_positions.end(), [i](const vec2_t &a, const vec2_t &b)
                                                                { return a[i] < b[i]; });
                    mesh_extent = std::max(mesh_extent, h_initial_positions[max_it - h_initial_positions.begin()][i] - h_initial_positions[min_it - h_initial_positions.begin()][i]);
                }
                real omega = config.lbs_control_config.omega * mesh_extent;
                control::compute_lbs_weight(
                    getNumVertices(),
                    cnum,
                    d_lbs_weight.data(), d_lbs_dist.data(),
                    omega,
                    nullptr);

                m_lbs_stiffness = config.lbs_control_config.stiffness;
            }
        }

        void Mesh2D::applyLBSControl(const std::vector<vec2_t> &lbs_shift, const std::vector<real> &lbs_rotation, CudaArray<vec2_t> &lbs_position, int offset, cudaStream_t stream)
        {
            if (!is_lbs_control_enabled)
                return;

            ASSERT(lbs_shift.size() == cnum && lbs_rotation.size() == cnum, "LBS shift and rotation must have the same size as cnum.");

            d_lbs_shift.uploadAsync(lbs_shift, stream);
            d_lbs_rotation.uploadAsync(lbs_rotation, stream);
            // LOG_INFO("Applying LBS control to mesh '{}': cnum = {}, offset = {}, num_vertices = {}.", m_id, cnum, offset, getNumVertices());

            control::compute_lbs_position(
                getNumVertices(), d_initial_positions.data(),
                cnum, lbs_position.data(), offset,
                d_lbs_weight.data(), d_lbs_shift.data(),
                d_lbs_rotation.data(),
                center,
                stream);

            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        void Mesh2D::loadFromFile(const std::string &filepath)
        {
            std::filesystem::path path(filepath);
            if (!std::filesystem::exists(path))
            {
                ASSERT(false, "Mesh file not found: {}", filepath);
                return;
            }

            std::string extension = path.extension().string();

            if (extension == ".mesh")
            {
                io::MeshLoader2D loader;
                if (!loader.load(filepath))
                {
                    ASSERT(false, "Failed to load .mesh file: {}", filepath);
                    return;
                }

                auto vertices = loader.getVertices();
                h_initial_positions.resize(vertices.size());
                for (int i = 0; i < vertices.size(); i++)
                {
                    h_initial_positions[i] = vec2_t(vertices[i].x, vertices[i].y);
                }
                h_triangles = loader.getTriangles();
                h_edges = loader.getEdges();
            }
            else
            {
                LOG_WARN("Loading from legacy format '{}'. Consider converting to .mesh format.", extension);
                if (extension == ".obj")
                {
                    ASSERT(false, "Loading from .obj format is not implemented yet.");
                }
                else
                {
                    ASSERT(false, "Unsupported mesh file format: {}", extension);
                }
            }
        }

        void Mesh2D::applyInitialTransform(const SolidBodyConfig2D &config)
        {
            if (getNumVertices() == 0)
                return;

            LOG_INFO("Applying initial transform to mesh '{}'...", m_id);

            mat2_t scale_mat(
                static_cast<float>(config.scale[0]), 0.0f,
                0.0f, static_cast<float>(config.scale[1]));

            real angle_rad = glm::radians(static_cast<float>(config.rotate));
            mat2_t rotation_mat(
                cos(angle_rad), -sin(angle_rad),
                sin(angle_rad), cos(angle_rad));

            vec2_t translate = vec2_t(static_cast<float>(config.translate[0]), static_cast<float>(config.translate[1]));

            for (auto &vertex : h_initial_positions)
            {
                vertex = rotation_mat * (scale_mat * vertex) + translate;
            }
        }

        void Mesh2D::extractEdges()
        {
            LOG_INFO("Extracting edges for mesh '{}'...", m_id);

            h_surface_edges_local_indices.clear();

            std::set<std::array<unsigned int, 2>> edge_set;

            if (getNumEdges() > 0)
            {
                for (int i = 0; i < getNumEdges(); ++i)
                {
                    const unsigned int *edge_nodes = &h_edges[i * 2];
                    std::array<unsigned int, 2> edge = {edge_nodes[0], edge_nodes[1]};
                    std::sort(edge.begin(), edge.end());
                    edge_set.insert(edge);
                }
                LOG_INFO("  - Appending {} co-dimensional edges.", getNumTriangles());
            }

            if (getNumTriangles() > 0)
            {
                LOG_INFO("  - Extracted {} edges from boundary edges.", edge_set.size());
                std::map<std::array<unsigned int, 2>, std::pair<int, std::array<unsigned int, 2>>> edge_counts;

                for (int i = 0; i < getNumTriangles(); ++i)
                {
                    const unsigned int *tri_nodes = &h_triangles[i * 3];

                    std::array<unsigned int, 2> edges_of_tri[3] = {
                        {tri_nodes[0], tri_nodes[1]},
                        {tri_nodes[1], tri_nodes[2]},
                        {tri_nodes[2], tri_nodes[0]}};

                    for (const auto &edge_orig : edges_of_tri)
                    {
                        std::array<unsigned int, 2> edge_sorted = edge_orig;
                        std::sort(edge_sorted.begin(), edge_sorted.end());
                        edge_counts[edge_sorted].first++;
                        edge_counts[edge_sorted].second = edge_sorted;
                    }
                }

                for (const auto &pair : edge_counts)
                {
                    if (pair.second.first == 1)
                    {
                        const auto &edge = pair.second.second;
                        edge_set.insert(edge);
                    }
                }
            }

            for (const auto &edge : edge_set)
            {
                h_surface_edges_local_indices.push_back(edge[0]);
                h_surface_edges_local_indices.push_back(edge[1]);
            }

            LOG_INFO("Edge extraction complete. Total surface edges for '{}': {}.",
                     m_id, h_surface_edges_local_indices.size() / 2);

            h_surface_to_area_map_local.clear();
            std::vector<bool> is_surface_vertex(getNumVertices(), false);
            for (int i = 0; i < h_surface_edges_local_indices.size(); i += 2)
            {
                is_surface_vertex[h_surface_edges_local_indices[i]] = true;
                is_surface_vertex[h_surface_edges_local_indices[i + 1]] = true;
            }
            for (int i = 0; i < getNumVertices(); ++i)
            {
                if (is_surface_vertex[i])
                {
                    h_surface_to_area_map_local.push_back(i);
                }
            }
            LOG_INFO("  - Found {} unique surface vertices for this mesh.", h_surface_to_area_map_local.size());
        }

    } // namespace fem
} // namespace fsi