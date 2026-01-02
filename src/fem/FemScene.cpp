// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/fem/FemScene.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include "FSI_Simulator/graph/Mcs.hpp"
#include "FSI_Simulator/graph/TetMeshVertexGraph.hpp"
#include "FSI_Simulator/control/LBSControlUtils.cuh"
#include "FSI_Simulator/common/SparseMatrixUtils.hpp"
#include "FSI_Simulator/utils/StringUtils.hpp"
#include "FSI_Simulator/fsi/FsiCouplingUtils.cuh"

namespace fsi
{
    namespace fem
    {

        void FemScene3D::addMesh(std::unique_ptr<Mesh3D> mesh)
        {
            if (mesh)
            {
                m_meshes.push_back(std::move(mesh));
            }
        }

        void FemScene3D::finalizeAndUpload()
        {
            ASSERT(!m_meshes.empty(), "Cannot finalize an empty FEM scene.");
            LOG_INFO("Finalizing FEM scene and merging {} meshes...", m_meshes.size());

            // --- 1. total scene stats ---
            size_t total_volume_vertices = 0;
            size_t total_surface_vertices = 0;
            size_t total_tets = 0;
            size_t total_triangles = 0;
            size_t total_surface_tris = 0;

            // surface info
            std::vector<std::vector<int>> surface_to_volume_maps_local;
            std::vector<std::map<int, int>> volume_to_surface_maps_local;

            m_ctrl_verts_start_idx.resize(m_meshes.size() + 1, 0);

            for (int i = 0; i < m_meshes.size(); i++)
            {
                const auto &mesh = m_meshes[i];
                total_volume_vertices += mesh->getNumVertices();
                total_tets += mesh->getNumTetrahedra();
                total_triangles += mesh->getNumTriangles();

                // --- extract and count surface info ---
                std::vector<bool> is_surface_vertex(mesh->getNumVertices(), false);
                for (int tri_v_idx : mesh->h_surface_triangles_local_indices)
                {
                    is_surface_vertex[tri_v_idx] = true;
                }

                std::vector<int> s_to_v_map;
                std::map<int, int> v_to_s_map;
                int current_surface_v_idx = 0;
                for (int i = 0; i < mesh->getNumVertices(); ++i)
                {
                    if (is_surface_vertex[i])
                    {
                        s_to_v_map.push_back(i);                 // surf_idx -> vol_idx
                        v_to_s_map[i] = current_surface_v_idx++; // vol_idx -> surf_idx
                    }
                }

                surface_to_volume_maps_local.push_back(s_to_v_map);
                volume_to_surface_maps_local.push_back(v_to_s_map);

                total_surface_vertices += s_to_v_map.size();
                total_surface_tris += mesh->h_surface_triangles_local_indices.size() / 3;

                m_ctrl_verts_start_idx[i + 1] = m_ctrl_verts_start_idx[i] + mesh->getNumControlPoint();
            }
            LOG_INFO("Total scene stats: {} volume vertices, {} tets, {} triangles, {} surface vertices, {} surface triangles.",
                     total_volume_vertices, total_tets, total_triangles, total_surface_vertices, total_surface_tris);

            // --- 2. prepare merged data ---
            std::vector<vec3_t> merged_positions_flat(total_volume_vertices);
            std::vector<real> merged_vertex_masses(total_volume_vertices, 0.0);
            std::vector<real> merged_lbs_stiffness(total_volume_vertices, 0.0);

            std::vector<unsigned int> merged_tets(total_tets * 4);
            std::vector<real> merged_tet_mu(total_tets);
            std::vector<real> merged_tet_lambda(total_tets);
            std::vector<real> merged_tet_kd(total_tets);
            std::vector<mat3_t> merged_tet_DmInv(total_tets);
            std::vector<real> merged_tet_volume(total_tets);
            std::vector<Matrix12_9> merged_tet_At(total_tets);

            std::vector<unsigned int> merged_tris(total_triangles * 3);
            std::vector<real> merged_tri_mu(total_triangles);
            std::vector<real> merged_tri_lambda(total_triangles);
            std::vector<real> merged_tri_area(total_triangles);

            std::vector<unsigned int> merged_surf_tris(total_surface_tris * 3);
            std::vector<vec3_t> merged_surf_pos_flat(total_surface_vertices);
            std::vector<unsigned int> merged_surf_to_vol_map(total_surface_vertices);

            m_tet_FaInv.resize(total_tets);

            // --- 3. tranverse meshes and merge data ---
            size_t vol_vertex_offset = 0;
            size_t tet_offset = 0;
            size_t tri_offset = 0;

            size_t surf_vertex_offset = 0;
            size_t surf_tri_offset = 0;
            size_t mesh_idx = 0;

            m_verts_start_idx.resize(m_meshes.size());
            m_tet_start_idx.resize(m_meshes.size());

            for (const auto &mesh : m_meshes)
            {
                const size_t num_verts = mesh->getNumVertices();
                const size_t num_tets = mesh->getNumTetrahedra();
                const size_t num_tris = mesh->getNumTriangles();

                // --- a. merge vertex positions ---
                for (size_t i = 0; i < num_verts; ++i)
                {
                    const size_t merged_idx = vol_vertex_offset + i;
                    merged_positions_flat[merged_idx] = mesh->h_initial_positions[i];
                    merged_lbs_stiffness[merged_idx] = mesh->m_lbs_stiffness;
                }

                // --- b. merge tets, adjust indices ---
                for (size_t i = 0; i < num_tets; ++i)
                {
                    const size_t merged_idx = tet_offset + i;
                    for (int j = 0; j < 4; ++j)
                    {
                        merged_tets[merged_idx * 4 + j] = mesh->h_tetrahedra[i * 4 + j] + vol_vertex_offset;
                    }
                    real E = mesh->m_youngs_modulus;
                    real nu = mesh->m_poisson_ratio;
                    real kd = mesh->m_kd;
                    merged_tet_mu[merged_idx] = E / (2.0 * (1.0 + nu));
                    merged_tet_lambda[merged_idx] = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
                    merged_tet_kd[merged_idx] = kd;

                    auto v0 = mesh->h_initial_positions[mesh->h_tetrahedra[i * 4 + 0]];
                    auto e1 = mesh->h_initial_positions[mesh->h_tetrahedra[i * 4 + 1]] - v0;
                    auto e2 = mesh->h_initial_positions[mesh->h_tetrahedra[i * 4 + 2]] - v0;
                    auto e3 = mesh->h_initial_positions[mesh->h_tetrahedra[i * 4 + 3]] - v0;
                    mat3_t Dm = glm::transpose(mat3_t(e1, e2, e3));
                    mat3_t DmInv = glm::inverse(Dm);
                    merged_tet_DmInv[merged_idx] = DmInv;

                    real volume = std::abs(glm::determinant(Dm)) / 6.0;
                    if (NumericUtils::isApproxEqual(volume, 0.0))
                    {
                        LOG_CRITICAL("Tetrahedron {} has zero volume.", merged_idx);
                    }
                    merged_tet_volume[merged_idx] = volume;

                    real density = mesh->m_density;
                    real mass = density * volume;
                    merged_vertex_masses[mesh->h_tetrahedra[i * 4 + 0] + vol_vertex_offset] += mass / 4.0;
                    merged_vertex_masses[mesh->h_tetrahedra[i * 4 + 1] + vol_vertex_offset] += mass / 4.0;
                    merged_vertex_masses[mesh->h_tetrahedra[i * 4 + 2] + vol_vertex_offset] += mass / 4.0;
                    merged_vertex_masses[mesh->h_tetrahedra[i * 4 + 3] + vol_vertex_offset] += mass / 4.0;

                    const real ms[4][3] = {
                        {-DmInv[0][0] - DmInv[1][0] - DmInv[2][0], -DmInv[0][1] - DmInv[1][1] - DmInv[2][1], -DmInv[0][2] - DmInv[1][2] - DmInv[2][2]},
                        {DmInv[0][0], DmInv[0][1], DmInv[0][2]},
                        {DmInv[1][0], DmInv[1][1], DmInv[1][2]},
                        {DmInv[2][0], DmInv[2][1], DmInv[2][2]}};
                    Matrix12_9 Ate = Matrix12_9::Zero();
                    for (int i = 0; i < 3; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            for (int k = 0; k < 4; k++)
                            {
                                int row = i + j * 3;
                                int col = i + k * 3;
                                real val = ms[k][j];
                                Ate(col, row) = val;
                            }
                        }
                    }
                    merged_tet_At[merged_idx] = Ate;

                    m_tet_FaInv[merged_idx] = glm::identity<mat3_t>();
                }

                // --- c. merge triangles, adjust indices ---
                for (size_t i = 0; i < num_tris; ++i)
                {
                    const size_t merged_idx = tri_offset + i;
                    for (int j = 0; j < 3; ++j)
                    {
                        merged_tris[merged_idx * 3 + j] = mesh->h_triangles[i * 3 + j] + vol_vertex_offset;
                    }
                    auto v0 = mesh->h_initial_positions[mesh->h_triangles[i * 3 + 0]];
                    auto e1 = mesh->h_initial_positions[mesh->h_triangles[i * 3 + 1]] - v0;
                    auto e2 = mesh->h_initial_positions[mesh->h_triangles[i * 3 + 2]] - v0;
                    real area = 0.5 * glm::length(glm::cross(e1, e2));
                    if (NumericUtils::isApproxEqual(area, 0.0))
                    {
                        LOG_CRITICAL("Triangle {} has zero area.", merged_idx);
                    }
                    merged_tri_area[merged_idx] = area;

                    real density = mesh->m_density;
                    real mass = density * area;
                    merged_vertex_masses[mesh->h_triangles[i * 3 + 0] + vol_vertex_offset] += mass / 3.0;
                    merged_vertex_masses[mesh->h_triangles[i * 3 + 1] + vol_vertex_offset] += mass / 3.0;
                    merged_vertex_masses[mesh->h_triangles[i * 3 + 2] + vol_vertex_offset] += mass / 3.0;

                    real E = mesh->m_youngs_modulus;
                    real nu = mesh->m_poisson_ratio;
                    merged_tri_mu[merged_idx] = E / (2.0 * (1.0 + nu));
                    merged_tri_lambda[merged_idx] = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
                }

                // --- merge surface triangles, adjust indices ---
                const auto &local_s_to_v_map = surface_to_volume_maps_local[mesh_idx];
                const auto &local_v_to_s_map = volume_to_surface_maps_local[mesh_idx];

                for (size_t i = 0; i < local_s_to_v_map.size(); ++i)
                {
                    int vol_idx = local_s_to_v_map[i];
                    const auto &pos = mesh->h_initial_positions[vol_idx];
                    size_t merged_surf_idx = surf_vertex_offset + i;
                    merged_surf_pos_flat[merged_surf_idx] = pos;
                }

                for (size_t i = 0; i < mesh->h_surface_triangles_local_indices.size() / 3; ++i)
                {
                    size_t merged_tri_idx = surf_tri_offset + i;
                    for (int j = 0; j < 3; ++j)
                    {
                        int local_vol_idx = mesh->h_surface_triangles_local_indices[i * 3 + j];
                        int local_surf_idx = local_v_to_s_map.at(local_vol_idx);
                        merged_surf_tris[merged_tri_idx * 3 + j] = local_surf_idx + surf_vertex_offset;
                    }
                }

                for (size_t i = 0; i < local_s_to_v_map.size(); ++i)
                {
                    int local_vol_idx = local_s_to_v_map[i];
                    size_t merged_surf_idx = surf_vertex_offset + i;
                    merged_surf_to_vol_map[merged_surf_idx] = local_vol_idx + vol_vertex_offset;
                }

                m_verts_start_idx[mesh_idx] = vol_vertex_offset;
                m_tet_start_idx[mesh_idx] = tet_offset;

                auto cp = mesh->getControlPoint();
                for (size_t i = 0; i < cp.size(); i++)
                {
                    m_ctrl_verts_idx.push_back(cp[i] + vol_vertex_offset);
                }

                // update offsets
                vol_vertex_offset += mesh->getNumVertices();
                surf_vertex_offset += local_s_to_v_map.size();
                tet_offset += mesh->getNumTetrahedra();
                tri_offset += mesh->getNumTriangles();
                surf_tri_offset += mesh->h_surface_triangles_local_indices.size() / 3;
                mesh_idx++;
            }

            // --- 4. compute tet neighbors ---
            std::vector<unsigned int> tet_neighbors_num(total_volume_vertices);
            std::vector<unsigned int> tet_neighbors_start_indices(total_volume_vertices + 1);
            std::vector<std::vector<unsigned int>> tmp_neitetIdx(total_volume_vertices);
            std::vector<std::vector<unsigned int>> tmp_Idxinneitet(total_volume_vertices);
            for (int i = 0; i < total_tets; i++)
            {
                tmp_neitetIdx[merged_tets[4 * i]].push_back(i);
                tmp_neitetIdx[merged_tets[4 * i + 1]].push_back(i);
                tmp_neitetIdx[merged_tets[4 * i + 2]].push_back(i);
                tmp_neitetIdx[merged_tets[4 * i + 3]].push_back(i);
                tmp_Idxinneitet[merged_tets[4 * i]].push_back(0);
                tmp_Idxinneitet[merged_tets[4 * i + 1]].push_back(1);
                tmp_Idxinneitet[merged_tets[4 * i + 2]].push_back(2);
                tmp_Idxinneitet[merged_tets[4 * i + 3]].push_back(3);
            }

            tet_neighbors_start_indices[0] = 0;
            for (int i = 0; i < total_volume_vertices; i++)
            {
                tet_neighbors_num[i] = tmp_neitetIdx[i].size();
                tet_neighbors_start_indices[i + 1] = tet_neighbors_start_indices[i] + tet_neighbors_num[i];
            }

            int tol = tet_neighbors_start_indices[total_volume_vertices];
            std::vector<unsigned int> neitetIdx_host(tol);
            std::vector<unsigned int> Idxinneitet_host(tol);
            for (int i = 0; i < total_volume_vertices; i++)
            {
                for (int j = 0; j < tmp_neitetIdx[i].size(); j++)
                {
                    neitetIdx_host[tet_neighbors_start_indices[i] + j] = tmp_neitetIdx[i][j];
                    Idxinneitet_host[tet_neighbors_start_indices[i] + j] = tmp_Idxinneitet[i][j];
                }
            }

            ////////////////////////////////////////////
            // Gradient related arrays.
            ////////////////////////////////////////////

            std::vector<std::vector<integer>> elastic_gradient_map_;
            // Assemble gradient map.
            {
                elastic_gradient_map_.clear();
                elastic_gradient_map_.resize(total_volume_vertices);
                for (integer e = 0; e < total_tets; ++e)
                {
                    for (integer i = 0; i < 4; ++i)
                    {
                        const integer dof_map_i = merged_tets[4 * e + i];
                        elastic_gradient_map_[dof_map_i].push_back(4 * e + i);
                    }
                }
            }

            // Assemble the nonzero structures in elastic energy Hessian and its projection.
            std::vector<std::vector<integer>> elastic_hessian_nonzero_map_;
            {
                std::vector<Eigen::Triplet<real>> elastic_hess_nonzeros;
                for (integer e = 0; e < total_tets; ++e)
                {
                    for (integer i = 0; i < 4; ++i)
                        for (integer j = 0; j < 4; ++j)
                            for (integer di = 0; di < 3; ++di)
                                for (integer dj = 0; dj < 3; ++dj)
                                {
                                    const integer row_idx = merged_tets[4 * e + i] * 3 + di;
                                    const integer col_idx = merged_tets[4 * e + j] * 3 + dj;
                                    elastic_hess_nonzeros.emplace_back(row_idx, col_idx, 1.0);
                                }
                }
                m_lbs_data.elastic_hessian_ = FromTriplet(3 * total_volume_vertices, 3 * total_volume_vertices, elastic_hess_nonzeros);
                // Rest assured that this is deep copy.
                m_lbs_data.elastic_hessian_projection_ = m_lbs_data.elastic_hessian_;

                m_lbs_data.elastic_hessian_nonzero_num_ = static_cast<integer>(m_lbs_data.elastic_hessian_.nonZeros());
                elastic_hessian_nonzero_map_.resize(m_lbs_data.elastic_hessian_nonzero_num_);
                for (integer e = 0; e < total_tets; ++e)
                {
                    for (integer i = 0; i < 4; ++i)
                        for (integer j = 0; j < 4; ++j)
                            for (integer di = 0; di < 3; ++di)
                                for (integer dj = 0; dj < 3; ++dj)
                                {
                                    const integer row_idx = merged_tets[4 * e + i] * 3 + di;
                                    const integer col_idx = merged_tets[4 * e + j] * 3 + dj;
                                    const integer k = &m_lbs_data.elastic_hessian_.coeffRef(row_idx, col_idx) - m_lbs_data.elastic_hessian_.valuePtr();
                                    elastic_hessian_nonzero_map_[k].push_back({144 * e + 12 * (i * 3 + di) + j * 3 + dj});
                                }
                }
            }

            std::vector<integer> elastic_gradient_map_begin_index(total_volume_vertices + 1, 0);
            std::vector<integer> elastic_gradient_map;
            for (integer k = 0; k < total_volume_vertices; ++k)
            {
                const auto &mapping = elastic_gradient_map_[k];
                const integer mapping_size = static_cast<integer>(mapping.size());
                elastic_gradient_map_begin_index[k + 1] = elastic_gradient_map_begin_index[k] + mapping_size;

                for (integer m = 0; m < mapping_size; ++m)
                {
                    elastic_gradient_map.push_back(mapping[m]);
                }
            }

            ////////////////////////////////////////////
            // Hessian related arrays.
            ////////////////////////////////////////////
            std::vector<integer> elastic_hessian_nonzero_map_begin_index(m_lbs_data.elastic_hessian_nonzero_num_ + 1, 0);
            std::vector<integer> elastic_hessian_nonzero_map;
            // std::vector<integer> is_diagonal(m_lbs_data.elastic_hessian_nonzero_num_, 0);

            for (integer k = 0; k < m_lbs_data.elastic_hessian_nonzero_num_; ++k)
            {
                const auto &mapping = elastic_hessian_nonzero_map_[k];
                const integer mapping_size = static_cast<integer>(mapping.size());
                elastic_hessian_nonzero_map_begin_index[k + 1] = elastic_hessian_nonzero_map_begin_index[k] + mapping_size;

                for (integer m = 0; m < mapping_size; ++m)
                {
                    elastic_hessian_nonzero_map.push_back(mapping[m]);
                }
            }

            VectorXr stiffness = VectorXr(3 * total_volume_vertices);
            for (integer v = 0; v < total_volume_vertices; ++v)
            {
                stiffness(3 * v + 0) = 2 * merged_lbs_stiffness[v];
                stiffness(3 * v + 1) = 2 * merged_lbs_stiffness[v];
                stiffness(3 * v + 2) = 2 * merged_lbs_stiffness[v];
            }
            m_lbs_data.hessian_lbs_ = FromDiagonal(stiffness);

            // --- 5. color the mesh ---
            GAIA::GraphColoring::TetMeshVertexGraph tetMeshVertexGraph;
            tetMeshVertexGraph.fromCoMesh3d(merged_tets.data(), merged_tris.data(), total_volume_vertices, total_tets, total_triangles);
            GAIA::GraphColoring::Mcs mcs(tetMeshVertexGraph);
            mcs.color();
            mcs.convertToColoredCategories();
            auto colors = mcs.get_categories();
            m_gpu_data.color_num = colors.size();
            m_gpu_data.color_vertex_nums.resize(colors.size());
            m_gpu_data.color_vertex_indices.resize(colors.size());

            LOG_INFO("Successfully set colors for {} meshes. Total colors: {}", m_meshes.size(), colors.size());
            LOG_INFO("Vertices in each color: ");

            for (size_t i = 0; i < colors.size(); i++)
            {
                m_gpu_data.color_vertex_nums[i] = colors[i].size();
                m_gpu_data.color_vertex_indices[i].resize(colors[i].size());
                m_gpu_data.color_vertex_indices[i].upload(colors[i]);
                LOG_INFO("Color {}: {} vertices", i, colors[i].size());
            }

            // --- 6. set scene data ---
            m_total_vertices = total_volume_vertices;
            m_total_tetrahedra = total_tets;
            m_total_triangles = total_triangles;
            m_total_surface_vertices = total_surface_vertices;
            m_total_surface_tris = total_surface_tris;

            // --- 7. allocate and upload data to GPU ---
            LOG_INFO("Uploading merged scene data to GPU...");
            m_gpu_data.positions.resize(m_total_vertices);
            m_gpu_data.positions.upload(merged_positions_flat);
            // LOG_INFO("merged_positions_flat: {}", utils::join(merged_positions_flat, ", "));
            m_gpu_data.pre_positions.resize(m_total_vertices);
            m_gpu_data.pre_positions.setZero();
            m_gpu_data.itr_pre_positions.resize(m_total_vertices);
            m_gpu_data.itr_pre_positions.setZero();
            m_gpu_data.itr_pre_pre_positions.resize(m_total_vertices);
            m_gpu_data.itr_pre_pre_positions.setZero();

            m_gpu_data.tetrahedra_indices.resize(m_total_tetrahedra * 4);
            m_gpu_data.tetrahedra_indices.upload(merged_tets);
            m_gpu_data.neibour_tetrahedra_nums.resize(m_total_vertices);
            m_gpu_data.neibour_tetrahedra_nums.upload(tet_neighbors_num);
            m_gpu_data.neibour_tetrahedra_start_indices.resize(m_total_vertices + 1);
            m_gpu_data.neibour_tetrahedra_start_indices.upload(tet_neighbors_start_indices);
            m_gpu_data.neibour_tetrahedra_indices.resize(tol);
            m_gpu_data.neibour_tetrahedra_indices.upload(neitetIdx_host);
            m_gpu_data.vertex_indices_in_neibour_tetrahedra.resize(tol);
            m_gpu_data.vertex_indices_in_neibour_tetrahedra.upload(Idxinneitet_host);

            m_gpu_data.triangles_indices.resize(m_total_triangles * 3);
            m_gpu_data.triangles_indices.upload(merged_tris);

            m_gpu_data.vertex_masses.resize(m_total_vertices);
            m_gpu_data.vertex_masses.upload(merged_vertex_masses);

            m_gpu_data.tet_mu.resize(m_total_tetrahedra);
            m_gpu_data.tet_mu.upload(merged_tet_mu);
            m_gpu_data.tet_lambda.resize(m_total_tetrahedra);
            m_gpu_data.tet_lambda.upload(merged_tet_lambda);
            m_gpu_data.tet_kd.resize(m_total_tetrahedra);
            m_gpu_data.tet_kd.upload(merged_tet_kd);
            m_gpu_data.tet_volumes.resize(m_total_tetrahedra);
            m_gpu_data.tet_volumes.upload(merged_tet_volume);
            m_gpu_data.tet_DmInv.resize(m_total_tetrahedra);
            m_gpu_data.tet_DmInv.upload(merged_tet_DmInv);
            m_gpu_data.tet_FaInv.resize(m_total_tetrahedra);
            m_gpu_data.tet_FaInv.upload(m_tet_FaInv);

            m_gpu_data.tet_At.resize(m_total_tetrahedra * 108);
            std::vector<real> tmp_tet_At(m_total_tetrahedra * 108);
            for (int i = 0; i < m_total_tetrahedra; i++)
            {
                memcpy(&tmp_tet_At[i * 108], merged_tet_At[i].data(), sizeof(real) * 108);
            }
            m_gpu_data.tet_At.upload(tmp_tet_At);

            m_gpu_data.tri_mu.resize(m_total_triangles);
            m_gpu_data.tri_mu.upload(merged_tri_mu);
            m_gpu_data.tri_lambda.resize(m_total_triangles);
            m_gpu_data.tri_lambda.upload(merged_tri_lambda);
            m_gpu_data.tri_areas.resize(m_total_triangles);
            m_gpu_data.tri_areas.upload(merged_tri_area);

            m_coupling_data.surface_positions.resize(m_total_surface_vertices * 3);
            m_coupling_data.surface_positions.setZero();
            m_coupling_data.surface_velocities.resize(m_total_surface_vertices * 3);
            m_coupling_data.surface_velocities.setZero();
            m_coupling_data.surface_forces.resize(m_total_surface_vertices * 3);
            m_coupling_data.surface_forces.setZero();
            m_coupling_data.surface_elements_indices.resize(m_total_surface_tris * 3);
            m_coupling_data.surface_elements_indices.upload(merged_surf_tris);
            m_coupling_data.d_surface_to_volume_map.resize(m_total_surface_vertices);
            m_coupling_data.d_surface_to_volume_map.upload(merged_surf_to_vol_map);

            m_gpu_data.velocities.resize(m_total_vertices);
            m_gpu_data.velocities.setZero();
            m_gpu_data.pre_velocities.resize(m_total_vertices);
            m_gpu_data.pre_velocities.setZero();
            m_gpu_data.inertia.resize(m_total_vertices);
            m_gpu_data.inertia.setZero();
            m_gpu_data.forces.resize(m_total_vertices);
            m_gpu_data.forces.setZero();

            m_gpu_data.vertex_num = m_total_vertices;
            m_gpu_data.tetrahedron_num = m_total_tetrahedra;
            m_gpu_data.triangle_num = m_total_triangles;

            m_lbs_data.position_rest.resize(m_total_vertices);
            m_lbs_data.position_rest.upload(merged_positions_flat);
            m_lbs_data.tet_DmInv.resize(m_total_tetrahedra);
            m_lbs_data.tet_DmInv.upload(merged_tet_DmInv);
            m_lbs_data.position_lbs.resize(m_total_vertices);
            m_lbs_data.position_lbs.upload(merged_positions_flat);
            m_lbs_data.position_target.resize(m_total_vertices);
            m_lbs_data.position_target.setZero();
            m_lbs_data.stiffness.resize(m_total_vertices);
            m_lbs_data.stiffness.upload(merged_lbs_stiffness);
            m_lbs_data.elastic_energy_integral.resize(m_total_tetrahedra);
            m_lbs_data.lbs_energy_integral.resize(m_total_vertices);

            m_lbs_data.elastic_gradient_integral.resize(m_total_tetrahedra * 12);
            m_lbs_data.elastic_gradient_map_begin_index.resize(m_total_vertices + 1);
            m_lbs_data.elastic_gradient_map_begin_index.upload(elastic_gradient_map_begin_index);
            m_lbs_data.elastic_gradient_map.resize(elastic_gradient_map.size());
            m_lbs_data.elastic_gradient_map.upload(elastic_gradient_map);
            m_lbs_data.elastic_gradient_value_ptr.resize(m_total_vertices * 3);
            m_lbs_data.elastic_gradient_value_ptr.setZero();

            m_lbs_data.elastic_hessian_integral.resize(m_total_tetrahedra * 144);
            m_lbs_data.elastic_hessian_projection_integral.resize(m_total_tetrahedra * 144);
            m_lbs_data.elastic_hessian_nonzero_map_begin_index.resize(m_lbs_data.elastic_hessian_nonzero_num_ + 1);
            m_lbs_data.elastic_hessian_nonzero_map_begin_index.upload(elastic_hessian_nonzero_map_begin_index);
            m_lbs_data.elastic_hessian_nonzero_map.resize(elastic_hessian_nonzero_map.size());
            m_lbs_data.elastic_hessian_nonzero_map.upload(elastic_hessian_nonzero_map);
            m_lbs_data.elastic_hessian_value_ptr.resize(m_lbs_data.elastic_hessian_nonzero_num_);
            m_lbs_data.elastic_hessian_value_ptr.setZero();
            // m_lbs_data.hessian_is_diagnal.resize(m_lbs_data.elastic_hessian_nonzero_num_);
            // m_lbs_data.hessian_is_diagnal.upload(is_diagonal);

            copyGlmToEigen(merged_positions_flat, m_cpu_data.position);
            copyVecToEigen4X(merged_tets, m_cpu_data.tetrahedra_indices);
            copyVecToEigen3X(merged_tris, m_cpu_data.triangles_indices);
            m_cpu_data.next_position.resize(3, m_total_vertices);
            m_cpu_data.next_position.setZero();
            m_cpu_data.external_acceleration.resize(3, m_total_vertices);
            m_cpu_data.external_acceleration.setZero();
            m_cpu_data.velocity.resize(3, m_total_vertices);
            m_cpu_data.velocity.setZero();
            m_cpu_data.next_velocity.resize(3, m_total_vertices);
            m_cpu_data.next_velocity.setZero();

            m_cpu_data.tetMeshMu = merged_tet_mu;
            m_cpu_data.tetMeshLambda = merged_tet_lambda;
            m_cpu_data.tetMeshKd = merged_tet_kd;
            m_cpu_data.tetVolume = merged_tet_volume;
            m_cpu_data.tetDmInv.resize(m_total_tetrahedra);
            for (int i = 0; i < m_total_tetrahedra; i++)
            {
                m_cpu_data.tetDmInv[i] = convertGlmToEigen(merged_tet_DmInv[i]);
            }
            m_cpu_data.tetAt = merged_tet_At;
            m_cpu_data.tetrahedron_num = m_total_tetrahedra;
            m_cpu_data.triangle_num = m_total_triangles;
            m_cpu_data.vertex_num = m_total_vertices;
            m_cpu_data.mass_matrix = FromDiagonal(merged_vertex_masses.data(), m_total_vertices);

            LOG_INFO("FEM scene finalized and ready for simulation.");
        }

        void FemScene3D::reset()
        {
            m_gpu_data.velocities.setZero();
            m_gpu_data.pre_velocities.setZero();
            m_gpu_data.forces.setZero();
            m_gpu_data.positions.copyFrom(m_lbs_data.position_rest);
            m_gpu_data.pre_positions.copyFrom(m_lbs_data.position_rest);
            m_gpu_data.tet_DmInv.copyFrom(m_lbs_data.tet_DmInv);
            m_tet_FaInv.assign(m_total_tetrahedra, glm::identity<mat3_t>());
            m_gpu_data.tet_FaInv.upload(m_tet_FaInv);
        }

        SolidGeometryProxy_Device<3> FemScene3D::getSurfaceProxyForFSI()
        {
            coupling::updateSurfaceStatesFromVolume(
                m_gpu_data,
                m_coupling_data,
                m_total_surface_vertices,
                0);

            return {
                m_coupling_data.surface_positions.data(),
                m_coupling_data.surface_velocities.data(),
                m_coupling_data.surface_elements_indices.data(),
                m_coupling_data.surface_forces.data(),
                static_cast<int>(m_total_surface_vertices),
                static_cast<int>(m_total_surface_tris)};
        }

        void FemScene3D::lazyApplyLBSControl(int mesh_id, const std::vector<vec3_t> &lbs_shift, const std::vector<mat3_t> &lbs_rotation, cudaStream_t stream)
        {
            m_meshes[mesh_id]->applyLBSControl(lbs_shift, lbs_rotation, m_lbs_data.position_lbs.data(), m_verts_start_idx[mesh_id], stream);
        }

        void FemScene3D::lazyApplyActiveStrain(int mesh_id, int tet_id, vec3_t dir, real magnitude, cudaStream_t stream)
        {
            vec3_t n = glm::normalize(dir);
            real amp = magnitude;

            vec3_t m = glm::cross(vec3_t(1.0f, 0.0f, 0.0f), n);

            if (glm::length(m) < 1e-6f)
            {
                m = glm::cross(vec3_t(0.0f, 1.0f, 0.0f), n);
            }
            m = glm::normalize(m);

            vec3_t k = glm::cross(n, m);
            n = glm::normalize(n);

            mat3_t identity = glm::identity<mat3_t>();
            mat3_t nnT = glm::outerProduct(n, n);                               // n * n^T
            mat3_t mmT_kkT = glm::outerProduct(m, m) + glm::outerProduct(k, k); // m*m^T + k*k^T

            mat3_t Fa = identity +
                        (amp - 1.0f) * nnT +
                        (1.0f / sqrt(amp) - 1.0f) * mmT_kkT;

            int global_tet_id = m_tet_start_idx[mesh_id] + tet_id;
            m_tet_FaInv[global_tet_id] = glm::inverse(Fa);
        }

        void FemScene3D::updateLBSControl(cudaStream_t stream)
        {
            // Update rest configuration by the target position
            control::update_target_position(
                m_total_vertices,
                m_lbs_data.position_target.data(),
                m_total_tetrahedra,
                m_gpu_data.tetrahedra_indices.data(),
                m_gpu_data.tet_DmInv.data(),
                stream);

            // Reset lbs position
            m_lbs_data.position_lbs.copyFromAsync(m_lbs_data.position_rest, stream);
        }

        void FemScene3D::updateActiveStrain(cudaStream_t stream)
        {
            m_gpu_data.tet_FaInv.uploadAsync(m_tet_FaInv, stream);
        }

        // 2D FemScene

        void FemScene2D::addMesh(std::unique_ptr<Mesh2D> mesh)
        {
            if (mesh)
            {
                m_meshes.push_back(std::move(mesh));
            }
        }

        void FemScene2D::finalizeAndUpload()
        {
            ASSERT(!m_meshes.empty(), "Cannot finalize an empty FEM scene.");
            LOG_INFO("Finalizing 2D FEM scene and merging {} meshes...", m_meshes.size());

            // --- 1. total scene stats ---
            size_t total_volume_vertices = 0;
            size_t total_surface_vertices = 0;
            size_t total_edges = 0;
            size_t total_triangles = 0;
            size_t total_surface_edges = 0;

            std::vector<std::vector<int>> surface_to_volume_maps_local;
            std::vector<std::map<int, int>> volume_to_surface_maps_local;

            m_ctrl_verts_start_idx.resize(m_meshes.size() + 1, 0);

            for (int i = 0; i < m_meshes.size(); i++)
            {
                const auto &mesh = m_meshes[i];
                total_volume_vertices += mesh->getNumVertices();
                total_edges += mesh->getNumEdges();
                total_triangles += mesh->getNumTriangles();

                // --- extract surface vertices ---
                std::vector<bool> is_surface_vertex(mesh->getNumVertices(), false);
                for (int tri_v_idx : mesh->h_surface_edges_local_indices)
                {
                    is_surface_vertex[tri_v_idx] = true;
                }

                std::vector<int> s_to_v_map;
                std::map<int, int> v_to_s_map;
                int current_surface_v_idx = 0;
                for (int i = 0; i < mesh->getNumVertices(); ++i)
                {
                    if (is_surface_vertex[i])
                    {
                        s_to_v_map.push_back(i);                 // surf_idx -> vol_idx
                        v_to_s_map[i] = current_surface_v_idx++; // vol_idx -> surf_idx
                    }
                }

                surface_to_volume_maps_local.push_back(s_to_v_map);
                volume_to_surface_maps_local.push_back(v_to_s_map);

                total_surface_vertices += s_to_v_map.size();
                total_surface_edges += mesh->h_surface_edges_local_indices.size() / 2;

                m_ctrl_verts_start_idx[i + 1] = m_ctrl_verts_start_idx[i] + mesh->getNumControlPoint();
            }
            LOG_INFO("Total scene stats: {} volume vertices, {} triangles, {} edges, {} surface vertices, {} surface edges.",
                     total_volume_vertices, total_triangles, total_edges, total_surface_vertices, total_surface_edges);

            // --- 2. prepare merged data ---
            std::vector<vec2_t> merged_positions_flat(total_volume_vertices);
            std::vector<real> merged_vertex_masses(total_volume_vertices, 0.0);
            std::vector<real> merged_lbs_stiffness(total_volume_vertices, 0.0);

            std::vector<unsigned int> merged_tris(total_triangles * 3);
            std::vector<real> merged_tri_mu(total_triangles);
            std::vector<real> merged_tri_lambda(total_triangles);
            std::vector<real> merged_tri_kd(total_triangles);
            std::vector<real> merged_tri_area(total_triangles);
            std::vector<mat2_t> merged_tri_DmInv(total_triangles);
            m_tri_FaInv = std::vector<mat2_t>(total_triangles, glm::identity<mat2_t>());

            // FIXME: edge stiffness and edge kd from config
            std::vector<unsigned int> merged_edges(total_edges * 2);
            std::vector<real> merged_edge_stiff(total_edges);
            std::vector<real> merged_edge_kd(total_edges);
            std::vector<real> merged_edge_len(total_edges);

            std::vector<unsigned int> merged_surf_edges(total_surface_edges * 2);
            std::vector<vec2_t> merged_surf_pos_flat(total_surface_vertices);
            std::vector<unsigned int> merged_surf_to_vol_map(total_surface_vertices);
            std::vector<Matrix64> merged_tri_At(total_triangles);

            // --- 3. tranverse meshes and merge data ---
            size_t vol_vertex_offset = 0;
            size_t edge_offset = 0;
            size_t tri_offset = 0;

            size_t surf_vertex_offset = 0;
            size_t surf_edge_offset = 0;
            size_t mesh_idx = 0;

            m_verts_start_idx.resize(m_meshes.size());
            m_tri_start_idx.resize(m_meshes.size());

            for (const auto &mesh : m_meshes)
            {
                const size_t num_verts = mesh->getNumVertices();
                const size_t num_edges = mesh->getNumEdges();
                const size_t num_tris = mesh->getNumTriangles();

                for (size_t i = 0; i < num_verts; ++i)
                {
                    const size_t merged_idx = vol_vertex_offset + i;
                    merged_positions_flat[merged_idx] = mesh->h_initial_positions[i];
                    merged_lbs_stiffness[merged_idx] = mesh->m_lbs_stiffness;
                }

                for (size_t i = 0; i < num_edges; ++i)
                {
                    const size_t merged_idx = edge_offset + i;
                    for (int j = 0; j < 2; ++j)
                    {
                        merged_edges[merged_idx * 2 + j] = mesh->h_edges[i * 2 + j] + vol_vertex_offset;
                    }

                    real kd = mesh->m_kd;
                    merged_edge_kd[merged_idx] = kd;

                    auto v0 = mesh->h_initial_positions[mesh->h_edges[i * 2 + 0]];
                    auto v1 = mesh->h_initial_positions[mesh->h_edges[i * 2 + 1]];
                    merged_edge_len[merged_idx] = glm::length(v1 - v0);

                    // FIXME: seperate desity for edge and triangle (same issue for 3D)
                    real density = mesh->m_density;
                    real mass = density * merged_edge_len[merged_idx];
                    merged_vertex_masses[mesh->h_edges[i * 2 + 0] + vol_vertex_offset] += mass / 2.0;
                    merged_vertex_masses[mesh->h_edges[i * 2 + 1] + vol_vertex_offset] += mass / 2.0;
                }

                for (size_t i = 0; i < num_tris; ++i)
                {
                    const size_t merged_idx = tri_offset + i;
                    for (int j = 0; j < 3; ++j)
                    {
                        merged_tris[merged_idx * 3 + j] = mesh->h_triangles[i * 3 + j] + vol_vertex_offset;
                    }
                    auto v2 = mesh->h_initial_positions[mesh->h_triangles[i * 3 + 2]];
                    auto e1 = mesh->h_initial_positions[mesh->h_triangles[i * 3 + 0]] - v2;
                    auto e2 = mesh->h_initial_positions[mesh->h_triangles[i * 3 + 1]] - v2;
                    auto det = e1.x * e2.y - e1.y * e2.x;
                    real area = 0.5 * abs(det);
                    if (NumericUtils::isApproxEqual(area, 0.0))
                    {
                        LOG_CRITICAL("Triangle {} has zero area.", merged_idx);
                    }
                    merged_tri_area[merged_idx] = area;

                    real density = mesh->m_density;
                    real mass = density * area;
                    merged_vertex_masses[mesh->h_triangles[i * 3 + 0] + vol_vertex_offset] += mass / 3.0;
                    merged_vertex_masses[mesh->h_triangles[i * 3 + 1] + vol_vertex_offset] += mass / 3.0;
                    merged_vertex_masses[mesh->h_triangles[i * 3 + 2] + vol_vertex_offset] += mass / 3.0;

                    real E = mesh->m_youngs_modulus;
                    real nu = mesh->m_poisson_ratio;
                    merged_tri_mu[merged_idx] = E / (2.0 * (1.0 + nu));
                    merged_tri_lambda[merged_idx] = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
                    merged_tri_kd[merged_idx] = mesh->m_kd;

                    mat2_t Dm(e1[0], e2[0],
                              e1[1], e2[1]);
                    mat2_t Dm_inv = glm::inverse(Dm);
                    merged_tri_DmInv[merged_idx] = Dm_inv;

                    Matrix32 grad_undeformed_sample_weight{
                        {Dm_inv[0][0], Dm_inv[0][1]},
                        {Dm_inv[1][0], Dm_inv[1][1]},
                        {-Dm_inv[0][0] - Dm_inv[1][0], -Dm_inv[0][1] - Dm_inv[1][1]}};
                    Matrix64 Ate = Matrix64::Zero();
                    int v_dim = 2;
                    int e_dim = 3;
                    for (int i = 0; i < v_dim; i++)
                    {
                        for (int j = 0; j < v_dim; j++)
                        {
                            for (int k = 0; k < e_dim; k++)
                            {
                                int row = i + j * v_dim;
                                int col = i + k * v_dim;
                                real val = grad_undeformed_sample_weight(k, j);
                                Ate(col, row) = val;
                            }
                        }
                    }
                    merged_tri_At[merged_idx] = Ate;
                }

                const auto &local_s_to_v_map = surface_to_volume_maps_local[mesh_idx];
                const auto &local_v_to_s_map = volume_to_surface_maps_local[mesh_idx];

                for (size_t i = 0; i < local_s_to_v_map.size(); ++i)
                {
                    int vol_idx = local_s_to_v_map[i];
                    const auto &pos = mesh->h_initial_positions[vol_idx];
                    size_t merged_surf_idx = surf_vertex_offset + i;
                    merged_surf_pos_flat[merged_surf_idx] = pos;
                }

                for (size_t i = 0; i < mesh->h_surface_edges_local_indices.size() / 2; ++i)
                {
                    size_t merged_edge_idx = surf_edge_offset + i;
                    for (int j = 0; j < 2; ++j)
                    {
                        int local_vol_idx = mesh->h_surface_edges_local_indices[i * 2 + j];
                        int local_surf_idx = local_v_to_s_map.at(local_vol_idx);
                        merged_surf_edges[merged_edge_idx * 2 + j] = local_surf_idx + surf_vertex_offset;
                    }
                }

                for (size_t i = 0; i < local_s_to_v_map.size(); ++i)
                {
                    int local_vol_idx = local_s_to_v_map[i];
                    size_t merged_surf_idx = surf_vertex_offset + i;
                    merged_surf_to_vol_map[merged_surf_idx] = local_vol_idx + vol_vertex_offset;
                }

                m_verts_start_idx[mesh_idx] = vol_vertex_offset;
                m_tri_start_idx[mesh_idx] = tri_offset;

                auto cp = mesh->getControlPoint();
                for (size_t i = 0; i < cp.size(); i++)
                {
                    m_ctrl_verts_idx.push_back(cp[i] + vol_vertex_offset);
                }

                // update offsets
                vol_vertex_offset += mesh->getNumVertices();
                surf_vertex_offset += local_s_to_v_map.size();
                edge_offset += mesh->getNumEdges();
                tri_offset += mesh->getNumTriangles();
                surf_edge_offset += mesh->h_surface_edges_local_indices.size() / 2;
                mesh_idx++;
            }

            // --- 4. compute merged tri neighbors ---
            std::vector<unsigned int> tri_neighbors_num(total_volume_vertices);
            std::vector<unsigned int> tri_neighbors_start_indices(total_volume_vertices + 1);
            std::vector<std::vector<unsigned int>> tmp_neitriIdx(total_volume_vertices);
            std::vector<std::vector<unsigned int>> tmp_Idxinneitri(total_volume_vertices);
            for (int i = 0; i < total_triangles; i++)
            {
                tmp_neitriIdx[merged_tris[3 * i]].push_back(i);
                tmp_neitriIdx[merged_tris[3 * i + 1]].push_back(i);
                tmp_neitriIdx[merged_tris[3 * i + 2]].push_back(i);
                tmp_Idxinneitri[merged_tris[3 * i]].push_back(0);
                tmp_Idxinneitri[merged_tris[3 * i + 1]].push_back(1);
                tmp_Idxinneitri[merged_tris[3 * i + 2]].push_back(2);
            }

            tri_neighbors_start_indices[0] = 0;
            for (int i = 0; i < total_volume_vertices; i++)
            {
                tri_neighbors_num[i] = tmp_neitriIdx[i].size();
                tri_neighbors_start_indices[i + 1] = tri_neighbors_start_indices[i] + tri_neighbors_num[i];
            }

            int tol = tri_neighbors_start_indices[total_volume_vertices];
            std::vector<unsigned int> neitriIdx_host(tol);
            std::vector<unsigned int> Idxinneitri_host(tol);
            for (int i = 0; i < total_volume_vertices; i++)
            {
                for (int j = 0; j < tmp_neitriIdx[i].size(); j++)
                {
                    neitriIdx_host[tri_neighbors_start_indices[i] + j] = tmp_neitriIdx[i][j];
                    Idxinneitri_host[tri_neighbors_start_indices[i] + j] = tmp_Idxinneitri[i][j];
                }
            }

            ////////////////////////////////////////////
            // Gradient related arrays.
            ////////////////////////////////////////////

            std::vector<std::vector<integer>> elastic_gradient_map_;
            // Assemble gradient map.
            {
                elastic_gradient_map_.clear();
                elastic_gradient_map_.resize(total_volume_vertices);
                for (integer e = 0; e < total_triangles; ++e)
                {
                    for (integer i = 0; i < 3; ++i)
                    {
                        const integer dof_map_i = merged_tris[3 * e + i];
                        elastic_gradient_map_[dof_map_i].push_back(3 * e + i);
                    }
                }
            }

            // Assemble the nonzero structures in elastic energy Hessian and its projection.
            std::vector<std::vector<integer>> elastic_hessian_nonzero_map_;
            {
                std::vector<Eigen::Triplet<real>> elastic_hess_nonzeros;
                for (integer e = 0; e < total_triangles; ++e)
                {
                    for (integer i = 0; i < 3; ++i)
                        for (integer j = 0; j < 3; ++j)
                            for (integer di = 0; di < 2; ++di)
                                for (integer dj = 0; dj < 2; ++dj)
                                {
                                    const integer row_idx = merged_tris[3 * e + i] * 2 + di;
                                    const integer col_idx = merged_tris[3 * e + j] * 2 + dj;
                                    elastic_hess_nonzeros.emplace_back(row_idx, col_idx, 1.0);
                                }
                }
                m_lbs_data.elastic_hessian_ = FromTriplet(2 * total_volume_vertices, 2 * total_volume_vertices, elastic_hess_nonzeros);
                // Rest assured that this is deep copy.
                m_lbs_data.elastic_hessian_projection_ = m_lbs_data.elastic_hessian_;

                m_lbs_data.elastic_hessian_nonzero_num_ = static_cast<integer>(m_lbs_data.elastic_hessian_.nonZeros());
                elastic_hessian_nonzero_map_.resize(m_lbs_data.elastic_hessian_nonzero_num_);
                for (integer e = 0; e < total_triangles; ++e)
                {
                    for (integer i = 0; i < 3; ++i)
                        for (integer j = 0; j < 3; ++j)
                            for (integer di = 0; di < 2; ++di)
                                for (integer dj = 0; dj < 2; ++dj)
                                {
                                    const integer row_idx = merged_tris[3 * e + i] * 2 + di;
                                    const integer col_idx = merged_tris[3 * e + j] * 2 + dj;
                                    const integer k = &m_lbs_data.elastic_hessian_.coeffRef(row_idx, col_idx) - m_lbs_data.elastic_hessian_.valuePtr();
                                    elastic_hessian_nonzero_map_[k].push_back({36 * e + 6 * (i * 2 + di) + j * 2 + dj});
                                }
                }
            }

            std::vector<integer> elastic_gradient_map_begin_index(total_volume_vertices + 1, 0);
            std::vector<integer> elastic_gradient_map;
            for (integer k = 0; k < total_volume_vertices; ++k)
            {
                const auto &mapping = elastic_gradient_map_[k];
                const integer mapping_size = static_cast<integer>(mapping.size());
                elastic_gradient_map_begin_index[k + 1] = elastic_gradient_map_begin_index[k] + mapping_size;

                for (integer m = 0; m < mapping_size; ++m)
                {
                    elastic_gradient_map.push_back(mapping[m]);
                }
            }

            ////////////////////////////////////////////
            // Hessian related arrays.
            ////////////////////////////////////////////
            std::vector<integer> elastic_hessian_nonzero_map_begin_index(m_lbs_data.elastic_hessian_nonzero_num_ + 1, 0);
            std::vector<integer> elastic_hessian_nonzero_map;
            // std::vector<integer> is_diagonal(m_lbs_data.elastic_hessian_nonzero_num_, 0);

            for (integer k = 0; k < m_lbs_data.elastic_hessian_nonzero_num_; ++k)
            {
                const auto &mapping = elastic_hessian_nonzero_map_[k];
                const integer mapping_size = static_cast<integer>(mapping.size());
                elastic_hessian_nonzero_map_begin_index[k + 1] = elastic_hessian_nonzero_map_begin_index[k] + mapping_size;

                for (integer m = 0; m < mapping_size; ++m)
                {
                    elastic_hessian_nonzero_map.push_back(mapping[m]);
                }
            }

            VectorXr stiffness(2 * total_volume_vertices);
            for (integer v = 0; v < total_volume_vertices; ++v)
            {
                stiffness(2 * v + 0) = 2 * merged_lbs_stiffness[v];
                stiffness(2 * v + 1) = 2 * merged_lbs_stiffness[v];
            }
            m_lbs_data.hessian_lbs_ = FromDiagonal(stiffness);

            // --- 5. coloring ---
            GAIA::GraphColoring::TetMeshVertexGraph tetMeshVertexGraph;
            tetMeshVertexGraph.fromCoMesh2d(merged_tris.data(), merged_edges.data(), total_volume_vertices, total_triangles, total_edges);
            GAIA::GraphColoring::Mcs mcs(tetMeshVertexGraph);
            mcs.color();
            mcs.convertToColoredCategories();
            auto colors = mcs.get_categories();
            m_gpu_data.color_num = colors.size();
            m_gpu_data.color_vertex_nums.resize(colors.size());
            m_gpu_data.color_vertex_indices.resize(colors.size());

            LOG_INFO("Successfully set colors for {} meshes. Total colors: {}", m_meshes.size(), colors.size());
            LOG_INFO("Vertices in each color: ");

            for (size_t i = 0; i < colors.size(); i++)
            {
                m_gpu_data.color_vertex_nums[i] = colors[i].size();
                m_gpu_data.color_vertex_indices[i].resize(colors[i].size());
                m_gpu_data.color_vertex_indices[i].upload(colors[i]);
                LOG_INFO("Color {}: {} vertices", i, colors[i].size());
            }

            // --- 6. set scene data ---
            m_total_vertices = total_volume_vertices;
            m_total_edges = total_edges;
            m_total_triangles = total_triangles;
            m_total_surface_vertices = total_surface_vertices;
            m_total_surface_edges = total_surface_edges;

            // --- 7. allocate memory for merged scene data ---
            LOG_INFO("Uploading merged scene data to GPU...");
            m_gpu_data.positions.resize(m_total_vertices);
            m_gpu_data.positions.upload(merged_positions_flat);
            // LOG_INFO("merged_positions_flat: {}", utils::join(merged_positions_flat, ", "));
            m_gpu_data.pre_positions.resize(m_total_vertices);
            m_gpu_data.pre_positions.setZero();
            m_gpu_data.itr_pre_positions.resize(m_total_vertices);
            m_gpu_data.itr_pre_positions.setZero();
            m_gpu_data.itr_pre_pre_positions.resize(m_total_vertices);
            m_gpu_data.itr_pre_pre_positions.setZero();

            m_gpu_data.triangles_indices.resize(m_total_triangles * 3);
            m_gpu_data.triangles_indices.upload(merged_tris);
            m_gpu_data.neibour_triangles_nums.resize(m_total_vertices);
            m_gpu_data.neibour_triangles_nums.upload(tri_neighbors_num);
            m_gpu_data.neibour_triangles_start_indices.resize(m_total_vertices + 1);
            m_gpu_data.neibour_triangles_start_indices.upload(tri_neighbors_start_indices);
            m_gpu_data.neibour_triangles_indices.resize(tol);
            m_gpu_data.neibour_triangles_indices.upload(neitriIdx_host);
            m_gpu_data.vertex_indices_in_neibour_triangles.resize(tol);
            m_gpu_data.vertex_indices_in_neibour_triangles.upload(Idxinneitri_host);

            m_gpu_data.vertex_masses.resize(m_total_vertices);
            m_gpu_data.vertex_masses.upload(merged_vertex_masses);

            m_gpu_data.tri_mu.resize(m_total_triangles);
            m_gpu_data.tri_mu.upload(merged_tri_mu);
            m_gpu_data.tri_lambda.resize(m_total_triangles);
            m_gpu_data.tri_lambda.upload(merged_tri_lambda);
            m_gpu_data.tri_kd.resize(m_total_triangles);
            m_gpu_data.tri_kd.upload(merged_tri_kd);
            m_gpu_data.tri_areas.resize(m_total_triangles);
            m_gpu_data.tri_areas.upload(merged_tri_area);
            m_gpu_data.tri_DmInv.resize(m_total_triangles);
            m_gpu_data.tri_DmInv.upload(merged_tri_DmInv);
            m_gpu_data.tri_FaInv.resize(m_total_triangles);
            m_gpu_data.tri_FaInv.upload(m_tri_FaInv);

            m_gpu_data.tri_At.resize(m_total_triangles * 24);
            std::vector<real> tmp_tri_At(m_total_triangles * 24);
            for (int i = 0; i < m_total_triangles; i++)
            {
                memcpy(&tmp_tri_At[i * 24], merged_tri_At[i].data(), sizeof(real) * 24);
            }
            m_gpu_data.tri_At.upload(tmp_tri_At);

            m_gpu_data.edges_indices.resize(m_total_edges * 2);
            m_gpu_data.edges_indices.upload(merged_edges);
            m_gpu_data.edge_kd.resize(m_total_edges);
            m_gpu_data.edge_kd.upload(merged_edge_kd);
            m_gpu_data.edge_rest_lengths.resize(m_total_edges);
            m_gpu_data.edge_rest_lengths.upload(merged_edge_len);
            m_gpu_data.edge_stiffness.resize(m_total_edges);
            m_gpu_data.edge_stiffness.upload(merged_edge_stiff);

            m_coupling_data.surface_positions.resize(m_total_surface_vertices * 2);
            m_coupling_data.surface_positions.setZero();
            m_coupling_data.surface_velocities.resize(m_total_surface_vertices * 2);
            m_coupling_data.surface_velocities.setZero();
            m_coupling_data.surface_forces.resize(m_total_surface_vertices * 2);
            m_coupling_data.surface_forces.setZero();
            m_coupling_data.surface_elements_indices.resize(m_total_surface_edges * 2);
            m_coupling_data.surface_elements_indices.upload(merged_surf_edges);
            m_coupling_data.d_surface_to_volume_map.resize(m_total_surface_vertices);
            m_coupling_data.d_surface_to_volume_map.upload(merged_surf_to_vol_map);

            m_gpu_data.velocities.resize(m_total_vertices);
            m_gpu_data.velocities.setZero();
            m_gpu_data.pre_velocities.resize(m_total_vertices);
            m_gpu_data.pre_velocities.setZero();
            m_gpu_data.inertia.resize(m_total_vertices);
            m_gpu_data.inertia.setZero();
            m_gpu_data.forces.resize(m_total_vertices);
            m_gpu_data.forces.setZero();

            m_gpu_data.vertex_num = m_total_vertices;
            m_gpu_data.edge_num = m_total_edges;
            m_gpu_data.triangle_num = m_total_triangles;

            m_lbs_data.position_rest.resize(m_total_vertices);
            m_lbs_data.position_rest.upload(merged_positions_flat);
            m_lbs_data.tri_DmInv.resize(m_total_triangles);
            m_lbs_data.tri_DmInv.upload(merged_tri_DmInv);
            m_lbs_data.position_lbs.resize(m_total_vertices);
            m_lbs_data.position_lbs.upload(merged_positions_flat);
            m_lbs_data.position_target.resize(m_total_vertices);
            m_lbs_data.position_target.setZero();
            m_lbs_data.stiffness.resize(m_total_vertices);
            m_lbs_data.stiffness.upload(merged_lbs_stiffness);

            m_lbs_data.elastic_energy_integral.resize(m_total_triangles);
            m_lbs_data.lbs_energy_integral.resize(m_total_vertices);
            m_lbs_data.elastic_gradient_integral.resize(m_total_triangles * 6);
            m_lbs_data.elastic_gradient_map_begin_index.resize(m_total_vertices + 1);
            m_lbs_data.elastic_gradient_map.resize(elastic_gradient_map.size());
            m_lbs_data.elastic_gradient_map_begin_index.upload(elastic_gradient_map_begin_index);
            m_lbs_data.elastic_gradient_map.upload(elastic_gradient_map);
            m_lbs_data.elastic_gradient_value_ptr.resize(m_total_vertices * 2);
            m_lbs_data.elastic_hessian_integral.resize(m_total_triangles * 36);
            m_lbs_data.elastic_hessian_projection_integral.resize(m_total_triangles * 36);
            m_lbs_data.elastic_hessian_nonzero_map_begin_index.resize(m_lbs_data.elastic_hessian_nonzero_num_ + 1);
            m_lbs_data.elastic_hessian_nonzero_map.resize(elastic_hessian_nonzero_map.size());
            m_lbs_data.elastic_hessian_nonzero_map_begin_index.upload(elastic_hessian_nonzero_map_begin_index);
            m_lbs_data.elastic_hessian_nonzero_map.upload(elastic_hessian_nonzero_map);
            m_lbs_data.elastic_hessian_value_ptr.resize(m_lbs_data.elastic_hessian_nonzero_num_);
            // m_lbs_data.hessian_is_diagnal.resize(m_lbs_data.elastic_hessian_nonzero_num_);
            // m_lbs_data.hessian_is_diagnal.upload(is_diagonal);

            // copyGlmToEigen(merged_positions_flat, m_cpu_data.position);
            // copyVecToEigen2X(merged_edges, m_cpu_data.edges_indices);
            // copyVecToEigen3X(merged_tris, m_cpu_data.triangles_indices);
            // m_cpu_data.next_position.resize(2, m_total_vertices);
            // m_cpu_data.next_position.setZero();
            // m_cpu_data.external_acceleration.resize(2, m_total_vertices);
            // m_cpu_data.external_acceleration.setZero();
            // m_cpu_data.velocity.resize(2, m_total_vertices);
            // m_cpu_data.velocity.setZero();
            // m_cpu_data.next_velocity.resize(2, m_total_vertices);
            // m_cpu_data.next_velocity.setZero();

            // m_cpu_data.tetMeshMu = merged_tet_mu;
            // m_cpu_data.tetMeshLambda = merged_tet_lambda;
            // m_cpu_data.tetMeshKd = merged_tet_kd;
            // m_cpu_data.tetVolume = merged_tet_volume;
            // m_cpu_data.tetDmInv.resize(m_total_tetrahedra);
            // for (int i = 0; i < m_total_tetrahedra; i++) {
            //     m_cpu_data.tetDmInv[i] = convertGlmToEigen(merged_tet_DmInv[i]);
            // }
            // m_cpu_data.tetAt.resize(m_total_tetrahedra);
            // m_cpu_data.tetrahedron_num = m_total_tetrahedra;
            // m_cpu_data.triangle_num = m_total_triangles;
            // m_cpu_data.vertex_num = m_total_vertices;
            m_cpu_data.mass_matrix = FromDiagonal(merged_vertex_masses.data(), m_total_vertices);

            LOG_INFO("FEM scene finalized and ready for simulation.");
        }

        void FemScene2D::reset()
        {
            m_gpu_data.velocities.setZero();
            m_gpu_data.pre_velocities.setZero();
            m_gpu_data.forces.setZero();
            m_gpu_data.positions.copyFrom(m_lbs_data.position_rest);
            m_gpu_data.pre_positions.copyFrom(m_lbs_data.position_rest);
            m_gpu_data.tri_DmInv.copyFrom(m_lbs_data.tri_DmInv);
            m_tri_FaInv.assign(m_total_triangles, glm::identity<mat2_t>());
            m_gpu_data.tri_FaInv.upload(m_tri_FaInv);
        }

        void FemScene2D::lazyApplyLBSControl(int mesh_id, const std::vector<vec2_t> &lbs_shift, const std::vector<real> &lbs_rotation, cudaStream_t stream)
        {
            m_meshes[mesh_id]->applyLBSControl(lbs_shift, lbs_rotation, m_lbs_data.position_lbs, m_verts_start_idx[mesh_id], stream);
        }

        void FemScene2D::lazyApplyActiveStrain(int mesh_id, int tri_id, vec2_t dir, real magnitude, cudaStream_t stream)
        {
            mat2_t fainv;
            real tmp_co = 1.0 / magnitude - magnitude;
            fainv[0][0] = tmp_co * dir[0] * dir[0] + magnitude;
            fainv[0][1] = tmp_co * dir[0] * dir[1];
            fainv[1][0] = fainv[0][1];
            fainv[1][1] = tmp_co * dir[1] * dir[1] + magnitude;

            int idx = m_tri_start_idx[mesh_id] + tri_id;

            m_tri_FaInv[idx] = fainv;
        }

        void FemScene2D::updateLBSControl(cudaStream_t stream)
        {
            // Update rest configuration by the target position
            control::update_target_position(
                m_total_vertices,
                m_lbs_data.position_target.data(),
                m_total_triangles,
                m_gpu_data.triangles_indices.data(),
                m_gpu_data.tri_DmInv.data(),
                stream);

            // Reset lbs position
            m_lbs_data.position_lbs.copyFromAsync(m_lbs_data.position_rest, stream);
        }

        void FemScene2D::updateActiveStrain(cudaStream_t stream)
        {
            // Upload updated FaInv to GPU
            m_gpu_data.tri_FaInv.uploadAsync(m_tri_FaInv, stream);
        }

        SolidGeometryProxy_Device<2> FemScene2D::getSurfaceProxyForFSI()
        {
            coupling::updateSurfaceStatesFromVolume(
                m_gpu_data,
                m_coupling_data,
                m_total_surface_vertices,
                0);
            return {
                m_coupling_data.surface_positions.data(),
                m_coupling_data.surface_velocities.data(),
                m_coupling_data.surface_elements_indices.data(),
                m_coupling_data.surface_forces.data(),
                static_cast<int>(m_total_surface_vertices),
                static_cast<int>(m_total_surface_edges)};
        }

    } // namespace fem
} // namespace fsi