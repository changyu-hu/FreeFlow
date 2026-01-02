// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once
#include <vector>
#include <string>
#include <memory>
#include <glm/glm.hpp>

#include "FSI_Simulator/utils/Config.hpp"
#include "FSI_Simulator/core/SolidGeometryProxy_Device.cuh"
#include "FSI_Simulator/fem/FemScene.hpp"
#include "FSI_Simulator/io/VtkWriter.hpp"

namespace fsi
{

    namespace fem
    {

        class ISolidSolver2D
        {
        public:
            virtual ~ISolidSolver2D() = default;

            virtual void initialize(const SimulationParameters2D &params)
            {
                m_scene = std::make_unique<FemScene2D>();
                for (const auto &solid_config : params.solids)
                {
                    m_scene->addMesh(std::make_unique<Mesh2D>(solid_config));
                }

                m_scene->finalizeAndUpload();
            }

            virtual void reset()
            {
                m_scene->reset();
            }

            // excute one time step of the solver
            virtual void advanceTimeStep() = 0;

            // solid properties getters
            virtual std::vector<vec2_t> getPositions() const
            {
                ASSERT(m_scene, "Solver not initialized.");

                auto positions_2d = m_scene->getGpuData().positions.download();
                return positions_2d;
            }
            virtual std::vector<vec2_t> getVelocities() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                auto velocity_2d = m_scene->getGpuData().velocities.download();
                return velocity_2d;
            }
            virtual std::vector<vec2_t> getForce() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                return m_scene->getGpuData().forces.download();
            }
            virtual std::vector<unsigned int> getTriangleIdx() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                return m_scene->getGpuData().triangles_indices.download();
            }
            virtual std::vector<vec2_t> getBoundaryVertices() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                auto proxy = m_scene->getSurfaceProxyForFSI();
                auto surf_verts = m_scene->getCouplingData().surface_positions.download();
                std::vector<vec2_t> vec_data(surf_verts.size() / 2);
                for (size_t i = 0; i < vec_data.size(); ++i)
                {
                    vec_data[i] = vec2_t(
                        surf_verts[i * 2 + 0],
                        surf_verts[i * 2 + 1]);
                }
                return vec_data;
            }
            virtual std::vector<unsigned int> getBoundaryElements() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                return m_scene->getCouplingData().surface_elements_indices.download();
            }
            virtual std::vector<unsigned int> getBoundaryPointIdx() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                return m_scene->getCouplingData().d_surface_to_volume_map.download();
            }
            virtual std::vector<vec2_t> getLBSVertices() const
            {
                ASSERT(false, "LBS control not supported for current solid solver.");
                return std::vector<vec2_t>();
            }
            virtual std::vector<int> getControlPointIdx() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                return m_scene->getControlPointIdx();
            }

            // apply fluid forces to solid
            virtual void applyForces() = 0;

            // apply control
            virtual void applyLBSControl(int mesh_id, const std::vector<vec2_t> &lbs_shift, const std::vector<real> &lbs_rotation)
            {
                LOG_ERROR("LBS control not supported for current solid solver.");
            }

            virtual void applyActiveStrain(int mesh_id, int tri_id, vec2_t dir, real magnitude)
            {
                LOG_ERROR("Active strain control not supported for current solid solver.");
            }

            virtual void applyKinematicControl(int mesh_id, const std::vector<vec2_t> &target_pos, real vel)
            {
                LOG_ERROR("Kinematic control not supported for current solid solver.");
            }

            SolidGeometryProxy_Device<2> getSolidGeometryProxy()
            {
                ASSERT(m_scene, "Solver not initialized!");
                return m_scene->getSurfaceProxyForFSI();
            }

            virtual void saveFrameData(std::string filepath) const
            {
            }

        protected:
            std::unique_ptr<FemScene2D> m_scene;
        };

        class ISolidSolver3D
        {
        public:
            virtual ~ISolidSolver3D() = default;

            virtual void initialize(const SimulationParameters3D &params)
            {
                m_scene = std::make_unique<FemScene3D>();
                for (const auto &solid_config : params.solids)
                {
                    m_scene->addMesh(std::make_unique<Mesh3D>(solid_config));
                }

                m_scene->finalizeAndUpload();
            }

            virtual void reset()
            {
                m_scene->reset();
            }

            virtual void advanceTimeStep() = 0;

            virtual std::vector<vec3_t> getPositions() const
            {
                ASSERT(m_scene, "Solver not initialized.");

                auto positions_3d = m_scene->getGpuData().positions.download();
                return positions_3d;
            }
            virtual std::vector<vec3_t> getVelocities() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                auto velocity_3d = m_scene->getGpuData().velocities.download();
                return velocity_3d;
            }
            virtual std::vector<vec3_t> getForce() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                return m_scene->getGpuData().forces.download();
            }
            virtual std::vector<unsigned int> getTetrahedronIdx() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                return m_scene->getGpuData().tetrahedra_indices.download();
            }
            virtual std::vector<vec3_t> getBoundaryVertices() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                auto proxy = m_scene->getSurfaceProxyForFSI();
                auto surf_verts = m_scene->getCouplingData().surface_positions.download();
                std::vector<vec3_t> vec_data(surf_verts.size() / 3);
                for (size_t i = 0; i < vec_data.size(); ++i)
                {
                    vec_data[i] = vec3_t(
                        surf_verts[i * 3 + 0],
                        surf_verts[i * 3 + 1],
                        surf_verts[i * 3 + 2]);
                }
                return vec_data;
            }
            virtual std::vector<unsigned int> getBoundaryElements() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                return m_scene->getCouplingData().surface_elements_indices.download();
            }
            virtual std::vector<vec3_t> getLBSVertices() const
            {
                ASSERT(false, "LBS control not supported for current solid solver.");
                return std::vector<vec3_t>();
            }
            virtual std::vector<unsigned int> getBoundaryPointIdx() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                return m_scene->getCouplingData().d_surface_to_volume_map.download();
            }
            virtual std::vector<int> getControlPointIdx() const
            {
                ASSERT(m_scene, "Solver not initialized.");
                return m_scene->getControlPointIdx();
            }

            virtual void applyForces() = 0;

            virtual void applyLBSControl(int mesh_id, const std::vector<vec3_t> &lbs_shift, const std::vector<mat3_t> &lbs_rotation)
            {
                LOG_ERROR("LBS control not supported for current solid solver.");
            }

            virtual void applyActiveStrain(int mesh_id, int tet_id, vec3_t dir, real magnitude)
            {
                LOG_ERROR("Active strain control not supported for current solid solver.");
            }

            SolidGeometryProxy_Device<3> getSolidGeometryProxy()
            {
                ASSERT(m_scene, "Solver not initialized!");
                return m_scene->getSurfaceProxyForFSI();
            }

            virtual void applyKinematicControl(int mesh_id, const std::vector<vec3_t> &target_pos, real vel)
            {
                LOG_ERROR("Kinematic control not supported for current solid solver.");
            }

            virtual void saveFrameData(std::string filepath) const
            {
                auto vertices = m_scene->getGpuData().positions.download();

                std::vector<std::pair<VtkCellType, std::vector<uint32_t>>> element_groups;

                if (m_scene->getTotalTetrahedra() > 0)
                {
                    std::vector<uint32_t> tets_int = m_scene->getGpuData().tetrahedra_indices.download();
                    std::vector<uint32_t> tets_uint(tets_int.begin(), tets_int.end());
                    element_groups.push_back({VtkCellType::Tetra, tets_uint});
                }

                if (m_scene->getTotalTriangles() > 0)
                {
                    std::vector<uint32_t> tris_int = m_scene->getGpuData().triangles_indices.download();
                    std::vector<uint32_t> tris_uint(tris_int.begin(), tris_int.end());
                    element_groups.push_back({VtkCellType::Triangle, tris_uint});
                }

                io::VtkWriter::writeUnstructuredGrid(filepath, vertices, element_groups);
            }

        protected:
            std::unique_ptr<FemScene3D> m_scene;
        };

    } // namespace fem

} // namespace fsi