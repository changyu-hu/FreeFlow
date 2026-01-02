// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/fem/DeformableMesh.hpp"
#include "FSI_Simulator/fem/VBDSceneDataGpu.cuh"
#include "FSI_Simulator/control/LBSDataGpu.cuh"
#include "FSI_Simulator/core/SolidGeometryProxy_Device.cuh"
#include "FSI_Simulator/fem/NewtonSceneDataCpu.hpp"
#include "FSI_Simulator/fsi/FsiCouplingDataGpu.cuh"
#include <vector>
#include <memory>

namespace fsi
{
    namespace fem
    {

        class FemScene3D
        {
        public:
            FemScene3D() = default;

            FemScene3D(const FemScene3D &) = delete;
            FemScene3D &operator=(const FemScene3D &) = delete;
            FemScene3D(FemScene3D &&) = default;
            FemScene3D &operator=(FemScene3D &&) = default;

            void addMesh(std::unique_ptr<Mesh3D> mesh);

            void finalizeAndUpload();

            void reset();

            const VbdSceneDataGpu3D &getGpuData() const { return m_gpu_data; }
            VbdSceneDataGpu3D &getGpuData() { return m_gpu_data; }

            const coupling::FsiCouplingDataGpu &getCouplingData() const { return m_coupling_data; }
            coupling::FsiCouplingDataGpu &getCouplingData() { return m_coupling_data; }

            const NewtonSceneDataCpu &getCpuData() const { return m_cpu_data; }
            NewtonSceneDataCpu &getCpuData() { return m_cpu_data; }

            const control::LBSDataGpu3D &getLBSData() const { return m_lbs_data; }
            control::LBSDataGpu3D &getLBSData() { return m_lbs_data; }

            size_t getNumMeshes() const { return m_meshes.size(); }

            size_t getTotalVertices() const { return m_total_vertices; }

            size_t getTotalSurfaceVertices() const { return m_total_surface_vertices; }

            size_t getTotalTetrahedra() const { return m_total_tetrahedra; }

            size_t getTotalTriangles() const { return m_total_triangles; }

            std::vector<int> getControlPointIdx() const { return m_ctrl_verts_idx; }

            SolidGeometryProxy_Device<3> getSurfaceProxyForFSI();

            // LBS control
            void lazyApplyLBSControl(int mesh_id, const std::vector<vec3_t> &lbs_shift, const std::vector<mat3_t> &lbs_rotation, cudaStream_t stream);

            void updateLBSControl(cudaStream_t stream);

            void lazyApplyActiveStrain(int mesh_id, int tet_id, vec3_t dir, real magnitude, cudaStream_t stream);

            void updateActiveStrain(cudaStream_t stream);

        private:
            std::vector<std::unique_ptr<Mesh3D>> m_meshes;

            VbdSceneDataGpu3D m_gpu_data;
            coupling::FsiCouplingDataGpu m_coupling_data;

            NewtonSceneDataCpu m_cpu_data;

            control::LBSDataGpu3D m_lbs_data;

            std::vector<int> m_verts_start_idx;
            std::vector<int> m_tet_start_idx;
            std::vector<int> m_ctrl_verts_start_idx;
            std::vector<int> m_ctrl_verts_idx;

            std::vector<mat3_t> m_tet_FaInv;

            size_t m_total_vertices = 0;
            size_t m_total_tetrahedra = 0;
            size_t m_total_triangles = 0;
            size_t m_total_surface_vertices = 0;
            size_t m_total_surface_tris = 0;
        };

        class FemScene2D
        {
        public:
            FemScene2D() = default;

            FemScene2D(const FemScene2D &) = delete;
            FemScene2D &operator=(const FemScene2D &) = delete;
            FemScene2D(FemScene2D &&) = default;
            FemScene2D &operator=(FemScene2D &&) = default;

            void addMesh(std::unique_ptr<Mesh2D> mesh);
            void finalizeAndUpload();

            void reset();

            void lazyApplyLBSControl(int mesh_id, const std::vector<vec2_t> &lbs_shift, const std::vector<real> &lbs_rotation, cudaStream_t stream);

            void lazyApplyActiveStrain(int mesh_id, int tri_id, vec2_t dir, real magnitude, cudaStream_t stream);

            void updateLBSControl(cudaStream_t stream);

            void updateActiveStrain(cudaStream_t stream);

            const VbdSceneDataGpu2D &getGpuData() const { return m_gpu_data; }
            VbdSceneDataGpu2D &getGpuData() { return m_gpu_data; }

            SolidGeometryProxy_Device<2> getSurfaceProxyForFSI();

            const coupling::FsiCouplingDataGpu &getCouplingData() const { return m_coupling_data; }
            coupling::FsiCouplingDataGpu &getCouplingData() { return m_coupling_data; }

            const control::LBSDataGpu2D &getLBSData() const { return m_lbs_data; }
            control::LBSDataGpu2D &getLBSData() { return m_lbs_data; }

            const NewtonSceneDataCpu &getCpuData() const { return m_cpu_data; }
            NewtonSceneDataCpu &getCpuData() { return m_cpu_data; }
            size_t getTotalSurfaceVertices() const { return m_total_surface_vertices; }
            size_t getNumMeshes() const { return m_meshes.size(); }
            size_t getTotalVertices() const { return m_total_vertices; }
            size_t getTotalTriangles() const { return m_total_triangles; }
            std::vector<int> getControlPointIdx() const { return m_ctrl_verts_idx; }

        private:
            std::vector<std::unique_ptr<Mesh2D>> m_meshes;

            VbdSceneDataGpu2D m_gpu_data;
            coupling::FsiCouplingDataGpu m_coupling_data;
            NewtonSceneDataCpu m_cpu_data;
            control::LBSDataGpu2D m_lbs_data;
            std::vector<mat2_t> m_tri_FaInv;

            std::vector<int> m_verts_start_idx;
            std::vector<int> m_tri_start_idx;
            std::vector<int> m_ctrl_verts_start_idx;
            std::vector<int> m_ctrl_verts_idx;

            size_t m_total_vertices = 0;
            size_t m_total_triangles = 0;
            size_t m_total_edges = 0;
            size_t m_total_surface_vertices = 0;
            size_t m_total_surface_edges = 0;
        };

    } // namespace fem
} // namespace fsi