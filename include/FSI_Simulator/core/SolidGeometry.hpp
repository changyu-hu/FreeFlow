// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/utils/CudaArray.cuh"
#include "SolidGeometryProxy_Device.cuh"
#include <vector>
#include <string>

namespace fsi {

// 主机端几何管理类
template <int DIM>
class SolidGeometry {
public:
    // 从网格文件和配置中构造
    SolidGeometry(const std::string& mesh_path, const SolidConfig& config);
    ~SolidGeometry() = default; // RAII，不需要手动释放

    // 更新GPU上的顶点位置
    void updatePositions(const std::vector<float>& host_vertices);

    // 更新GPU上的顶点速度
    void updateVelocities(const std::vector<float>& host_velocities);

    // 从GPU取回计算出的流固耦合力
    std::vector<float> getForces();

    // 重置GPU上的力数组为0
    void resetForces();

    // 创建并返回一个设备端代理的实例
    // 这个代理可以被传递给CUDA内核
    SolidGeometryProxy_Device<DIM> getDeviceProxy();

private:
    void loadMesh(const std::string& mesh_path);

    // --- GPU 内存 ---
    // 使用RAII封装来管理GPU内存
    CudaArray<float> d_vertices;
    CudaArray<float> d_velocities;
    CudaArray<int>   d_elements;
    CudaArray<float> d_forces;

    // --- 主机端元数据 ---
    int m_num_vertices;
    int m_num_elements;
    int m_num_dofs;
};

} // namespace fsi