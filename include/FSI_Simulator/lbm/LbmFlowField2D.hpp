// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/utils/CudaArray.cuh"
#include "FSI_Simulator/lbm/LbmDataTypes.cuh"
#include "FSI_Simulator/utils/Config.hpp"

namespace fsi
{

    namespace lbm
    {

        class LbmFlowField2D
        {
        public:
            // 从仿真参数构造
            explicit LbmFlowField2D(const SimulationParameters2D &params);
            ~LbmFlowField2D() = default; // RAII，CudaArray会自动处理

            void reset();

            // --- 提供对设备端数据的访问 ---
            LbmMoments_Device getMoments_Device();

            // 获取格点标志数组
            CudaArray<LbmNodeFlag> &getFlags() { return m_flags; }
            const CudaArray<LbmNodeFlag> &getFlags() const { return m_flags; }

            // --- 查询信息 ---
            long getNumNodes() const { return m_num_nodes; }
            int getNx() const { return m_nx; }
            int getNy() const { return m_ny; }

            void swapMoments()
            {
                std::swap(m_current_moments, m_next_moments);
            }

        public:
            // --- Grid Dimensions ---
            long m_num_nodes;
            int m_nx, m_ny;
            float m_viscosity;

            // --- GPU Data Arrays ---
            // 使用 CudaArray 进行自动内存管理
            CudaArray<LbmNodeFlag> m_flags;
            CudaArray<float> m_fMom;
            CudaArray<float> m_fMomPost;
            CudaArray<float> *m_current_moments;
            CudaArray<float> *m_next_moments;
            CudaArray<float> m_fluidForce;
            CudaArray<uint32_t> m_hitIndex;
        };

    } // namespace lbm

} // namespace fsi