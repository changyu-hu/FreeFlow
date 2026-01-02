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

        class LbmFlowField3D
        {
        public:
            explicit LbmFlowField3D(const SimulationParameters3D &params);
            ~LbmFlowField3D() = default; // RAII

            void reset();
            LbmMoments_Device getMoments_Device();

            CudaArray<LbmNodeFlag> &getFlags() { return m_flags; }
            const CudaArray<LbmNodeFlag> &getFlags() const { return m_flags; }

            long getNumNodes() const { return m_num_nodes; }
            int getNx() const { return m_nx; }
            int getNy() const { return m_ny; }
            int getNz() const { return m_nz; }
            float getViscosity() const { return m_viscosity; }
            float getDx() const { return m_dx; }

            void swapMoments()
            {
                std::swap(m_current_moments, m_next_moments);
            }

            void saveFrameData(std::string filepath) const;

        public:
            // --- Grid Dimensions ---
            long m_num_nodes;
            int m_nx, m_ny, m_nz;
            float m_viscosity, m_dx;

            // --- GPU Data Arrays ---
            CudaArray<LbmNodeFlag> m_flags;
            CudaArray<float> m_moments_pre;
            CudaArray<float> m_moments_post;
            CudaArray<float> *m_current_moments;
            CudaArray<float> *m_next_moments;
            CudaArray<float> m_fluidForce;
            CudaArray<uint32_t> m_hitIndex;
        };

    } // namespace lbm

} // namespace fsi