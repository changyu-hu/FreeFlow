// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/lbm/LbmFlowField2D.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include "FSI_Simulator/lbm/LbmConstants.hpp"

namespace fsi
{

    namespace lbm
    {

        LbmFlowField2D::LbmFlowField2D(const SimulationParameters2D &params)
        {

            m_viscosity = params.fluid_viscosity;
            m_nx = params.fluid_nx;
            m_ny = params.fluid_ny;
            m_num_nodes = (long)m_nx * m_ny;

            LOG_INFO("Creating LbmFlowField2D with grid size {}x{}", m_nx, m_ny);

            // GPU memory allocation
            m_flags.resize(m_num_nodes);
            m_fMom.resize(m_num_nodes * 6);
            m_fMomPost.resize(m_num_nodes * 6);
            m_fluidForce.resize(m_num_nodes * 2);
            m_fluidForce.setZero();
            m_hitIndex.resize(m_num_nodes * 9);

            m_current_moments = &m_fMom;
            m_next_moments = &m_fMomPost;

            LbmD2Q9::uploadToGpu();
            LOG_INFO("LbmFlowField2D GPU memory allocated successfully.");
        }

        void LbmFlowField2D::reset()
        {
            m_current_moments = &m_fMom;
            m_next_moments = &m_fMomPost;
            m_fluidForce.setZero();
        }

    } // namespace lbm

} // namespace fsi