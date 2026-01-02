// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/lbm/LbmFlowField3D.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include "FSI_Simulator/io/VtkWriter.hpp"
#include "FSI_Simulator/lbm/LbmConstants.hpp"

namespace fsi
{

    namespace lbm
    {

        LbmFlowField3D::LbmFlowField3D(const SimulationParameters3D &params)
        {

            m_viscosity = params.fluid_viscosity;
            m_nx = params.fluid_nx;
            m_ny = params.fluid_ny;
            m_nz = params.fluid_nz;
            m_num_nodes = (long)m_nx * m_ny * m_nz;
            m_dx = params.fluid_dx;
            LOG_INFO("Creating LbmFlowField3D with grid size {}x{}x{}", m_nx, m_ny, m_nz);

            // GPU memory allocation
            m_flags.resize(m_num_nodes);
            m_moments_pre.resize(m_num_nodes * 10);
            m_moments_post.resize(m_num_nodes * 10);
            m_current_moments = &m_moments_pre;
            m_next_moments = &m_moments_post;
            m_fluidForce.resize(m_num_nodes * 3);
            m_fluidForce.setZero();
            m_hitIndex.resize(m_num_nodes * 27);

            LbmD3Q27::uploadToGpu();

            LOG_INFO("LbmFlowField3D GPU memory allocated successfully.");
        }

        void LbmFlowField3D::reset()
        {
            m_current_moments = &m_moments_pre;
            m_next_moments = &m_moments_post;
            m_fluidForce.setZero();
        }

        void LbmFlowField3D::saveFrameData(std::string filepath) const
        {
            // save velocity field
            auto moments = m_moments_pre.download();
            auto size = m_nx * m_ny * m_nz;
            std::vector<float> vx(m_nx * m_ny * m_nz);
            std::vector<float> vy(m_nx * m_ny * m_nz);
            std::vector<float> vz(m_nx * m_ny * m_nz);
            for (int i = 0; i < m_nx; ++i)
            {
                for (int j = 0; j < m_ny; ++j)
                {
                    for (int k = 0; k < m_nz; ++k)
                    {
                        vx[k * m_nx * m_ny + j * m_nx + i] = moments[10 * (k * m_nx * m_ny + j * m_nx + i) + 1];
                        vy[k * m_nx * m_ny + j * m_nx + i] = moments[10 * (k * m_nx * m_ny + j * m_nx + i) + 2];
                        vz[k * m_nx * m_ny + j * m_nx + i] = moments[10 * (k * m_nx * m_ny + j * m_nx + i) + 3];
                    }
                }
            }

            io::VtkWriter::writeImageDataVectorField(filepath,
                                                     vx.data(), vy.data(), vz.data(),
                                                     m_nx, m_ny, m_nz, "velocity", glm::vec3(m_dx), glm::vec3(0.0f));
        }

    } // namespace lbm

} // namespace fsi