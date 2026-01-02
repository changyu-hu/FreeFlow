// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/lbm/LbmFlowField2D.hpp"
#include "FSI_Simulator/lbm/LbmFlowField3D.hpp"
#include "FSI_Simulator/core/SolidGeometryProxy_Device.cuh"

namespace fsi
{

    namespace lbm
    {

        namespace fsi_coupling
        {
            void solveLbmAndFsiStep2D(
                LbmFlowField2D &flow_field,
                const SolidGeometryProxy_Device<2> &solid_proxy,
                const SimulationParameters2D &params,
                cudaStream_t stream);

            void fillSolidFlags2D(
                LbmFlowField2D &flow_field,
                const SolidGeometryProxy_Device<2> &solid_proxy,
                const SimulationParameters2D &params,
                cudaStream_t stream);

            // 3D版本的函数
            void solveLbmAndFsiStep3D(
                lbm::LbmFlowField3D &flow_field,
                const SolidGeometryProxy_Device<3> &solid_proxy,
                const SimulationParameters3D &params,
                cudaStream_t stream);
        }

    } // namespace lbm

} // namespace fsi