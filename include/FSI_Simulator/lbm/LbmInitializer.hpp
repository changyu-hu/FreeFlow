// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

class LbmFlowField2D;
class LbmFlowField3D;
class SimulationParameters2D;
class SimulationParameters3D;

namespace fsi
{

    namespace lbm
    {

        namespace LbmInitializer
        {
            void initializeState2D(LbmFlowField2D &flow_field, const SimulationParameters2D &params);

            void initializeState3D(LbmFlowField3D &flow_field, const SimulationParameters3D &params);
        } // namespace LbmInitializer

    } // namespace lbm

} // namespace fsi