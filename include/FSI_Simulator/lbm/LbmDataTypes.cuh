// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/common/CudaCommon.cuh"

namespace fsi
{

    namespace lbm
    {

        enum class LbmNodeFlag : unsigned char
        {
            Invalid = 0,

            // --- Fluid Node ---
            Fluid,
            FluidRest,

            // --- Solid/Boundary Node ---
            Solid,
            SolidDynamic,
            Wall,

            WallLeft,
            WallRight,
            WallFront,
            WallBack,
            WallDown,
            WallUp,
            WallCorner,

            InletLeft,
            InletRight,
            InletUp,
            InletDown,
            InletFront,
            InletBack,

            OutletLeft,
            OutletRight,
            OutletUp,
            OutletDown,
            OutletFront,
            OutletBack,

            // --- Special Node ---
            Receptor,
            Deactivated,
            Smoke,
        };

        struct LbmMoments_Device
        {
            float *m_pre;
            float *m_post;
        };

    } // namespace lbm

} // namespace fsi