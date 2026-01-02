// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/fem/VBDSceneDataGpu.cuh"
#include "FSI_Simulator/control/LBSDataGpu.cuh"
#include "FSI_Simulator/common/CudaCommon.cuh"
#include "FSI_Simulator/common/Types.hpp"

namespace fsi
{
    namespace fem
    {
        void set_vel(VbdSceneDataGpu2D &data, vec2_t vel, cudaStream_t stream = 0);
        void add_pos(VbdSceneDataGpu2D &data, vec2_t shift, cudaStream_t stream = 0);

        void set_vel(VbdSceneDataGpu3D &data, vec3_t vel, cudaStream_t stream = 0);
        void add_pos(VbdSceneDataGpu3D &data, vec3_t shift, cudaStream_t stream = 0);

    }
}