// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "builtin_types.h"
#include "device_launch_parameters.h"

#include <cublas_v2.h>

#pragma push
#if defined(__NVCC__)
#pragma nv_diag_suppress 20012
#elif defined(_MSC_VER)
#endif

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#pragma pop
#if defined(__NVCC__)
#elif defined(_MSC_VER)
#endif

#include "FSI_Simulator/utils/CudaErrorCheck.cuh"

namespace KernelConfig
{

    // --- (Block Size) ---

    constexpr int BLOCK_SIZE_1D = 512;

    constexpr int VBD3D_THREAD_DIM_FORTET = 64;

    constexpr int VBD2D_THREAD_DIM_FORTRI = 128;

    // the number of threads in 2D block
    constexpr int BLOCK_SIZE_2D_X = 16;
    constexpr int BLOCK_SIZE_2D_Y = 16;

    // the number of threads in 3D block
    constexpr int BLOCK_SIZE_3D_X = 8;
    constexpr int BLOCK_SIZE_3D_Y = 4;
    constexpr int BLOCK_SIZE_3D_Z = 4;
    constexpr int SHAREDMEM_LBM_SIZE_3D = (BLOCK_SIZE_3D_X + 2) * (BLOCK_SIZE_3D_Y + 2) * (BLOCK_SIZE_3D_Z + 2);

} // namespace KernelConfig