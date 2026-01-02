// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/utils/Logger.hpp" // 引入日志系统

#include "FSI_Simulator/common/CudaCommon.cuh"

#include <stdexcept>

namespace fsi
{
#ifndef NDEBUG
#define CUDA_CHECK(call)                                    \
    do                                                      \
    {                                                       \
        cudaError_t err = call;                             \
        if (err != cudaSuccess)                             \
        {                                                   \
            LOG_CRITICAL("CUDA Error in {}:{} : {}",        \
                         __FILE__, __LINE__, err);          \
            throw std::runtime_error("CUDA runtime error"); \
        }                                                   \
    } while (0)
#else
#define CUDA_CHECK(call) call
#endif

#ifndef NDEBUG
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#else
#define CUDA_CHECK_KERNEL()
#endif

} // namespace fsi