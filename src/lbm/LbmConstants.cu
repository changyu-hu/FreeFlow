// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/lbm/LbmConstants.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include "FSI_Simulator/common/CudaCommon.cuh"

namespace fsi
{

    namespace lbm
    {

        // --- D3Q27 GPU Constant Memory ---
        namespace LbmD3Q27
        {
            __constant__ float c_ex[Q];
            __constant__ float c_ey[Q];
            __constant__ float c_ez[Q];
            __constant__ int c_inv[Q];
            __constant__ float c_w[Q];
            __constant__ float c_cs2;

            void uploadToGpu()
            {
                // Create a CPU-side constant instance
                const Constants host_constants;

                CUDA_CHECK(cudaMemcpyToSymbol(c_ex, host_constants.ex.data(), Q * sizeof(float)));

                CUDA_CHECK(cudaMemcpyToSymbol(c_ey, host_constants.ey.data(), Q * sizeof(float)));

                CUDA_CHECK(cudaMemcpyToSymbol(c_ez, host_constants.ez.data(), Q * sizeof(float)));

                CUDA_CHECK(cudaMemcpyToSymbol(c_inv, host_constants.inv.data(), Q * sizeof(int)));

                CUDA_CHECK(cudaMemcpyToSymbol(c_w, host_constants.w.data(), Q * sizeof(float)));

                CUDA_CHECK(cudaMemcpyToSymbol(c_cs2, &host_constants.cs2, sizeof(float)));

                LOG_INFO("D3Q27 LBM constants uploaded to GPU constant memory.");
            }
        } // namespace LbmD3Q27

        //--- D2Q9 GPU Constant Memory ---
        namespace LbmD2Q9
        {
            __constant__ float c_ex[Q];
            __constant__ float c_ey[Q];
            __constant__ int c_inv[Q];
            __constant__ float c_w[Q];
            __constant__ float c_cs2;

            void uploadToGpu()
            {
                // Create a CPU-side constant instance
                const Constants host_constants;

                CUDA_CHECK(cudaMemcpyToSymbol(c_ex, host_constants.ex.data(), Q * sizeof(float)));

                CUDA_CHECK(cudaMemcpyToSymbol(c_ey, host_constants.ey.data(), Q * sizeof(float)));

                CUDA_CHECK(cudaMemcpyToSymbol(c_inv, host_constants.inv.data(), Q * sizeof(int)));

                CUDA_CHECK(cudaMemcpyToSymbol(c_w, host_constants.w.data(), Q * sizeof(float)));

                CUDA_CHECK(cudaMemcpyToSymbol(c_cs2, &host_constants.cs2, sizeof(float)));

                LOG_INFO("D2Q9 LBM constants uploaded to GPU constant memory.");
            }
        } // namespace LbmD2Q9

    } // namespace lbm

} // namespace fsi