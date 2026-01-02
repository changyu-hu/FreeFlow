// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/common/CudaCommon.cuh"

#include "FSI_Simulator/utils/CudaArray.cuh"

namespace fsi
{
    namespace coupling
    {

        struct FsiCouplingDataGpu
        {
            // --- surface data (for FSI) ---
            CudaArray<float> surface_positions;
            CudaArray<float> surface_velocities;
            CudaArray<unsigned int> surface_elements_indices;
            CudaArray<float> surface_forces;

            // --- mapping: surface to volume (on GPU) ---
            // This array has length equal to the total number of surface vertices.
            // d_surface_to_volume_map[i] is the index of the global volume vertex corresponding to the i-th surface vertex.
            CudaArray<unsigned int> d_surface_to_volume_map;
        };
    } // namespace coupling
} // namespace fsi