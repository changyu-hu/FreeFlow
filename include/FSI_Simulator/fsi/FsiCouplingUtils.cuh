// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/fem/FemScene.hpp"
#include <cuda_runtime.h>

namespace fsi
{
    namespace coupling
    {

        void updateSurfaceStatesFromVolume(
            fem::VbdSceneDataGpu3D &scene_data,
            FsiCouplingDataGpu &coupling_data,
            size_t num_surface_vertices,
            cudaStream_t stream);

        void updateSurfaceStatesFromVolume(
            fem::VbdSceneDataGpu2D &scene_data,
            FsiCouplingDataGpu &coupling_data,
            size_t num_surface_vertices,
            cudaStream_t stream);

        void scatterSurfaceForcesToVolume(
            fem::VbdSceneDataGpu3D &scene_data,
            FsiCouplingDataGpu &coupling_data,
            size_t num_surface_vertices,
            cudaStream_t stream);

        void scatterSurfaceForcesToVolume(
            fem::VbdSceneDataGpu2D &scene_data,
            FsiCouplingDataGpu &coupling_data,
            size_t num_surface_vertices,
            cudaStream_t stream);

    } // namespace coupling
} // namespace fsi