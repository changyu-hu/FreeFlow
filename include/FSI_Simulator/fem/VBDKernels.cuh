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
        void predictPositions(VbdSceneDataGpu3D &data, real dt, cudaStream_t stream);
        void solveTetrahedronConstraints(VbdSceneDataGpu3D &data, real dt, int itr_idx, real itr_omega, cudaStream_t stream);
        void updateVelocitiesAndPositions(VbdSceneDataGpu3D &data, real dt, cudaStream_t stream);
        void solveLBSDynamicCorrection(VbdSceneDataGpu3D &data, control::LBSDataGpu3D &lbs_data, int itr_idx, real itr_omega, cudaStream_t stream);

        void predictPositions(VbdSceneDataGpu2D &data, real dt, cudaStream_t stream);
        void solveConstraints(VbdSceneDataGpu2D &data, real dt, int itr_idx, real itr_omega, cudaStream_t stream);
        void updateVelocitiesAndPositions(VbdSceneDataGpu2D &data, real dt, cudaStream_t stream);
        void solveLBSDynamicCorrection(VbdSceneDataGpu2D &data, control::LBSDataGpu2D &lbs_data, int itr_idx, real itr_omega, cudaStream_t stream);
    }
}
