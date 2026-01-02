// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/fem/FemScene.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include "FSI_Simulator/utils/Profiler.hpp"
#include "FSI_Simulator/fem/FemConfig.hpp"

namespace fsi
{
    namespace fem
    {

        void solveNewtonFEMStep(NewtonSceneDataCpu &cpu_data, real dt, const FemSolverOptions &options);

        void solveNewtonDynamicCorrectionStep(NewtonSceneDataCpu &cpu_data, control::LBSDataGpu3D &lbs_control_data, const FemSolverOptions &options);

        void solveNewtonDynamicCorrectionStepGpu(NewtonSceneDataCpu &cpu_data, VbdSceneDataGpu3D &gpu_data, control::LBSDataGpu3D &lbs_control_data, const FemSolverOptions &options, cudaStream_t stream);

        void solveNewtonDynamicCorrectionStepGpu(NewtonSceneDataCpu &cpu_data, VbdSceneDataGpu2D &gpu_data, control::LBSDataGpu2D &lbs_control_data, const FemSolverOptions &options, cudaStream_t stream);

    } // namespace fem
} // namespace fsi