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

        real ComputeElementElasticEnergyGpu(
            const Eigen::Matrix<real, 3, Eigen::Dynamic> &position, VbdSceneDataGpu3D &scene_data, control::LBSDataGpu3D &lbs_data, cudaStream_t stream);
        Eigen::Matrix<real, 3, Eigen::Dynamic> ComputeElasticForceGpu(
            const Eigen::Matrix<real, 3, Eigen::Dynamic> &position, VbdSceneDataGpu3D &scene_data, control::LBSDataGpu3D &lbs_data, cudaStream_t stream);
        std::pair<SparseMatrixXr, SparseMatrixXr> ComputeElasticHessianAndProjectionGpu(
            const Eigen::Matrix<real, 3, Eigen::Dynamic> &position, VbdSceneDataGpu3D &scene_data, control::LBSDataGpu3D &lbs_data,
            const bool compute_hessian = true, const bool compute_projection = false, cudaStream_t stream = 0);

        real ComputeElementElasticEnergyGpu(
            const Eigen::Matrix<real, 2, Eigen::Dynamic> &position, VbdSceneDataGpu2D &scene_data, control::LBSDataGpu2D &lbs_data, cudaStream_t stream);
        Eigen::Matrix<real, 2, Eigen::Dynamic> ComputeElasticForceGpu(
            const Eigen::Matrix<real, 2, Eigen::Dynamic> &position, VbdSceneDataGpu2D &scene_data, control::LBSDataGpu2D &lbs_data, cudaStream_t stream);
        std::pair<SparseMatrixXr, SparseMatrixXr> ComputeElasticHessianAndProjectionGpu(
            const Eigen::Matrix<real, 2, Eigen::Dynamic> &position, VbdSceneDataGpu2D &scene_data, control::LBSDataGpu2D &lbs_data,
            const bool compute_hessian = true, const bool compute_projection = false, cudaStream_t stream = 0);
    }
}