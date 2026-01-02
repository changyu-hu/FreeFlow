// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

// 前向声明具体的2D/3D类
namespace fsi {

namespace lbm {

class LbmFlowField2D;
class LbmFlowField3D;

} // namespace lbm

// ... 其他组件 ...
class SolidGeometry2D;
class SolidGeometry3D;

// 主模板
template <int DIM>
struct SimulationTraits;

// 2D特化
template <>
struct SimulationTraits<2> {
    // 定义2D仿真需要的所有类型别名
    using FlowFieldType = lbm::LbmFlowField2D;
    using SolidType = SolidGeometry2D;
    // ... 可以添加 LbmInitializerType, FsiSolverType 等 ...
};

// 3D特化
template <>
struct SimulationTraits<3> {
    // 定义3D仿真需要的所有类型别名
    using FlowFieldType = lbm::LbmFlowField3D;
    using SolidType = SolidGeometry3D;
    // ...
};