// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include <vector>
#include <array>

namespace fsi
{
    enum class VtkCellType : unsigned char
    {
        Line = 3,
        Triangle = 5,
        Tetra = 10,
    };

    template <int N>
    struct MeshElement
    {
        std::array<int, N> vertex_indices;
    };

    using LineElement = MeshElement<2>;
    using TriangleElement = MeshElement<3>;
    using TetraElement = MeshElement<4>;

} // namespace fsi