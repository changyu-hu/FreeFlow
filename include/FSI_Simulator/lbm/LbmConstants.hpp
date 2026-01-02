// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include <array>
#include <vector>

namespace fsi {

namespace lbm {
enum class LbmModelType {
    D2Q9,
    D3Q27
};

namespace LbmD3Q27 {
    constexpr int Q = 27;

    struct Constants {
        std::array<float, Q> ex = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1 };
        std::array<float, Q> ey = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1 };
        std::array<float, Q> ez = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1 };
        std::array<int, Q>   inv = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25 };
        std::array<float, Q> w = {
            8.0f / 27.0f,
            2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f,
            1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f,
            1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f,
            1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f,
            1.0f / 216.0f, 1.0f / 216.0f
        };
        float cs2 = 1.0f / 3.0f;
    };
    
    void uploadToGpu();

} // namespace LbmD3Q27


namespace LbmD2Q9 {
    constexpr int Q = 9;
    
    struct Constants {
        std::array<float, Q> ex = { 0, 1, 0,-1, 0, 1,-1,-1, 1 };
        std::array<float, Q> ey = { 0, 0, 1, 0,-1, 1, 1,-1,-1 };
        std::array<int, Q>  inv = { 0, 3, 4, 1, 2, 7, 8, 5, 6 };
        std::array<float, Q> w = { 
            4.0f / 9.0f, 
            1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
            1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };
        float cs2 = 1.0f / 3.0f;
    };

    void uploadToGpu();
} // namespace LbmD2Q9

} // namespace lbm

} // namespace fsi