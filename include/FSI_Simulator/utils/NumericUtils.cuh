// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include <cmath>
#include <type_traits>

#include "FSI_Simulator/common/CudaCommon.cuh"

#if defined(__CUDA_ARCH__)
#include <cfloat>
#endif

namespace fsi
{
    namespace NumericUtils
    {

        template <typename T>
        static inline constexpr __device__ T Epsilon = std::is_same_v<T, float> ? 1e-6f : 1e-9;

        /**
         * @brief check if a is approximately equal to b  (|a - b| <= epsilon)
         */
        template <typename T>
        [[nodiscard]] __host__ __device__ inline bool isApproxEqual(T a, T b, T epsilon = Epsilon<T>)
        {
            return std::abs(a - b) <= epsilon;
        }

        /**
         * @brief check if a is approximately zero (|a| <= epsilon)
         */
        template <typename T>
        [[nodiscard]] __host__ __device__ inline bool isApproxZero(T a, T epsilon = Epsilon<T>)
        {
            return std::abs(a) <= epsilon;
        }

        /**
         * @brief check if a is approximately less than b (a < b - epsilon)
         */
        template <typename T>
        [[nodiscard]] __host__ __device__ inline bool isApproxLess(T a, T b, T epsilon = Epsilon<T>)
        {
            return a < b - epsilon;
        }

        /**
         * @brief check if a is approximately less than or equal to b (a <= b + epsilon)
         */
        template <typename T>
        [[nodiscard]] __host__ __device__ inline bool isApproxLessEqual(T a, T b, T epsilon = Epsilon<T>)
        {
            return a <= b + epsilon;
        }

        /**
         * @brief check if a is approximately greater than b (a > b + epsilon)
         */
        template <typename T>
        [[nodiscard]] __host__ __device__ inline bool isApproxGreater(T a, T b, T epsilon = Epsilon<T>)
        {
            return a > b + epsilon;
        }

        /**
         * @brief check if a is approximately greater than or equal to b (a >= b - epsilon)
         */
        template <typename T>
        [[nodiscard]] __host__ __device__ inline bool isApproxGreaterEqual(T a, T b, T epsilon = Epsilon<T>)
        {
            return a >= b - epsilon;
        }

        /**
         * @brief check if a is approximately less than or equal to b
         *        considering both absolute and relative tolerances
         *        |a - b| <= abs_epsilon || |a - b| <= rel_epsilon * max(|a|, |b|)
         */
        template <typename T>
        [[nodiscard]] __host__ __device__ inline bool isAlmostLessEqual(T a, T b,
                                                                        T abs_epsilon = Epsilon<T>,
                                                                        T rel_epsilon = Epsilon<T>)
        {
            if (a <= b)
                return true;
            T diff = a - b; // a is greater than b
            return diff <= abs_epsilon || diff <= rel_epsilon * std::max(std::abs(a), std::abs(b));
        }

    } // namespace NumericUtils

} // namespace fsi