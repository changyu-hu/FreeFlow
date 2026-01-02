// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <iterator> // For std::ostream_iterator

namespace fsi
{
    namespace utils
    {

        template <typename T>
        std::string join(const T &container, const std::string &delimiter = ", ")
        {
            if (container.empty())
            {
                return "";
            }

            std::stringstream ss;
            // 使用 ostream_iterator 将容器内容优雅地输出到流中
            // C++11 风格
            auto it = std::begin(container);
            ss << *it;
            for (++it; it != std::end(container); ++it)
            {
                ss << delimiter << *it;
            }

            return ss.str();
        }

    } // namespace utils
} // namespace fsi