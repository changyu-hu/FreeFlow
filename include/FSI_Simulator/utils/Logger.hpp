// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h> // for ostream support
#include <memory>

#include "FSI_Simulator/common/CudaCommon.cuh"
#include <Eigen/Core>
#include <glm/glm.hpp>

template <int D, typename T, glm::qualifier Q>
struct fmt::formatter<glm::vec<D, T, Q>>
{
    constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const glm::vec<D, T, Q> &v, FormatContext &ctx) const
    {
        auto out = ctx.out();
        out = fmt::format_to(out, "[");
        for (int i = 0; i < D; ++i)
        {
            out = fmt::format_to(out, "{:.4f}{}", v[i], (i < D - 1) ? ", " : "");
        }
        out = fmt::format_to(out, "]");
        return out;
    }
};

template <int C, int R, typename T, glm::qualifier Q>
struct fmt::formatter<glm::mat<C, R, T, Q>>
{
    constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const glm::mat<C, R, T, Q> &m, FormatContext &ctx) const
    {
        auto out = ctx.out();
        out = fmt::format_to(out, "\n[\n");
        for (int i = 0; i < R; ++i)
        {
            out = fmt::format_to(out, "  [");
            for (int j = 0; j < C; ++j)
            {
                out = fmt::format_to(out, "{: .4f}{}", m[j][i], (j < C - 1) ? ", " : "");
            }
            out = fmt::format_to(out, "]\n");
        }
        out = fmt::format_to(out, "]");
        return out;
    }
};

template <typename T, int R, int C, int O, int MR, int MC>
struct fmt::formatter<Eigen::Matrix<T, R, C, O, MR, MC>>
{
    constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const Eigen::Matrix<T, R, C, O, MR, MC> &m, FormatContext &ctx) const
    {
        auto out = ctx.out();
        if (m.cols() == 1)
        {
            out = fmt::format_to(out, "[");
            for (int i = 0; i < m.rows(); ++i)
            {
                out = fmt::format_to(out, "{:.4f}{}", m(i, 0), (i < m.rows() - 1) ? ", " : "");
            }
            out = fmt::format_to(out, "]");
        }
        else
        {
            out = fmt::format_to(out, "\n[\n");
            for (int i = 0; i < m.rows(); ++i)
            {
                out = fmt::format_to(out, "  [");
                for (int j = 0; j < m.cols(); ++j)
                {
                    out = fmt::format_to(out, "{: .4f}{}", m(i, j), (j < m.cols() - 1) ? ", " : "");
                }
                out = fmt::format_to(out, "]\n");
            }
            out = fmt::format_to(out, "]");
        }
        return out;
    }
};

template <>
struct fmt::formatter<cudaError_t> : fmt::formatter<const char *>
{
    auto format(cudaError_t err, format_context &ctx) const
    {
        return fmt::formatter<const char *>::format(cudaGetErrorString(err), ctx);
    }
};

namespace fsi
{
    class Logger
    {
    public:
        static void init(const std::string &level_str = "info", const std::string &log_filepath = "simulation.log");

        static std::shared_ptr<spdlog::logger> &getCoreLogger() { return s_CoreLogger; }

    private:
        static std::shared_ptr<spdlog::logger> s_CoreLogger;
    };

#define LOG_TRACE(fmt, ...) Logger::getCoreLogger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::trace, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) Logger::getCoreLogger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::info, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) Logger::getCoreLogger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::warn, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) Logger::getCoreLogger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::err, fmt, ##__VA_ARGS__)
#define LOG_CRITICAL(fmt, ...) Logger::getCoreLogger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::critical, fmt, ##__VA_ARGS__)

#ifdef NDEBUG // Release mode
#define ASSERT(condition, fmt, ...) ((void)0)
#else // Debug mode
#define ASSERT(condition, fmt, ...)                            \
    if (!(condition))                                          \
    {                                                          \
        LOG_CRITICAL("Assertion Failed: " fmt, ##__VA_ARGS__); \
        std::abort();                                          \
    }
#endif

} // namespace fsi