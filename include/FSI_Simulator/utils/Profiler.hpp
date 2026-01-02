// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include <string>
#include <chrono>
#include <map>
#include <mutex>
#include <memory>

#include "FSI_Simulator/common/CudaCommon.cuh"

namespace fsi
{
// --- Profiler Macros for easy use ---
// Use these in your code. They handle unique variable names automatically.
#define PROFILE_SESSION(name) Profiler::get().beginSession(name)
#define PROFILE_END_SESSION() Profiler::get().endSession()

#define PROFILE_FUNCTION() ProfileTimer ANONYMOUS_VARIABLE_LINE(__profiler_timer_)(__FUNCTION__)
#define PROFILE_SCOPE(name) ProfileTimer ANONYMOUS_VARIABLE_LINE(__profiler_timer_)(name)
#define PROFILE_CUDA_SCOPE(name, stream) ProfileCudaTimer ANONYMOUS_VARIABLE_LINE(__profiler_cuda_timer_)(name, stream)

// Helper macro to create unique variable names
#define PASTE_HELPER(a, b) a##b
#define PASTE(a, b) PASTE_HELPER(a, b)
#define ANONYMOUS_VARIABLE_LINE(name) PASTE(name, __LINE__)

    // Forward declarations for timer classes used by macros
    class ProfileTimer;
    class ProfileCudaTimer;

    enum class ProfileType
    {
        CPU,
        GPU
    };

    struct ProfileResult
    {
        std::string name;
        ProfileType type;
        long long count = 0;
        double totalTime = 0.0;
        double minTime = std::numeric_limits<double>::max();
        double maxTime = 0.0;
    };

    class Profiler
    {
    public:
        // Singleton access
        static Profiler &get()
        {
            static Profiler instance;
            return instance;
        }

        // Disable copy/move
        Profiler(const Profiler &) = delete;
        Profiler &operator=(const Profiler &) = delete;

        void beginSession(const std::string &name);
        void endSession();

        void submitResult(const ProfileResult &result);

    private:
        Profiler() = default;
        ~Profiler() = default;

        void printResults();

        std::mutex m_Mutex;
        std::string m_CurrentSessionName = "Untitled";
        std::map<std::string, ProfileResult> m_Results;
    };

    // --- RAII Timer Classes ---
    // These classes should not be used directly. Use the macros above.

    class ProfileTimer
    { // For CPU
    public:
        ProfileTimer(const char *name);
        ~ProfileTimer();

    private:
        const char *m_Name;
        std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
    };

    class ProfileCudaTimer
    { // For GPU
    public:
        ProfileCudaTimer(const char *name, cudaStream_t stream);
        ~ProfileCudaTimer();

    private:
        const char *m_Name;
        cudaStream_t m_Stream;
        cudaEvent_t m_Start, m_Stop;
    };

} // namespace fsi