// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/utils/Profiler.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include <iomanip>
#include <algorithm>
#include <vector>
#include <sstream> // 使用 stringstream 来构建多行日志消息

namespace fsi {

// --- Profiler Implementation ---

void Profiler::beginSession(const std::string& name) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    m_CurrentSessionName = name;
    m_Results.clear();
    LOG_INFO("Profiler session started: '{}'", m_CurrentSessionName);
}

void Profiler::endSession() {
    std::lock_guard<std::mutex> lock(m_Mutex);
    printResults();
    m_Results.clear();
    LOG_INFO("Profiler session ended: '{}'", m_CurrentSessionName);
}

void Profiler::submitResult(const ProfileResult& result) {
    std::lock_guard<std::mutex> lock(m_Mutex);
    
    auto it = m_Results.find(result.name);
    if (it == m_Results.end()) {
        m_Results[result.name] = result;
    } else {
        it->second.count++;
        it->second.totalTime += result.totalTime;
        it->second.minTime = std::min(it->second.minTime, result.totalTime);
        it->second.maxTime = std::max(it->second.maxTime, result.totalTime);
    }
}

void Profiler::printResults() {
    // 使用 stringstream 来构建一个多行的字符串，然后一次性通过 Logger 输出
    // 这样可以避免多条日志消息被其他线程的日志打断
    std::stringstream report;
    
    report << std::fixed << std::setprecision(3);
    report << "\n\n" // Add some newlines to make it stand out
           << "==================== Profiler Report: " << m_CurrentSessionName << " ====================\n"
           << std::left << std::setw(40) << "Scope Name"
           << std::setw(8) << "Type"
           << std::setw(12) << "Avg (ms)"
           << std::setw(12) << "Total (ms)"
           << std::setw(12) << "Min (ms)"
           << std::setw(12) << "Max (ms)"
           << std::setw(10) << "Calls" << "\n"
           << std::string(105, '-') << "\n";
    
    std::vector<ProfileResult> sorted_results;
    for (const auto& pair : m_Results) {
        sorted_results.push_back(pair.second);
    }
    std::sort(sorted_results.begin(), sorted_results.end(), [](const auto& a, const auto& b) {
        return a.totalTime > b.totalTime;
    });

    for (const auto& result : sorted_results) {
        double avgTime = result.count > 0 ? result.totalTime / result.count : 0.0;
        const char* typeStr = (result.type == ProfileType::GPU) ? "GPU" : "CPU";

        report << std::left << std::setw(40) << result.name
               << std::setw(8) << typeStr
               << std::setw(12) << avgTime
               << std::setw(12) << result.totalTime
               << std::setw(12) << result.minTime
               << std::setw(12) << result.maxTime
               << std::setw(10) << result.count << "\n";
    }

    report << "========================================================================================\n";
    
    // 使用一个不带格式化的 LOG_INFO 调用来输出整个报告
    // spdlog 默认会处理多行字符串
    LOG_INFO("{}", report.str());
}


// --- Timer Implementations (No changes needed here) ---

ProfileTimer::ProfileTimer(const char* name)
    : m_Name(name) {
    m_StartTimepoint = std::chrono::high_resolution_clock::now();
}

ProfileTimer::~ProfileTimer() {
    auto endTimepoint = std::chrono::high_resolution_clock::now();
    long long start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch().count();
    long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();
    double duration = (end - start) * 0.001;
    Profiler::get().submitResult({m_Name, ProfileType::CPU, 1, duration});
}

ProfileCudaTimer::ProfileCudaTimer(const char* name, cudaStream_t stream)
    : m_Name(name), m_Stream(stream) {
    cudaEventCreate(&m_Start);
    cudaEventCreate(&m_Stop);
    cudaEventRecord(m_Start, m_Stream);
}

ProfileCudaTimer::~ProfileCudaTimer() {
    cudaEventRecord(m_Stop, m_Stream);
    cudaEventSynchronize(m_Stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_Start, m_Stop);
    Profiler::get().submitResult({m_Name, ProfileType::GPU, 1, (double)milliseconds});
    cudaEventDestroy(m_Start);
    cudaEventDestroy(m_Stop);
}

} // namespace fsi