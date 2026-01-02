// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/utils/Logger.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <vector>
#include <map>
#include <iostream>

namespace fsi {
std::shared_ptr<spdlog::logger> Logger::s_CoreLogger;

void Logger::init(const std::string& level_str, const std::string& log_filepath) {
    const std::map<std::string, spdlog::level::level_enum> level_map = {
        {"trace", spdlog::level::trace},
        {"debug", spdlog::level::debug},
        {"info", spdlog::level::info},
        {"warn", spdlog::level::warn},
        {"error", spdlog::level::err},
        {"critical", spdlog::level::critical},
        {"off", spdlog::level::off}
    };

    auto it = level_map.find(level_str);
    spdlog::level::level_enum log_level = spdlog::level::info; // 默认值
    if (it != level_map.end()) {
        log_level = it->second;
    } else {
        // 如果用户提供了无效的级别字符串，可以打印一个警告
        // 注意：此时Logger可能还未完全初始化，所以用std::cerr
        std::cerr << "[Logger Warning] Invalid log level string '" << level_str 
                  << "'. Defaulting to 'info'." << std::endl;
    }

    // --- 创建sinks ---
    std::vector<spdlog::sink_ptr> sinks;
    // 控制台 sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    
#ifdef NDEBUG
    // --- RELEASE模式 ---
    // 简洁美观的格式: [时间] [级别] 日志信息
    console_sink->set_pattern("%^[%T] [%l] %v%$");
    // 在Release模式下，默认将控制台的日志级别提高到info，除非用户在config中指定了更低的级别
    spdlog::level::level_enum console_level = std::max(log_level, spdlog::level::info);
    console_sink->set_level(console_level);
#else
    // --- DEBUG模式 ---
    // 详细的格式: [时间] [级别] [线程ID] 日志信息 (文件名:行号)
    console_sink->set_pattern("%^[%T.%e] [%l] [thread %t] %v%$ %@");
    // 在Debug模式下，使用用户在config中指定的级别
    console_sink->set_level(log_level);
#endif

    // 文件 sink (使用从config传入的文件路径)
    // 第二个参数 `true` 表示每次运行都清空旧的日志文件
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_filepath, true);
    file_sink->set_pattern("[%Y-%m-%d %T.%e] [%l] [thread %t] [%s:%# %!()] %v");

    sinks.push_back(console_sink);
    sinks.push_back(file_sink);

    s_CoreLogger = std::make_shared<spdlog::logger>("FSI_SIM", begin(sinks), end(sinks));
    spdlog::register_logger(s_CoreLogger);
    
    // 设置从配置中读取的日志级别
    s_CoreLogger->set_level(log_level);
    s_CoreLogger->flush_on(spdlog::level::trace); // 确保所有信息都立即写入

    // --- 注册一个全局的 atexit 或析构函数来确保日志被刷新 ---
    // spdlog 提供了自动注册的功能
    spdlog::set_automatic_registration(true);
}

} // namespace fsi