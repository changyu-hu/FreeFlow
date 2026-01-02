// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/utils/Config.hpp"
#include <fstream>
#include <filesystem>

namespace fsi {

bool ConfigBase::loadBase(const std::string& filepath, nlohmann::json& data) {
    // 阶段1: Logger 不可用，使用 std::cerr
    if (!std::filesystem::exists(filepath)) {
        std::cerr << "[Config FATAL] Configuration file not found at path: '" << filepath << "'" << std::endl;
        return false;
    }

    std::ifstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "[Config FATAL] Could not open configuration file: '" << filepath << "'. Check permissions." << std::endl;
        return false;
    }

    // 尝试解析JSON，这是可能抛出异常的第一步
    try {
        data = nlohmann::json::parse(f);
    } catch (const nlohmann::json::exception& e) {
        std::cerr << "[Config FATAL] JSON parsing failed in file '" << filepath << "':\n    - Error: " << e.what() << std::endl;
        return false;
    }

    // 阶段2: JSON已解析，可以初始化Logger了
    // 从JSON中提取日志配置，如果找不到就使用默认值
    std::string log_level = data.value("log_level", "info");
    std::string output_path = data.value("output_path", "output");
    std::string log_file_path = output_path + "/simulation.log";
    
    // 初始化Logger
    Logger::init(log_level, log_file_path);
    
    // 现在可以使用Logger了！
    LOG_INFO("Logger initialized. Log level: '{}', Log file: '{}'", log_level, log_file_path);
    LOG_INFO("Loading configuration from: '{}'", filepath);

    return true;
}

bool Config2D::load(const std::string& filepath) {
    nlohmann::json data;
    if (!loadBase(filepath, data)) {
        return false;
    }

    // 阶段3: Logger可用，解析主要参数
    try {
        m_params = data.get<SimulationParameters2D>(); 
        LOG_INFO("2D simulation parameters parsed successfully.");
        
        postProcess();

    } catch (const nlohmann::json::exception& e) {
        // 现在可以用Logger来报告详细的验证错误
        LOG_CRITICAL("JSON validation failed for 2D parameters in file '{}':\n    - Error: {}", filepath, e.what());
        return false;
    }

    return true;
}

void Config2D::postProcess() {
    LOG_INFO("Post-processing 2D configuration...");
    // 验证边界条件
    for (const auto& bc : m_params.boundaries) {
        if (bc.position.size() != 2) {
            LOG_ERROR("Invalid position size for a boundary condition in 2D config. Expected 2, got {}.", bc.position.size());
            //可以抛出异常或设置错误状态
        }
        if (bc.flag == lbm::LbmNodeFlag::Invalid) {
            LOG_WARN("Boundary condition at [{}, {}] has an unrecognized type.", bc.position[0], bc.position[1]);
        }
    }
    // 验证速度数组大小
    if (!m_params.boundary_velocities.empty() && m_params.boundary_velocities.size() != 4) {
        LOG_WARN("`boundary_velocities` for 2D should have 4 elements. Found {}.", m_params.boundary_velocities.size());
    }
    LOG_INFO("Configuration validated.");
}

bool Config3D::load(const std::string& filepath) {
    nlohmann::json data;
    if (!loadBase(filepath, data)) {
        return false;
    }
    
    try {
        m_params = data.get<SimulationParameters3D>(); 
        LOG_INFO("3D simulation parameters parsed successfully.");
        
        postProcess();

    } catch (const nlohmann::json::exception& e) {
        LOG_CRITICAL("JSON validation failed for 3D parameters in file '{}':\n    - Error: {}", filepath, e.what());
        return false;
    }

    return true;
}

void Config3D::postProcess() {
    LOG_INFO("Post-processing 3D configuration...");
    // 验证边界条件
    for (const auto& bc : m_params.boundaries) {
        if (bc.position.size() != 3) {
            LOG_ERROR("Invalid position size for a boundary condition in 3D config. Expected 3, got {}.", bc.position.size());
        }
        if (bc.flag == lbm::LbmNodeFlag::Invalid) {
            LOG_WARN("Boundary condition at [{}, {}, {}] has an unrecognized type.", bc.position[0], bc.position[1], bc.position[2]);
        }
    }
    // 验证速度数组大小
    if (!m_params.boundary_velocities.empty() && m_params.boundary_velocities.size() != 6) {
        LOG_WARN("`boundary_velocities` for 3D should have 6 elements. Found {}.", m_params.boundary_velocities.size());
    }
    LOG_INFO("Configuration validated.");
}

} // namespace fsi