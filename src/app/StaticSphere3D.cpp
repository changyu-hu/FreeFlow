// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/utils/Logger.hpp"
#include "FSI_Simulator/utils/Profiler.hpp"
#include "FSI_Simulator/utils/Config.hpp"
#include "FSI_Simulator/core/Simulator3D.hpp"
#include <iostream>
#include <string>

using namespace fsi;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_config.json>" << std::endl;
        return -1;
    }
    std::string config_filepath = argv[1];

    fsi::Config3D pre_config;
    if (!pre_config.load(config_filepath))
    {
        std::cerr << "Fatal: Failed to load configuration file." << std::endl;
        return -1;
    }
    const auto &pre_params = pre_config.getParams();

    LOG_INFO("Application starting up.");

    try
    {
        auto simulator = std::make_unique<fsi::Simulator3D>(pre_params);

        simulator->initialize();

        PROFILE_SESSION("Static3D");
        for (int i = 0; i < 5000; ++i)
        {
            simulator->step();
            if (i % 100 == 0)
            {
                simulator->saveFrameData(i / 100);
            }
        }
        PROFILE_END_SESSION();
    }
    catch (const std::exception &e)
    {
        LOG_CRITICAL("An unhandled exception occurred: {}", e.what());
        return -1;
    }
    catch (...)
    {
        LOG_CRITICAL("An unknown exception occurred.");
        return -1;
    }

    LOG_INFO("Application shutting down gracefully.");
    return 0;
}