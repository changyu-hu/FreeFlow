// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once
#include "FSI_Simulator/utils/Config.hpp"
#include "FSI_Simulator/lbm/LbmFlowField3D.hpp"
#include "FSI_Simulator/core/SolidGeometryProxy_Device.cuh"
#include "FSI_Simulator/fem/ISolidSolver.hpp" // 使用具体的VBD求解器
#include <memory>
#include <vector>

namespace fsi {

class Simulator3D {
public:
    explicit Simulator3D(const SimulationParameters3D& params);
    ~Simulator3D() {
        if (m_stream) {
            cudaStreamSynchronize(m_stream);
            cudaStreamDestroy(m_stream);
        }
    }
    void initialize();
    void reset();
    void step();

    void enableFluidSolver(bool enable) { m_fluid_solver_enabled = enable; }

    void applyLBSControl(int mesh_id, const std::vector<vec3_t>& lbs_shift, const std::vector<mat3_t>& lbs_rotation);
    void applyActiveStrain(int mesh_id, int tet_id, vec3_t dir, real magnitude);
    void applyKinematicControl(int mesh_id, const std::vector<vec3_t>& target_pos, real vel);
    void saveFrameData(int frame_index, bool save_fluid = true, bool save_solid = true) const;

    std::vector<float> getFluidMoment() const { 
        return m_flow_field->m_current_moments->download(); 
    }

    std::vector<vec3_t> getVertices() {
        return m_solid_solver->getPositions();
    }
    std::vector<vec3_t> getVelocity() {
        return m_solid_solver->getVelocities();
    }
    std::vector<vec3_t> getForce() {
        return m_solid_solver->getForce();
    }
    std::vector<unsigned int> getTetrahedra() {
        return m_solid_solver->getTetrahedronIdx();
    }
    std::vector<vec3_t> getBoundaryVertices() {
        return m_solid_solver->getBoundaryVertices();
    }
    std::vector<unsigned int> getBoundaryElements() {
        return m_solid_solver->getBoundaryElements();
    }
    std::vector<int> getControlPointIdx() {
        return m_solid_solver->getControlPointIdx();
    }
    std::vector<unsigned int> getBoundaryPointIdx() {
        return m_solid_solver->getBoundaryPointIdx();
    }
    std::vector<vec3_t> getLBSVertices() {
        return m_solid_solver->getLBSVertices();
    }

    const SimulationParameters3D& m_params;
private:
    std::unique_ptr<lbm::LbmFlowField3D> m_flow_field;
    std::unique_ptr<fem::ISolidSolver3D> m_solid_solver;
    bool m_fluid_solver_enabled = true;

    cudaStream_t m_stream;
};

} // namespace fsi