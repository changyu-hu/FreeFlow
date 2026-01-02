// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once
#include "FSI_Simulator/utils/Config.hpp"
#include "FSI_Simulator/lbm/LbmFlowField2D.hpp"
#include "FSI_Simulator/core/SolidGeometryProxy_Device.cuh"
#include "FSI_Simulator/fem/ISolidSolver.hpp" // 使用具体的VBD求解器
#include <memory>
#include <vector>

namespace fsi {

class Simulator2D {
public:
    explicit Simulator2D(const SimulationParameters2D& params);
    ~Simulator2D() {
        if (m_stream) {
            cudaStreamSynchronize(m_stream);
            cudaStreamDestroy(m_stream);
        }
    }

    void initialize();
    void reset();
    void step();
    
    std::vector<std::array<int, 2>> fillSolid();
    
    void enableFluidSolver(bool enable) { m_fluid_solver_enabled = enable; }

    void applyLBSControl(int mesh_id, const std::vector<vec2_t>& lbs_shift, const std::vector<real>& lbs_rotation);
    void applyKinematicControl(int mesh_id, const std::vector<vec2_t>& target_pos, real vel);
    void applyActiveStrain(int mesh_id, int tri_id, vec2_t dir, real magnitude);

    void saveFrameData(int frame_index) const;

    std::vector<float> getFluidMoment() const { 
        return m_flow_field->m_fMom.download(); 
    }
    std::vector<vec2_t> getVertices() {
        return m_solid_solver->getPositions();
    }
    std::vector<vec2_t> getVelocity() {
        return m_solid_solver->getVelocities();
    }
    std::vector<unsigned int> getTriangles() {
        return m_solid_solver->getTriangleIdx();
    }
    std::vector<vec2_t> getForce() {
        return m_solid_solver->getForce();
    }
    std::vector<int> getControlPointIdx() {
        return m_solid_solver->getControlPointIdx();
    }
    std::vector<vec2_t> getBoundaryVertices() {
        return m_solid_solver->getBoundaryVertices();
    }
    std::vector<unsigned int> getBoundaryElements() {
        return m_solid_solver->getBoundaryElements();
    }
    std::vector<unsigned int> getBoundaryPointIdx() {
        return m_solid_solver->getBoundaryPointIdx();
    }
    std::vector<vec2_t> getLBSVertices() {
        return m_solid_solver->getLBSVertices();
    }

    const SimulationParameters2D& m_params;

private:

    std::unique_ptr<lbm::LbmFlowField2D> m_flow_field;
    std::unique_ptr<fem::ISolidSolver2D> m_solid_solver;
    bool m_fluid_solver_enabled = true;

    cudaStream_t m_stream;
};

} // namespace fsi