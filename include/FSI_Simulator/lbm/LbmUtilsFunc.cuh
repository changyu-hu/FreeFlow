// SPDX-License-Identifier: MIT
// Based on: "High-Order Moment-Encoded Kinetic Simulation of Turbulent Flows" (Li et al., 2023)
// Re-implemented from scratch based on paper equations.

#pragma once

#include "FSI_Simulator/common/CudaCommon.cuh"
#include "FSI_Simulator/lbm/LbmConstants.hpp"

namespace fsi {
namespace lbm {

namespace LbmD3Q27 {
    extern __constant__ float c_w[Q];
    extern __constant__ float ex[Q];
    extern __constant__ float ey[Q];
    extern __constant__ float ez[Q];

}

namespace LbmD2Q9 {
    extern __constant__ float c_w[Q];
    extern __constant__ float ex[Q];
    extern __constant__ float ey[Q];
}

class LbmUtilsFuncGpu3D {
public:
    /**
     * @brief Reconstruct distribution function f_i using Third-Order Hermite expansion.
     * Implements Eq. (17) from the HOME-LBM paper.
     */
    __device__ void CalculateDistributionD3Q27AtIndex(
        float rho, float ux, float uy, float uz,
        float pixx, float piyy, float pizz,
        float pixy, float piyz, float pixz,
        int i, float &f_out);

    __device__ void CalculateDistributionD3Q27All(
        float rho, float ux, float uy, float uz,
        float pixx, float piyy, float pizz,
        float pixy, float piyz, float pixz,
        float *f_out);

    /**
     * @brief Perform Central-Moment-Based Collision to update velocity moments.
     * Implements Eqs. (21), (22), and (23).
     * Note: Inputs pixx..pixz are the moments BEFORE collision (S^*).
     * Output updates references to be the post-collision moments.
     */
    __device__ void Collision(
        float rho, float ux, float uy, float uz, 
        float Fx, float Fy, float Fz, float omega,
        float &pixx, float &piyy, float &pizz, 
        float &pixy, float &piyz, float &pixz);
};

class LbmUtilsFuncGpu2D {
public:
    /**
     * @brief 2D Reconstruction using Third-Order Hermite expansion.
     * Implements Eq. (29) (Appendix B).
     */
    __device__ void CalculateDistributionD2Q9AtIndex(
        float rho, float ux, float uy, 
        float pixx, float pixy, float piyy, 
        int i, float &f_out);

    __device__ void CalculateDistributionD2Q9ALL(
        float rho, float ux, float uy, 
        float pixx, float pixy, float piyy, 
        float *f_out);

    /**
     * @brief 2D Moment-Based Collision.
     * Implements Appendix C equations.
     */
    __device__ void Collision(
        float rho, float ux, float uy, 
        float Fx, float Fy, float omega,
        float &pixx, float &piyy, float &pixy);
};

} // namespace lbm
} // namespace fsi