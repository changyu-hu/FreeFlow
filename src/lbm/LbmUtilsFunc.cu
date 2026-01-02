#include "FSI_Simulator/lbm/LbmUtilsFunc.cuh"

namespace fsi {
namespace lbm {


// ============================================================================
// 3D IMPLEMENTATION (HOME-LBM)
// ============================================================================

__device__ void LbmUtilsFuncGpu3D::CalculateDistributionD3Q27AtIndex(
    float rho, float ux, float uy, float uz,
    float pixx, float piyy, float pizz,
    float pixy, float piyz, float pixz,
    int i, float &f_out)
{
    // ==========================================
    // 1. Pre-computation of Hermite Coefficients
    // ==========================================
    
    // Scale Factors derived from Lattice weights and Hermite normalization
    // 1st Order factor: 3.0
    // 2nd Order factor: 9.0 (used partially) -> 1.5 for diagonals
    // 3rd Order factor: 27.0
    
    // --- 1st Order Moments (Momentum) ---
    float m1_x = 3.0f * rho * ux;
    float m1_y = 3.0f * rho * uy;
    float m1_z = 3.0f * rho * uz;

    // --- 2nd Order Moments (Stress) ---
    // Scaled by 3.0 for base usage
    float m2_xx = 3.0f * rho * pixx;
    float m2_yy = 3.0f * rho * piyy;
    float m2_zz = 3.0f * rho * pizz;
    
    // Off-diagonals scaled by 9.0 (Standard Hermite weight adjustment)
    float m2_xy = 9.0f * rho * pixy;
    float m2_yz = 9.0f * rho * piyz;
    float m2_xz = 9.0f * rho * pixz;

    // --- 3rd Order Coefficients (The "Q" tensor) ---
    // Formula: Q_aab = Pi_aa * u_b + 2 * Pi_ab * u_a - 2 * rho * u_a^2 * u_b
    // Scaled by 9.0 for mixed terms (xxy, etc) and 27.0 for xyz
    
    auto calc_Q_mixed = [&](float Pi_aa, float u_b, float Pi_ab, float u_a) {
        return 9.0f * (rho * Pi_aa * u_b + 2.0f * rho * Pi_ab * u_a - 2.0f * rho * u_a * u_a * u_b);
    };

    float Q_xxy = calc_Q_mixed(pixx, uy, pixy, ux);
    float Q_xyy = calc_Q_mixed(piyy, ux, pixy, uy);
    
    float Q_xxz = calc_Q_mixed(pixx, uz, pixz, ux);
    float Q_xzz = calc_Q_mixed(pizz, ux, pixz, uz);
    
    float Q_yyz = calc_Q_mixed(piyy, uz, piyz, uy);
    float Q_yzz = calc_Q_mixed(pizz, uy, piyz, uz);

    // Fully mixed term Q_xyz
    // Eq: Pi_xy*uz + Pi_yz*ux + Pi_xz*uy - 2*rho*ux*uy*uz
    float Q_xyz = 27.0f * (rho * (pixy * uz + piyz * ux + pixz * uy - 2.0f * ux * uy * uz));

    // --- Isotropic / Bulk Term ---
    // 0th Order - 0.5 * Trace(2nd Order)
    // Coeff for trace subtraction in 2nd order expansion is -1/2 (relative to scaled variables)
    float bulk_part = rho - 0.5f * (m2_xx + m2_yy + m2_zz);

    float bulk_part2 = rho + (m2_xx + m2_yy + m2_zz);

    // ==========================================
    // 2. Assembly based on Lattice Direction
    // ==========================================
    
    // Fetch weight (Assuming standard D3Q27 weights accessible)
    float w = LbmD3Q27::c_w[i];
    float sum = 0.0f;

    // Note: The switch cases follow standard D3Q27 ordering assumptions.
    // 0: Center
    // 1-6: Axis (1,0,0)...
    // 7-18: Edge (1,1,0)...
    // 19-26: Corner (1,1,1)...
    
    switch (i)
    {
    case 0: // (0,0,0)
        sum = bulk_part;
        break;

    // --- Axis Aligned (+/- 1, 0, 0) ---
    // Formula: Bulk +/- M1 + 1.5*M2_diag - 0.5*(Q_transverse)
    case 1: // (1, 0, 0)
        sum = bulk_part + m1_x + 1.5f * m2_xx - 0.5f * (Q_xyy + Q_xzz); 
        break;
    case 2: // (-1, 0, 0)
        sum = bulk_part - m1_x + 1.5f * m2_xx + 0.5f * (Q_xyy + Q_xzz);
        break;
    case 3: // (0, 1, 0)
        sum = bulk_part + m1_y + 1.5f * m2_yy - 0.5f * (Q_xxy + Q_yzz);
        break;
    case 4: // (0, -1, 0)
        sum = bulk_part - m1_y + 1.5f * m2_yy + 0.5f * (Q_xxy + Q_yzz);
        break;
    case 5: // (0, 0, 1)
        sum = bulk_part + m1_z + 1.5f * m2_zz - 0.5f * (Q_xxz + Q_yyz);
        break;
    case 6: // (0, 0, -1)
        sum = bulk_part - m1_z + 1.5f * m2_zz + 0.5f * (Q_xxz + Q_yyz);
        break;

    // --- Face Diagonals (+/- 1, +/- 1, 0) ---
    // Formula: Bulk +/- M1s + M2s - 0.5*M2_missing + M2_cross +/- Qs ...
    // Note: The term 1.5*M2_ii splits into contributions.
    // Standard Hermite expansion for (1,1,0):
    // const + (u_x+u_y) + (Sxx + Syy - 0.5 Szz) + 2 Sxy + ...
    case 7: // (1, 1, 0)
        sum = rho + (m1_x + m1_y) + (m2_xx + m2_yy - 0.5f*m2_zz) + m2_xy + (Q_xxy + Q_xyy) - 0.5f*(Q_xzz + Q_yzz);
        break;
    case 8: // (-1, -1, 0)
        sum = rho - (m1_x + m1_y) + (m2_xx + m2_yy - 0.5f*m2_zz) + m2_xy - (Q_xxy + Q_xyy) + 0.5f*(Q_xzz + Q_yzz);
        break;
    case 9: // (1, 0, 1)
        sum = rho + (m1_x + m1_z) + (m2_xx + m2_zz - 0.5f*m2_yy) + m2_xz + (Q_xxz + Q_xzz) - 0.5f*(Q_xyy + Q_yyz);
        break;
    case 10: // (-1, 0, -1)
        sum = rho - (m1_x + m1_z) + (m2_xx + m2_zz - 0.5f*m2_yy) + m2_xz - (Q_xxz + Q_xzz) + 0.5f*(Q_xyy + Q_yyz);
        break;
    case 11: // (0, 1, 1)
        sum = rho + (m1_y + m1_z) + (m2_yy + m2_zz - 0.5f*m2_xx) + m2_yz + (Q_yyz + Q_yzz) - 0.5f*(Q_xxy + Q_xxz);
        break;
    case 12: // (0, -1, -1)
        sum = rho - (m1_y + m1_z) + (m2_yy + m2_zz - 0.5f*m2_xx) + m2_yz - (Q_yyz + Q_yzz) + 0.5f*(Q_xxy + Q_xxz);
        break;
    
    // Mixed Face Diagonals (Opposite signs)
    case 13: // (1, -1, 0)
        sum = rho + (m1_x - m1_y) + (m2_xx + m2_yy - 0.5f*m2_zz) - m2_xy - (Q_xxy - Q_xyy) - 0.5f*(Q_xzz - Q_yzz);
        break;
    case 14: // (-1, 1, 0)
        sum = rho - (m1_x - m1_y) + (m2_xx + m2_yy - 0.5f*m2_zz) - m2_xy + (Q_xxy - Q_xyy) + 0.5f*(Q_xzz - Q_yzz);
        break;
    case 15: // (1, 0, -1)
        sum = rho + (m1_x - m1_z) + (m2_xx + m2_zz - 0.5f*m2_yy) - m2_xz - (Q_xxz - Q_xzz) - 0.5f*(Q_xyy - Q_yyz);
        break;
    case 16: // (-1, 0, 1)
        sum = rho - (m1_x - m1_z) + (m2_xx + m2_zz - 0.5f*m2_yy) - m2_xz + (Q_xxz - Q_xzz) + 0.5f*(Q_xyy - Q_yyz);
        break;
    case 17: // (0, 1, -1)
        sum = rho + (m1_y - m1_z) + (m2_yy + m2_zz - 0.5f*m2_xx) - m2_yz - (Q_yyz - Q_yzz) - 0.5f*(Q_xxy - Q_xxz);
        break;
    case 18: // (0, -1, 1)
        sum = rho - (m1_y - m1_z) + (m2_yy + m2_zz - 0.5f*m2_xx) - m2_yz + (Q_yyz - Q_yzz) + 0.5f*(Q_xxy - Q_xxz);
        break;

    // --- Corners (1, 1, 1) etc ---
    // All additive. 
    // M1 terms: sum
    // M2 terms: sum diagonals + sum off-diagonals
    // Q terms: sum mixed + sum fully mixed (xyz)
    case 19: // (1, 1, 1)
        sum = bulk_part2 + (m1_x + m1_y + m1_z) 
              + (m2_xy + m2_yz + m2_xz)
              + (Q_xxy + Q_xxz + Q_xyy + Q_yyz + Q_xzz + Q_yzz) + Q_xyz;
        break;
    case 20: // (-1, -1, -1)
        sum = bulk_part2 - (m1_x + m1_y + m1_z) 
              + (m2_xy + m2_yz + m2_xz)
              - (Q_xxy + Q_xxz + Q_xyy + Q_yyz + Q_xzz + Q_yzz) - Q_xyz;
        break;
        
    // Permutations for other corners (Signs follow the coordinate product)
    // Example: (1, 1, -1) -> z negative. 
    // M1: x+y-z
    // M2 off-diag: xy (pos), yz (neg), xz (neg)
    // Q mixed: Q_xxy (+), Q_xxz (-), Q_xyy (+), Q_yyz (-), Q_xzz (+), Q_yzz (+) 
    // Actually, simply propagate signs of c_i.
    case 21: // (1, 1, -1)
        sum = bulk_part2 + (m1_x + m1_y - m1_z)
              + (m2_xy - m2_yz - m2_xz)
              + (Q_xxy - Q_xxz + Q_xyy - Q_yyz + Q_xzz + Q_yzz) - Q_xyz;
        break;
    case 22: // (-1, -1, 1)
        sum = bulk_part2 - (m1_x + m1_y - m1_z)
              + (m2_xy - m2_yz - m2_xz)
              - (Q_xxy - Q_xxz + Q_xyy - Q_yyz + Q_xzz + Q_yzz) + Q_xyz;
        break;
    case 23: // (1, -1, 1)
        sum = bulk_part2 + (m1_x - m1_y + m1_z)
              + (-m2_xy - m2_yz + m2_xz)
              + (-Q_xxy + Q_xxz + Q_xyy + Q_yyz + Q_xzz - Q_yzz) - Q_xyz;
        break;
    case 24: // (-1, 1, -1)
        sum = bulk_part2 - (m1_x - m1_y + m1_z)
              + (-m2_xy - m2_yz + m2_xz)
              - (-Q_xxy + Q_xxz + Q_xyy + Q_yyz + Q_xzz - Q_yzz) + Q_xyz;
        break;
    case 25: // (-1, 1, 1)
        sum = bulk_part2 + (-m1_x + m1_y + m1_z)
              + (-m2_xy + m2_yz - m2_xz)
              + (Q_xxy + Q_xxz - Q_xyy + Q_yyz - Q_xzz + Q_yzz) - Q_xyz;
        break;
    case 26: // (1, -1, -1)
        sum = bulk_part2 - (-m1_x + m1_y + m1_z)
              + (-m2_xy + m2_yz - m2_xz)
              - (Q_xxy + Q_xxz - Q_xyy + Q_yyz - Q_xzz + Q_yzz) + Q_xyz;
        break;
    }

    f_out = w * sum;
}

__device__ void LbmUtilsFuncGpu3D::CalculateDistributionD3Q27All(
    float rho, float ux, float uy, float uz,
    float pixx, float piyy, float pizz,
    float pixy, float piyz, float pixz,
    float *f_out)
{
    // 1. Calculate Common Terms Once
    float m1_x = 3.0f * rho * ux;
    float m1_y = 3.0f * rho * uy;
    float m1_z = 3.0f * rho * uz;

    float m2_xx = 3.0f * rho * pixx;
    float m2_yy = 3.0f * rho * piyy;
    float m2_zz = 3.0f * rho * pizz;
    float m2_xy = 9.0f * rho * pixy;
    float m2_yz = 9.0f * rho * piyz;
    float m2_xz = 9.0f * rho * pixz;

    auto calc_Q = [&](float Pi_aa, float u_b, float Pi_ab, float u_a) {
        return 9.0f * (rho * Pi_aa * u_b + 2.0f * rho * Pi_ab * u_a - 2.0f * rho * u_a * u_a * u_b);
    };

    float Q_xxy = calc_Q(pixx, uy, pixy, ux);
    float Q_xyy = calc_Q(piyy, ux, pixy, uy);
    float Q_xxz = calc_Q(pixx, uz, pixz, ux);
    float Q_xzz = calc_Q(pizz, ux, pixz, uz);
    float Q_yyz = calc_Q(piyy, uz, piyz, uy);
    float Q_yzz = calc_Q(pizz, uy, piyz, uz);
    float Q_xyz = 27.0f * (rho * (pixy * uz + piyz * ux + pixz * uy - 2.0f * ux * uy * uz));

    float bulk = rho - 0.5f * (m2_xx + m2_yy + m2_zz);

    // Weights (Load to registers)
    float w0 = LbmD3Q27::c_w[0];
    float w1 = LbmD3Q27::c_w[1];
    float w2 = LbmD3Q27::c_w[7];
    float w3 = LbmD3Q27::c_w[19];

    // 2. Explicit Assignments (Loop Unrolled)
    
    // Center
    f_out[0] = w0 * bulk;

    // Axis (Use symmetry to compute +/- pairs)
    // Common part for X-axis: bulk + 1.5 S_xx - 0.5(Q_transverse)
    float axis_base_x = bulk + 1.5f * m2_xx;
    f_out[1] = w1 * (axis_base_x + m1_x - 0.5f * (Q_xyy + Q_xzz));
    f_out[2] = w1 * (axis_base_x - m1_x + 0.5f * (Q_xyy + Q_xzz));

    float axis_base_y = bulk + 1.5f * m2_yy;
    f_out[3] = w1 * (axis_base_y + m1_y - 0.5f * (Q_xxy + Q_yzz));
    f_out[4] = w1 * (axis_base_y - m1_y + 0.5f * (Q_xxy + Q_yzz));

    float axis_base_z = bulk + 1.5f * m2_zz;
    f_out[5] = w1 * (axis_base_z + m1_z - 0.5f * (Q_xxz + Q_yyz));
    f_out[6] = w1 * (axis_base_z - m1_z + 0.5f * (Q_xxz + Q_yyz));

    // Edge Diagonals
    // Base: bulk + diags - 0.5*ortho
    float edge_base_xy = rho + m2_xx + m2_yy - 0.5f * m2_zz;
    float edge_Q_xy    = Q_xxy + Q_xyy;
    float edge_Q_orth_xy = 0.5f * (Q_xzz + Q_yzz);
    
    // (1,1,0) & (-1,-1,0)
    f_out[7] = w2 * (edge_base_xy + (m1_x + m1_y) + m2_xy + edge_Q_xy - edge_Q_orth_xy);
    f_out[8] = w2 * (edge_base_xy - (m1_x + m1_y) + m2_xy - edge_Q_xy + edge_Q_orth_xy);
    // (1,-1,0) & (-1,1,0)
    f_out[13] = w2 * (edge_base_xy + (m1_x - m1_y) - m2_xy - (Q_xxy - Q_xyy) - 0.5f*(Q_xzz - Q_yzz));
    f_out[14] = w2 * (edge_base_xy - (m1_x - m1_y) - m2_xy + (Q_xxy - Q_xyy) + 0.5f*(Q_xzz - Q_yzz));

    float edge_base_xz = rho + m2_xx + m2_zz - 0.5f * m2_yy;
    float edge_Q_xz    = Q_xxz + Q_xzz;
    float edge_Q_orth_xz = 0.5f * (Q_xyy + Q_yyz);

    f_out[9]  = w2 * (edge_base_xz + (m1_x + m1_z) + m2_xz + edge_Q_xz - edge_Q_orth_xz);
    f_out[10] = w2 * (edge_base_xz - (m1_x + m1_z) + m2_xz - edge_Q_xz + edge_Q_orth_xz);
    f_out[15] = w2 * (edge_base_xz + (m1_x - m1_z) - m2_xz - (Q_xxz - Q_xzz) - 0.5f*(Q_xyy - Q_yyz));
    f_out[16] = w2 * (edge_base_xz - (m1_x - m1_z) - m2_xz + (Q_xxz - Q_xzz) + 0.5f*(Q_xyy - Q_yyz));

    float edge_base_yz = rho + m2_yy + m2_zz - 0.5f * m2_xx;
    float edge_Q_yz    = Q_yyz + Q_yzz;
    float edge_Q_orth_yz = 0.5f * (Q_xxy + Q_xxz);

    f_out[11] = w2 * (edge_base_yz + (m1_y + m1_z) + m2_yz + edge_Q_yz - edge_Q_orth_yz);
    f_out[12] = w2 * (edge_base_yz - (m1_y + m1_z) + m2_yz - edge_Q_yz + edge_Q_orth_yz);
    f_out[17] = w2 * (edge_base_yz + (m1_y - m1_z) - m2_yz - (Q_yyz - Q_yzz) - 0.5f*(Q_xxy - Q_xxz));
    f_out[18] = w2 * (edge_base_yz - (m1_y - m1_z) - m2_yz + (Q_yyz - Q_yzz) + 0.5f*(Q_xxy - Q_xxz));

    // Corners
    // P = sum(M1), S = sum(M2_diag), O = sum(M2_off), Q_sum = sum(Q_mixed)
    float c_M1 = m1_x + m1_y + m1_z;
    float c_S  = m2_xx + m2_yy + m2_zz;
    float c_O  = m2_xy + m2_yz + m2_xz;
    float c_Q  = Q_xxy + Q_xxz + Q_xyy + Q_yyz + Q_xzz + Q_yzz;
    
    // Base for (1,1,1) is bulk + S + O + Q + Qxyz
    // However, signs flip. It's cleaner to implement explicit sums for correctness.
    // Optimization: Group terms by coordinate signs
    
    // Corner Base Term (Diag parts are always positive in 3rd order isotropic lattice expansion? No)
    // Actually, Diag M2 (m2_xx etc) are always added as +1.5 or +1.
    // In corners (+/-1, +/-1, +/-1), c_i*c_i is always 1. So Diag M2 is always added positive.
    float corner_common = rho + c_S; 

    // (1,1,1)
    f_out[19] = w3 * (corner_common + c_M1 + c_O + c_Q + Q_xyz);
    // (-1,-1,-1)
    f_out[20] = w3 * (corner_common - c_M1 + c_O - c_Q - Q_xyz);

    // (1,1,-1) -> z neg
    f_out[21] = w3 * (corner_common + (m1_x+m1_y-m1_z) + (m2_xy-m2_yz-m2_xz) + (Q_xxy-Q_xxz + Q_xyy-Q_yyz + Q_xzz+Q_yzz) - Q_xyz);
    // (-1,-1,1) -> z pos, x,y neg
    f_out[22] = w3 * (corner_common - (m1_x+m1_y-m1_z) + (m2_xy-m2_yz-m2_xz) - (Q_xxy-Q_xxz + Q_xyy-Q_yyz + Q_xzz+Q_yzz) + Q_xyz);

    // (1,-1,1) -> y neg
    f_out[23] = w3 * (corner_common + (m1_x-m1_y+m1_z) + (-m2_xy-m2_yz+m2_xz) + (-Q_xxy+Q_xxz + Q_xyy+Q_yyz + Q_xzz-Q_yzz) - Q_xyz);
    // (-1,1,-1) -> opposite
    f_out[24] = w3 * (corner_common - (m1_x-m1_y+m1_z) + (-m2_xy-m2_yz+m2_xz) - (-Q_xxy+Q_xxz + Q_xyy+Q_yyz + Q_xzz-Q_yzz) + Q_xyz);
    
    // (-1,1,1) -> x neg
    f_out[25] = w3 * (corner_common + (-m1_x+m1_y+m1_z) + (-m2_xy+m2_yz-m2_xz) + (Q_xxy+Q_xxz - Q_xyy+Q_yyz - Q_xzz+Q_yzz) - Q_xyz);
    // (1,-1,-1) -> opposite
    f_out[26] = w3 * (corner_common - (-m1_x+m1_y+m1_z) + (-m2_xy+m2_yz-m2_xz) - (Q_xxy+Q_xxz - Q_xyy+Q_yyz - Q_xzz+Q_yzz) + Q_xyz);
}

__device__ void LbmUtilsFuncGpu3D::Collision(
    float rho, float ux, float uy, float uz, 
    float Fx, float Fy, float Fz, float omega,
    float &pixx, float &piyy, float &pizz, 
    float &pixy, float &piyz, float &pixz)
{
    // High-performance MRT collision update for moments (Eq 23)
    // Optimized to use multiply-add and pre-calculated force terms.
    
    // 1. Constants
    float one_minus_w = 1.0f - omega;
    float dev_force_coeff = one_minus_w * 0.5f; 
    float off_diag_force_coeff = 1.0f - 0.5f * omega; 
    
    // 2. Precompute Equilibrium and Force Mixers
    float r_3 = rho * 0.33333333f; // rho/3 (Pressure part)
    float ruu_x = rho * ux * ux;
    float ruu_y = rho * uy * uy;
    float ruu_z = rho * uz * uz;
    
    // 3. Update Off-Diagonals (Simpler, no cross-coupling)
    // S_xy_new = (1-w) S_xy + w (Eq_xy) + Force_xy
    float w_rho = omega * rho;
    pixy = one_minus_w * pixy + w_rho * (ux * uy) + off_diag_force_coeff * (Fx * uy + Fy * ux);
    pixz = one_minus_w * pixz + w_rho * (ux * uz) + off_diag_force_coeff * (Fx * uz + Fz * ux);
    piyz = one_minus_w * piyz + w_rho * (uy * uz) + off_diag_force_coeff * (Fy * uz + Fz * uy);

    // 4. Update Diagonals (MRT coupling)
    // The paper uses a specific trace-conserving relaxation.
    // Decomposition: 
    //   Iso_Part = Pressure + 1/3 rho u^2
    //   Dev_Part = S_aa - 1/3 Trace(S)
    //   Update = Iso_Eq + (1-w) Dev_Part + Force
    
    // Helper: Compute Deviatoric Stress * (1-w) efficiently
    // D_xx = (2 Sxx - Syy - Szz) / 3
    // We compute this directly to avoid finding Trace first (which saves precision)
    float s_xx = pixx; float s_yy = piyy; float s_zz = pizz;
    
    float dev_xx = (2.0f * s_xx - s_yy - s_zz) * 0.33333333f;
    float dev_yy = (2.0f * s_yy - s_xx - s_zz) * 0.33333333f;
    float dev_zz = (2.0f * s_zz - s_xx - s_yy) * 0.33333333f;
    
    // Equilibrium Trace: P + 1/3 rho u^2
    // Note: The specific relaxation includes terms related to u^2 * omega.
    // From original code logic (proven correct for this paper): 
    // Term = R/3 + (1/3)(rho u^2) + (omega/3)(2 ruu_a - ruu_b - ruu_c)
    // This looks like Iso_Eq + Dev_Eq * omega? 
    // Actually it simplifies to: P + (1-w) Dev_Star + (Bulk_Eq)
    
    // Let's implement the grouped Force term first
    // F_term_a = F_a U_a + (1-w)/3 * (2 Fa Ua - Fb Ub - Fc Uc)
    float fu_x = Fx * ux;
    float fu_y = Fy * uy;
    float fu_z = Fz * uz;
    float f_dev_common = (dev_force_coeff * 0.6666667f); // (1-w)/2 * 2/3 = (1-w)/3
    
    // Compute Energy term (RUVW2)
    float energy = (ruu_x + ruu_y + ruu_z) * 0.33333333f;
    
    // Update X
    float relax_u2_x = (omega * 0.33333333f) * (2.0f * ruu_x - ruu_y - ruu_z);
    float force_x    = fu_x + f_dev_common * (2.0f * fu_x - fu_y - fu_z);
    
    pixx = r_3 + energy + one_minus_w * dev_xx + relax_u2_x + force_x;

    // Update Y
    float relax_u2_y = (omega * 0.33333333f) * (2.0f * ruu_y - ruu_x - ruu_z);
    float force_y    = fu_y + f_dev_common * (2.0f * fu_y - fu_x - fu_z);
    
    piyy = r_3 + energy + one_minus_w * dev_yy + relax_u2_y + force_y;

    // Update Z
    float relax_u2_z = (omega * 0.33333333f) * (2.0f * ruu_z - ruu_x - ruu_y);
    float force_z    = fu_z + f_dev_common * (2.0f * fu_z - fu_x - fu_y);
    
    pizz = r_3 + energy + one_minus_w * dev_zz + relax_u2_z + force_z;
}



// ============================================================================
// 2D IMPLEMENTATION (Appendix B & C)
// ============================================================================

__device__ void LbmUtilsFuncGpu2D::CalculateDistributionD2Q9AtIndex(
    float rho, float ux, float uy, 
    float pixx, float pixy, float piyy, 
    int i, float &f_out)
{
    // Optimized 2D Reconstruction (Eq 29)
    // Coeffs
    float m1_x = 3.0f * rho * ux;
    float m1_y = 3.0f * rho * uy;

    float m2_xx = 3.0f * rho * pixx; // Using factor 3 for base, 4.5 total
    float m2_yy = 3.0f * rho * piyy;
    float m2_xy = 9.0f * rho * pixy; // 4.5 * 2

    // 3rd Order Q terms (Scaled by 27/2 = 13.5)
    // Q_xxy = 13.5 * (Pi_xx uy + 2 Pi_xy ux - 2 rho ux^2 uy)
    // But let's use the shared logic factor (similar to 3D code style)
    auto calc_Q = [&](float Pi_aa, float u_b, float Pi_ab, float u_a) {
        return 9.0f * (Pi_aa * u_b + 2.0f * Pi_ab * u_a - 2.0f * rho * u_a * u_a * u_b);
    };

    float Q_xxy = calc_Q(pixx, uy, pixy, ux);
    float Q_xyy = calc_Q(piyy, ux, pixy, uy);

    // Bulk Term: rho - 1.5 * (Pi_xx + Pi_yy)
    // Since m2_xx = 3*rho*Pi, 1.5 Pi = 0.5 * m2/rho * rho = 0.5 m2?
    // Wait, m2_xx defined as 3*rho*Pi_xx. 
    // Term2 in Eq 29: (cx^2 - 1/3)*Pi_xx / (2/9) = 4.5 * (cx^2 - 1/3) Pi_xx
    // For Center (cx=0): -1.5 Pi_xx.
    // So Bulk = rho - 1.5(Pi_xx + Pi_yy).
    float bulk = rho - 0.5f * (m2_xx + m2_yy);
    float diag_base = bulk + 1.5f * (m2_xx + m2_yy);

    float w = LbmD2Q9::c_w[i];
    float sum = 0.0f;

    switch(i) {
        case 0: sum = bulk; break;
        // Axis
        case 1: sum = bulk + m1_x + 1.5f * m2_xx - 0.5f * Q_xyy; break;
        case 2: sum = bulk + m1_y + 1.5f * m2_yy - 0.5f * Q_xxy; break;
        case 3: sum = bulk - m1_x + 1.5f * m2_xx + 0.5f * Q_xyy; break;
        case 4: sum = bulk - m1_y + 1.5f * m2_yy + 0.5f * Q_xxy; break;
        // Diagonals (1,1)
        // M1: x+y, M2: xx+yy+2xy, Q: xxy+xyy

        case 5: sum = diag_base + (m1_x + m1_y) + m2_xy + (Q_xxy + Q_xyy); break; // (1, 1)
        case 6: sum = diag_base - m1_x + m1_y - m2_xy + Q_xxy - Q_xyy; break; // (-1, 1)
        case 7: sum = diag_base - m1_x - m1_y + m2_xy - Q_xxy - Q_xyy; break; // (-1, -1)
        case 8: sum = diag_base + m1_x - m1_y - m2_xy - Q_xxy + Q_xyy; break; // (1, -1)
    }
    f_out = w * sum;
}

__device__ void LbmUtilsFuncGpu2D::CalculateDistributionD2Q9ALL(
    float rho, float ux, float uy, 
    float pixx, float pixy, float piyy, 
    float *f_out)
{
    // Unrolled version
    float m1_x = 3.0f * rho * ux;
    float m1_y = 3.0f * rho * uy;
    float m2_xx = 3.0f * rho * pixx;
    float m2_yy = 3.0f * rho * piyy;
    float m2_xy = 9.0f * rho * pixy;

    auto calc_Q = [&](float Pi_aa, float u_b, float Pi_ab, float u_a) {
        return 9.0f * (Pi_aa * u_b + 2.0f * Pi_ab * u_a - 2.0f * rho * u_a * u_a * u_b);
    };
    float Q_xxy = calc_Q(pixx, uy, pixy, ux);
    float Q_xyy = calc_Q(piyy, ux, pixy, uy);
    float bulk = rho - 0.5f * (m2_xx + m2_yy);

    float *w = LbmD2Q9::c_w; // assuming access

    f_out[0] = w[0] * bulk;
    f_out[1] = w[1] * (bulk + m1_x + 1.5f * m2_xx - 0.5 * Q_xyy);
    f_out[2] = w[2] * (bulk + m1_y + 1.5f * m2_yy - 0.5 * Q_xxy);
    f_out[3] = w[3] * (bulk - m1_x + 1.5f * m2_xx + 0.5 * Q_xyy);
    f_out[4] = w[4] * (bulk - m1_y + 1.5f * m2_yy + 0.5 * Q_xxy);
    
    // float diag_common = bulk + m2_xx + m2_yy; 
    float diag_base = bulk + 1.5f * (m2_xx + m2_yy);
    
    f_out[5] = w[5] * (diag_base + (m1_x + m1_y) + m2_xy + (Q_xxy + Q_xyy));
    f_out[6] = w[6] * (diag_base - m1_x + m1_y - m2_xy + Q_xxy - Q_xyy);
    f_out[7] = w[7] * (diag_base - m1_x - m1_y + m2_xy - Q_xxy - Q_xyy);
    f_out[8] = w[8] * (diag_base + m1_x - m1_y - m2_xy - Q_xxy + Q_xyy);
}

__device__ void LbmUtilsFuncGpu2D::Collision(
    float rho, float ux, float uy, 
    float Fx, float Fy, float omega,
    float &pixx, float &piyy, float &pixy)
{
    float one_minus_w = 1.0f - omega;
    float p = rho / 3.0f;
    float ru2 = rho * ux * ux;
    float rv2 = rho * uy * uy;
    
    // Deviatoric part: (Pi_xx - Pi_yy)/2
    float dev = (pixx - piyy) * 0.5f;
    
    // Total Energy/Bulk part
    float bulk_energy = p + 0.5f * (ru2 + rv2);
    // Anisotropic Equilibrium part (w/2 * rho * (u^2 - v^2))
    float aniso_eq = 0.5f * omega * (ru2 - rv2);
    
    // Forces
    float force_factor = 0.5f * one_minus_w;
    float f_term_x = Fx * ux + force_factor * (Fx * ux - Fy * uy);
    float f_term_y = Fy * uy + force_factor * (Fy * uy - Fx * ux);
    
    float pi_xx_new = bulk_energy + dev * one_minus_w + aniso_eq + f_term_x;
    float pi_yy_new = bulk_energy - dev * one_minus_w - aniso_eq + f_term_y;
    
    float pi_xy_new = pixy * one_minus_w + (omega * rho * ux * uy) + (1.0f - 0.5f*omega)*(Fy*ux + Fx*uy);
    
    pixx = pi_xx_new;
    piyy = pi_yy_new;
    pixy = pi_xy_new;
}

} // namespace lbm
} // namespace fsi