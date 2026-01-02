// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/fem/NewtonKernels.cuh"
#include "FSI_Simulator/utils/NumericUtils.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

namespace fsi
{
    namespace fem
    {

        __device__ static const Matrix2r DeterminantGradient(const Matrix2r &A)
        {
            // J = |A|.
            // dJdA = J * A^{-T}.
            // A = [a, b]
            //     [c, d]
            // A^{-1} = [d, -b] / J.
            //          [-c, a]
            // A^{-T} = [d, -c] / J.
            //          [-b, a]
            // dJdA = [d, -c]
            //        [-b, a]
            Matrix2r dJdA;
            dJdA << A(1, 1), -A(1, 0),
                -A(0, 1), A(0, 0);
            return dJdA;
        }

        __device__ static const Matrix4r DeterminantHessian(const Matrix2r &A)
        {
            Matrix4r H = Matrix4r::Zero();
            H(3, 0) = 1;
            H(2, 1) = -1;
            H(1, 2) = -1;
            H(0, 3) = 1;
            return H;
        }

        __device__ real NeohookeanComputeEnergyDensityFromDeformationGradientGpu2d(const real la, const real mu, const real *deformation_grad)
        {
            const integer dim = 2;
            const Eigen::Map<const Matrix2r> F(deformation_grad);
            const Matrix2r C = F.transpose() * F;
            const real J = F.determinant();
            const real Ic = C.trace();
            const real t = mu / la;
            const real a = 2 * t + 1;               // > 0, 2t + 1.
            const real b = 2 * t * (dim - 1) + dim; // > 0, 2t + 2.
            const real c = -t * dim;                // < 0, -2t.
            const real delta = (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
            const real alpha = (1 - 1 / (dim + delta)) * mu / la + 1;
            return mu / 2 * (Ic - dim) + la / 2 * (J - alpha) * (J - alpha) - 0.5 * mu * std::log(Ic + delta);
        }

        __device__ void NeohookeanComputeStressTensorFromDeformationGradientGpu2d(const real la, const real mu, const real *deformation_grad,
                                                                                  real *stress_tensor)
        {
            const integer dim = 2;
            const Eigen::Map<const Matrix2r> F(deformation_grad);
            const Matrix2r C = F.transpose() * F;
            const real J = F.determinant();
            const real Ic = C.trace();
            const real t = mu / la;
            const real a = 2 * t + 1;               // > 0, 2t + 1.
            const real b = 2 * t * (dim - 1) + dim; // > 0, 2t + 2.
            const real c = -t * dim;                // < 0, -2t.
            const real delta = (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
            const real alpha = (1 - 1 / (dim + delta)) * mu / la + 1;
            const Matrix2r dJdF = DeterminantGradient(Matrix2r(F));
            Eigen::Map<Matrix2r> P(stress_tensor);
            P = (1 - 1 / (Ic + delta)) * mu * F + la * (J - alpha) * dJdF;
        }

        __device__ void NeohookeanComputeStressTensorDifferentialFromDeformationGradientGpu2d(
            const real la, const real mu, const real *deformation_grad, real *stress_tensor_differential)
        {

            const integer dim = 2;
            Eigen::Map<const Matrix2r> F(deformation_grad);
            const Matrix2r C = F.transpose() * F;
            const real J = F.determinant();
            const real Ic = C.trace();
            const real t = mu / la;
            const real a = 2 * t + 1;               // > 0, 2t + 1.
            const real b = 2 * t * (dim - 1) + dim; // > 0, 2t + 2.
            const real c = -t * dim;                // < 0, -2t.
            const real delta = (-b + std::sqrt(b * b - 4 * a * c)) / (2 * a);
            const real alpha = (1 - 1 / (dim + delta)) * mu / la + 1;
            // dJ/dF = JF^-T
            const Matrix2r dJdF = DeterminantGradient(Matrix2r(F));
            const Vector4r f = F.reshaped();
            // P = (1 - 1 / (Ic + delta)) * mu * F + la * (J - alpha) * dJdF.
            // Part I: (1 - 1 / (Ic + delta)) * mu * F.
            Eigen::Map<Matrix4r> dPdF(stress_tensor_differential);
            dPdF = ((1 - 1 / (Ic + delta)) * mu) * Matrix4r::Identity() + (2 * mu / ((Ic + delta) * (Ic + delta))) * (f * f.transpose());
            // Part II: la * (J - alpha) * dJdF.
            const Vector4r djdf = dJdF.reshaped();
            dPdF += la * (djdf * djdf.transpose());
            dPdF += la * (J - alpha) * DeterminantHessian(Matrix2r(F));
            // When F = 0, we want dPdF to reach semi-definite (one eigenvalue happens to be zero).
            // eig = mu * (1 - 1 / delta) +/- la * alpha.
            // mu / la * (1 - 1 / delta) +/- alpha.
            // We want the matrix to be negative semi-definite, so we have
            //
            // mu / la * (1 - 1 / delta) + alpha = 0.
            // [1 - 1 / (dim + delta)] * mu / la + 1 - alpha = 0
            //
            // Call t = mu / la.
            // t * (1 - 1 / delta) + 1 + (1 - 1 / (dim + delta)) * t = 0.
            // t * (delta - 1) / delta + 1 + t * (dim - 1 + delta) / (dim + delta) = 0.
            // t * (delta - 1) * (delta + dim) + delta * (dim + delta) + t * (dim - 1 + delta) * delta = 0.
            // t * (1, dim - 1, -dim) + (1, dim, 0) + (t, t * (dim - 1), 0) = 0.
        }

        __global__ static void ComputeElementElasticEnergyIntegral(const vec2_t *position_gpu, const unsigned *trivIdx,
                                                                   const real *lambda_gpu, const real *mu_gpu, const real *triArea,
                                                                   const mat2_t *triDmInv,
                                                                   real *elastic_energy_gpu,
                                                                   const integer element_num)
        {
            const integer e = static_cast<integer>(threadIdx.x + blockIdx.x * blockDim.x);
            if (e >= element_num)
                return;

            real energy = 0;
            int vidx_0 = trivIdx[3 * e];
            int vidx_1 = trivIdx[3 * e + 1];
            int vidx_2 = trivIdx[3 * e + 2];
            vec2_t v0 = position_gpu[vidx_0];
            vec2_t v1 = position_gpu[vidx_1];
            vec2_t v2 = position_gpu[vidx_2];
            real Ds[4];
            Ds[0] = v0[0] - v2[0];
            Ds[2] = v1[0] - v2[0];
            Ds[1] = v0[1] - v2[1];
            Ds[3] = v1[1] - v2[1];

            mat2_t DmInv_FaInv = triDmInv[e];
            real DmInvFaInv[4] = {
                DmInv_FaInv[0][0], DmInv_FaInv[1][0],
                DmInv_FaInv[0][1], DmInv_FaInv[1][1]};

            real F[4];
            // F = Ds * DmInvFaInv.
            F[0] = Ds[0] * DmInvFaInv[0] + Ds[2] * DmInvFaInv[1];
            F[2] = Ds[0] * DmInvFaInv[2] + Ds[2] * DmInvFaInv[3];
            F[1] = Ds[1] * DmInvFaInv[0] + Ds[3] * DmInvFaInv[1];
            F[3] = Ds[1] * DmInvFaInv[2] + Ds[3] * DmInvFaInv[3];
            energy = NeohookeanComputeEnergyDensityFromDeformationGradientGpu2d(lambda_gpu[e], mu_gpu[e], F) * triArea[e];

            elastic_energy_gpu[e] = energy;
        }

        __global__ void ComputeLBSEnergyIntegral(const vec2_t *position_gpu, const vec2_t *vpos_lbs_gpu,
                                                 real *lbs_energy_integral_gpu,
                                                 const real *stiffness, const integer vNum)
        {
            const integer v = static_cast<integer>(threadIdx.x + blockIdx.x * blockDim.x);
            if (v >= vNum)
                return;

            // Compute LBS energy.
            auto diff = position_gpu[v] - vpos_lbs_gpu[v];
            lbs_energy_integral_gpu[v] = stiffness[v] * glm::dot(diff, diff);
        }

        real ComputeElementElasticEnergyGpu(const Eigen::Matrix<real, 2, Eigen::Dynamic> &position, VbdSceneDataGpu2D &scene_data, control::LBSDataGpu2D &lbs_data, cudaStream_t stream)
        {

            // We loop over all elements in parallel.
            const integer element_num = scene_data.triangle_num;
            const integer vNum = scene_data.vertex_num;
            const integer block_size = 256;
            const integer grid_size = (element_num + block_size - 1) / block_size;

            ///////////////////////////////////////////////
            // Prepare data.
            ///////////////////////////////////////////////
            std::vector<vec2_t> tmp_pos(vNum);
            copyEigenToGlm(position, tmp_pos);
            scene_data.itr_pre_pre_positions.uploadAsync(tmp_pos, stream);
            lbs_data.elastic_energy_integral.setZeroAsync(stream);
            lbs_data.lbs_energy_integral.setZeroAsync(stream);

            ///////////////////////////////////////////////
            // Call Cuda kernels.
            ///////////////////////////////////////////////

            ComputeElementElasticEnergyIntegral<<<grid_size, block_size, 0, stream>>>(scene_data.itr_pre_pre_positions.data(), scene_data.triangles_indices.data(),
                                                                                      scene_data.tri_lambda.data(), scene_data.tri_mu.data(), scene_data.tri_areas.data(),
                                                                                      lbs_data.tri_DmInv.data(),
                                                                                      lbs_data.elastic_energy_integral.data(),
                                                                                      element_num);

            const integer grid_size2 = (vNum + block_size - 1) / block_size;

            ComputeLBSEnergyIntegral<<<grid_size2, block_size, 0, stream>>>(scene_data.itr_pre_pre_positions.data(), lbs_data.position_lbs.data(),
                                                                            lbs_data.lbs_energy_integral.data(),
                                                                            lbs_data.stiffness.data(), vNum);

            // Wait for GPU programs to finish.
            CUDA_CHECK(cudaStreamSynchronize(stream));

            ///////////////////////////////////////////////
            // Copy results back to CPU.
            ///////////////////////////////////////////////
            thrust::device_ptr<real> d_vec(lbs_data.elastic_energy_integral.data());
            real elastic_energy = thrust::reduce(d_vec, d_vec + element_num, 0.0, thrust::plus<real>());

            thrust::device_ptr<real> d_vec_lbs(lbs_data.lbs_energy_integral.data());
            real lbs_energy = thrust::reduce(d_vec_lbs, d_vec_lbs + vNum, 0.0, thrust::plus<real>());

            return elastic_energy + lbs_energy;
        }

        __global__ static void ComputeElasticGradientIntegral(
            const vec2_t *position_gpu, const unsigned *trivIdx,
            const real *lambda_gpu, const real *mu_gpu, const real *triArea,
            const mat2_t *actuation_deformation_gradient_inverse_gpu,
            const real *triAt_gpu,
            real *elastic_gradient_integral_gpu,
            const integer element_num)
        {
            const integer e = static_cast<integer>(threadIdx.x + blockIdx.x * blockDim.x);
            if (e >= element_num)
                return;

            int vidx_0 = trivIdx[3 * e];
            int vidx_1 = trivIdx[3 * e + 1];
            int vidx_2 = trivIdx[3 * e + 2];
            vec2_t v0 = position_gpu[vidx_0];
            vec2_t v1 = position_gpu[vidx_1];
            vec2_t v2 = position_gpu[vidx_2];
            real Ds[4];
            Ds[0] = v0[0] - v2[0];
            Ds[2] = v1[0] - v2[0];
            Ds[1] = v0[1] - v2[1];
            Ds[3] = v1[1] - v2[1];

            mat2_t DmInv_FaInv = actuation_deformation_gradient_inverse_gpu[e];
            real DmInvFaInv[4] = {
                DmInv_FaInv[0][0], DmInv_FaInv[1][0],
                DmInv_FaInv[0][1], DmInv_FaInv[1][1]};

            real F[4], P[4];
            // F = Ds * DmInvFaInv.
            F[0] = Ds[0] * DmInvFaInv[0] + Ds[2] * DmInvFaInv[1];
            F[2] = Ds[0] * DmInvFaInv[2] + Ds[2] * DmInvFaInv[3];
            F[1] = Ds[1] * DmInvFaInv[0] + Ds[3] * DmInvFaInv[1];
            F[3] = Ds[1] * DmInvFaInv[2] + Ds[3] * DmInvFaInv[3];

            NeohookeanComputeStressTensorFromDeformationGradientGpu2d(lambda_gpu[e], mu_gpu[e], F, P);

            const real *Ate = triAt_gpu + e * 24;
            for (int j = 0; j < 3; j++)
            {
                for (int i = 0; i < 2; i++)
                {
                    int idx = j * 2 + i;
                    real val = 0;
                    for (int k = 0; k < 4; k++)
                    {
                        val += P[k] * Ate[k * 6 + idx];
                    }
                    elastic_gradient_integral_gpu[6 * e + idx] = val * triArea[e];
                }
            }
        }

        __global__ static void ElasticGradientValueAssemble(
            const real *elastic_gradient_integral_gpu,
            const integer *elastic_gradient_map_begin_index_gpu,
            const integer *elastic_gradient_map_gpu,
            const integer dof_num,
            real *elastic_gradient_value_ptr_gpu,
            const vec2_t *position_gpu,
            const vec2_t *position_lbs_gpu,
            const real *stiffness)
        {
            const integer k = static_cast<integer>(threadIdx.x + blockIdx.x * blockDim.x);
            if (k >= dof_num)
                return;

            const integer begin_index = elastic_gradient_map_begin_index_gpu[k];
            const integer end_index = elastic_gradient_map_begin_index_gpu[k + 1];
            Eigen::Matrix<real, 2, 1> val;
            val.setZero();
            for (integer i = begin_index; i < end_index; ++i)
            {
                for (integer d = 0; d < 2; ++d)
                    val(d) += elastic_gradient_integral_gpu[2 * elastic_gradient_map_gpu[i] + d];
            }
            for (integer d = 0; d < 2; ++d)
                elastic_gradient_value_ptr_gpu[2 * k + d] = val(d) + 2 * stiffness[k] * (position_gpu[k][d] - position_lbs_gpu[k][d]);
        }

        Eigen::Matrix<real, 2, Eigen::Dynamic> ComputeElasticForceGpu(
            const Eigen::Matrix<real, 2, Eigen::Dynamic> &position, VbdSceneDataGpu2D &scene_data, control::LBSDataGpu2D &lbs_data, cudaStream_t stream)
        {

            // We loop over all elements in parallel.
            const integer element_num = scene_data.triangle_num;
            const integer dof_num = scene_data.vertex_num;
            const integer block_size = 256;
            const integer grid_size = (element_num + block_size - 1) / block_size;

            ///////////////////////////////////////////////
            // Prepare data.
            ///////////////////////////////////////////////
            std::vector<vec2_t> tmp_pos(dof_num);
            copyEigenToGlm(position, tmp_pos);
            scene_data.itr_pre_pre_positions.uploadAsync(tmp_pos, stream);
            lbs_data.elastic_gradient_integral.setZeroAsync(stream);
            lbs_data.lbs_energy_integral.setZeroAsync(stream);

            ///////////////////////////////////////////////
            // Call Cuda kernels.
            ///////////////////////////////////////////////

            ComputeElasticGradientIntegral<<<grid_size, block_size, 0, stream>>>(scene_data.itr_pre_pre_positions.data(), scene_data.triangles_indices.data(),
                                                                                 scene_data.tri_lambda.data(), scene_data.tri_mu.data(), scene_data.tri_areas.data(),
                                                                                 lbs_data.tri_DmInv.data(),
                                                                                 scene_data.tri_At.data(),
                                                                                 lbs_data.elastic_gradient_integral.data(),
                                                                                 element_num);

            const integer grid_size2 = (dof_num + block_size - 1) / block_size;

            ElasticGradientValueAssemble<<<grid_size2, block_size, 0, stream>>>(lbs_data.elastic_gradient_integral.data(),
                                                                                lbs_data.elastic_gradient_map_begin_index.data(), lbs_data.elastic_gradient_map.data(), dof_num,
                                                                                lbs_data.elastic_gradient_value_ptr.data(),
                                                                                scene_data.itr_pre_pre_positions.data(), lbs_data.position_lbs.data(),
                                                                                lbs_data.stiffness.data());

            ///////////////////////////////////////////////
            // Copy results back to CPU.
            ///////////////////////////////////////////////

            Eigen::Matrix<real, 2, Eigen::Dynamic> force(2, dof_num);
            CUDA_CHECK(cudaMemcpyAsync(force.data(), lbs_data.elastic_gradient_value_ptr.data(), 2 * dof_num * sizeof(real), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            return force;
        }

        __global__ static void ComputeElasticHessianAndProjectionIntegral(
            const vec2_t *position_gpu, const unsigned *trivIdx,
            const real *lambda_gpu, const real *mu_gpu, const real *triArea,
            const mat2_t *actuation_deformation_gradient_inverse_gpu,
            const real *triAt_gpu,
            real *elastic_hessian_integral_gpu,
            real *elastic_hessian_projection_integral_gpu,
            const integer element_num,
            const bool compute_hessian,
            const bool compute_projection)
        {
            const integer e = static_cast<integer>(threadIdx.x + blockIdx.x * blockDim.x);
            if (e >= element_num)
                return;

            int vidx_0 = trivIdx[3 * e];
            int vidx_1 = trivIdx[3 * e + 1];
            int vidx_2 = trivIdx[3 * e + 2];
            vec2_t v0 = position_gpu[vidx_0];
            vec2_t v1 = position_gpu[vidx_1];
            vec2_t v2 = position_gpu[vidx_2];
            real Ds[4];
            Ds[0] = v0[0] - v2[0];
            Ds[2] = v1[0] - v2[0];
            Ds[1] = v0[1] - v2[1];
            Ds[3] = v1[1] - v2[1];

            mat2_t DmInv_FaInv = actuation_deformation_gradient_inverse_gpu[e];
            real DmInvFaInv[4] = {
                DmInv_FaInv[0][0], DmInv_FaInv[1][0],
                DmInv_FaInv[0][1], DmInv_FaInv[1][1]};

            real F[4];
            // F = Ds * DmInvFaInv.
            F[0] = Ds[0] * DmInvFaInv[0] + Ds[2] * DmInvFaInv[1];
            F[2] = Ds[0] * DmInvFaInv[2] + Ds[2] * DmInvFaInv[3];
            F[1] = Ds[1] * DmInvFaInv[0] + Ds[3] * DmInvFaInv[1];
            F[3] = Ds[1] * DmInvFaInv[2] + Ds[3] * DmInvFaInv[3];
            Matrix4r dPdF;
            NeohookeanComputeStressTensorDifferentialFromDeformationGradientGpu2d(lambda_gpu[e], mu_gpu[e], F, dPdF.data());
            Matrix4r dPdF_proj = dPdF;

            // if (compute_projection)
            // {
            //     Eigen::SelfAdjointEigenSolver<Eigen::Matrix<real, 9, 9>> eig_solver(dPdF);
            //     const VectorXr &la = eig_solver.eigenvalues();
            //     const MatrixXr &V = eig_solver.eigenvectors();
            //     // V * la.asDiagonal() * V.transpose() = dPdF.
            //     dPdF_proj = V * la.cwiseMax(Eigen::Matrix<real, 9, 1>::Zero()).asDiagonal() * V.transpose();
            // }

            const real *Ate = triAt_gpu + e * 24;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    for (int di = 0; di < 2; di++)
                    {
                        for (int dj = 0; dj < 2; dj++)
                        {
                            Vector4r dFdxdii;
                            int rowii = i * 2 + di;
                            dFdxdii << Ate[rowii],
                                Ate[rowii + 6],
                                Ate[rowii + 12],
                                Ate[rowii + 18];
                            Vector4r dFdxdjj;
                            int rowjj = j * 2 + dj;
                            dFdxdjj << Ate[rowjj],
                                Ate[rowjj + 6],
                                Ate[rowjj + 12],
                                Ate[rowjj + 18];
                            int idx = 36 * e + 6 * rowii + rowjj;
                            if (compute_hessian)
                                elastic_hessian_integral_gpu[idx] = dFdxdii.dot(dPdF * dFdxdjj) * triArea[e];
                            if (compute_projection)
                                elastic_hessian_projection_integral_gpu[idx] = dFdxdii.dot(dPdF_proj * dFdxdjj) * triArea[e];
                        }
                    }
                }
            }
        }

        __global__ static void ElasticHessianValueAssemble2d(
            const real *elastic_hessian_integral_gpu,
            const integer *elastic_hessian_nonzero_map_begin_index_gpu,
            const integer *elastic_hessian_nonzero_map_gpu,
            const integer elastic_hessian_nonzero_num,
            real *elastic_hessian_value_ptr_gpu)
        {
            const integer k = static_cast<integer>(threadIdx.x + blockIdx.x * blockDim.x);
            if (k >= elastic_hessian_nonzero_num)
                return;
            elastic_hessian_value_ptr_gpu[k] = 0;

            const integer begin_index = elastic_hessian_nonzero_map_begin_index_gpu[k];
            const integer end_index = elastic_hessian_nonzero_map_begin_index_gpu[k + 1];
            real val = 0;
            for (integer i = begin_index; i < end_index; ++i)
            {
                val += elastic_hessian_integral_gpu[elastic_hessian_nonzero_map_gpu[i]];
            }
            elastic_hessian_value_ptr_gpu[k] = val;
        }

        std::pair<SparseMatrixXr, SparseMatrixXr> ComputeElasticHessianAndProjectionGpu(
            const Eigen::Matrix<real, 2, Eigen::Dynamic> &position, VbdSceneDataGpu2D &scene_data, control::LBSDataGpu2D &lbs_data,
            const bool compute_hessian, const bool compute_projection, cudaStream_t stream)
        {

            std::pair<SparseMatrixXr, SparseMatrixXr> ret{lbs_data.elastic_hessian_, lbs_data.elastic_hessian_projection_};

            // Basic size.
            // We loop over all elements in parallel.
            const integer element_num = scene_data.triangle_num;
            const integer dof_num = scene_data.vertex_num;
            const integer elastic_hessian_nonzero_num_ = lbs_data.elastic_hessian_nonzero_num_;
            const integer block_size = 256;
            const integer grid_size = (element_num + block_size - 1) / block_size;

            ///////////////////////////////////////////////
            // Prepare data.
            ///////////////////////////////////////////////
            std::vector<vec2_t> tmp_pos(dof_num);
            copyEigenToGlm(position, tmp_pos);
            scene_data.itr_pre_positions.uploadAsync(tmp_pos, stream);
            lbs_data.elastic_hessian_integral.setZeroAsync(stream);
            lbs_data.elastic_hessian_projection_integral.setZeroAsync(stream);

            ///////////////////////////////////////////////
            // Call Cuda kernels.
            ///////////////////////////////////////////////
            // Part I: Compute Hessian.
            ComputeElasticHessianAndProjectionIntegral<<<grid_size, block_size, 0, stream>>>(
                scene_data.itr_pre_positions.data(), scene_data.triangles_indices.data(),
                scene_data.tri_lambda.data(), scene_data.tri_mu.data(), scene_data.tri_areas.data(),
                lbs_data.tri_DmInv.data(),
                scene_data.tri_At.data(),
                lbs_data.elastic_hessian_integral.data(), lbs_data.elastic_hessian_projection_integral.data(),
                element_num,
                compute_hessian, compute_projection);

            // Wait for GPU programs to finish.
            // checkCudaErrors(cudaDeviceSynchronize());

            // Part II: Assemble local hessian to sparse matrix value buffer.
            const integer assemble_block_size = 256;
            const integer assemble_grid_size = (elastic_hessian_nonzero_num_ + assemble_block_size - 1) / assemble_block_size;

            if (compute_hessian)
            {
                ElasticHessianValueAssemble2d<<<assemble_grid_size, assemble_block_size, 0, stream>>>(
                    lbs_data.elastic_hessian_integral.data(),
                    lbs_data.elastic_hessian_nonzero_map_begin_index.data(),
                    lbs_data.elastic_hessian_nonzero_map.data(),
                    lbs_data.elastic_hessian_nonzero_num_,
                    lbs_data.elastic_hessian_value_ptr.data());
                // checkCudaErrors(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpyAsync(ret.first.valuePtr(), lbs_data.elastic_hessian_value_ptr.data(), elastic_hessian_nonzero_num_ * sizeof(real), cudaMemcpyDeviceToHost, stream));
            }
            if (compute_projection)
            {
                ElasticHessianValueAssemble2d<<<assemble_grid_size, assemble_block_size, 0, stream>>>(
                    lbs_data.elastic_hessian_projection_integral.data(),
                    lbs_data.elastic_hessian_nonzero_map_begin_index.data(),
                    lbs_data.elastic_hessian_nonzero_map.data(),
                    lbs_data.elastic_hessian_nonzero_num_,
                    lbs_data.elastic_hessian_value_ptr.data());
                // checkCudaErrors(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpyAsync(ret.second.valuePtr(), lbs_data.elastic_hessian_value_ptr.data(), elastic_hessian_nonzero_num_ * sizeof(real), cudaMemcpyDeviceToHost, stream));
            }

            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (compute_hessian)
                ret.first += lbs_data.hessian_lbs_;
            if (compute_projection)
                ret.second += lbs_data.hessian_lbs_;

            return ret;
        }

    }
}