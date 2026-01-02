// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/fem/NewtonKernels.cuh"
#include "FSI_Simulator/utils/NumericUtils.cuh"
#include "FSI_Simulator/utils/Profiler.hpp"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

namespace fsi
{
    namespace fem
    {

        __device__ const Matrix3r DeterminantGradientGpu(const Matrix3r &A)
        {
            // dJ/dA = JA^-T
            // A = [ a0 | a1 | a2 ].
            // J = a0.dot(a1 x a2).
            // dJ/dA = [ a1 x a2 | a2 x a0 | a0 x a1 ]
            Matrix3r dJdA;
            dJdA.col(0) = A.col(1).cross(A.col(2));
            dJdA.col(1) = A.col(2).cross(A.col(0));
            dJdA.col(2) = A.col(0).cross(A.col(1));
            return dJdA;
        }

        __device__ const Matrix3r CrossProductMatrixGpu(const Vector3r &a)
        {
            Matrix3r A = Matrix3r::Zero();
            A(1, 0) = a.z();
            A(2, 0) = -a.y();
            A(0, 1) = -a.z();
            A(2, 1) = a.x();
            A(0, 2) = a.y();
            A(1, 2) = -a.x();
            return A;
        }

        __device__ Matrix9r DeterminantHessianGpu(const Matrix3r &A)
        {
            Matrix9r H = Matrix9r::Zero();
            const Matrix3r A0 = CrossProductMatrixGpu(A.col(0));
            const Matrix3r A1 = CrossProductMatrixGpu(A.col(1));
            const Matrix3r A2 = CrossProductMatrixGpu(A.col(2));
            H.block<3, 3>(0, 3) += -A2;
            H.block<3, 3>(0, 6) += A1;
            H.block<3, 3>(3, 0) += A2;
            H.block<3, 3>(3, 6) += -A0;
            H.block<3, 3>(6, 0) += -A1;
            H.block<3, 3>(6, 3) += A0;
            return H;
        }

        __device__ real NeohookeanComputeEnergyDensityFromDeformationGradientGpu(const real la, const real mu, const real *deformation_grad)
        {
            const integer dim = 3;
            const Eigen::Map<const Matrix3r> F(deformation_grad);
            const Matrix3r C = F.transpose() * F;
            const real J = F.determinant();
            const real Ic = C.trace();
            const real delta = 1;
            const real alpha = (1 - 1 / (dim + delta)) * mu / la + 1;
            return mu / 2 * (Ic - dim) + la / 2 * (J - alpha) * (J - alpha) - 0.5 * mu * std::log(Ic + delta);
        }

        __device__ void NeohookeanComputeStressTensorFromDeformationGradientGpu(const real la, const real mu, const real *deformation_grad,
                                                                                real *stress_tensor)
        {
            const integer dim = 3;
            const Eigen::Map<const Matrix3r> F(deformation_grad);
            const Matrix3r C = F.transpose() * F;
            const real J = F.determinant();
            const real Ic = C.trace();
            const real delta = 1;
            const real alpha = (1 - 1 / (dim + delta)) * mu / la + 1;
            const Matrix3r dJdF = DeterminantGradientGpu(Matrix3r(F));
            Eigen::Map<Matrix3r> P(stress_tensor);
            P = (1 - 1 / (Ic + delta)) * mu * F + la * (J - alpha) * dJdF;
        }

        __global__ void NeohookeanComputeStressTensorFromDeformationGradientGpu_test(const real la, const real mu, const real *deformation_grad, real *stress_tensor)
        {
            // This is a test kernel to compute the stress tensor from the deformation gradient.
            NeohookeanComputeStressTensorFromDeformationGradientGpu(la, mu, deformation_grad, stress_tensor);
        }

        __device__ void NeohookeanComputeStressTensorDifferentialFromDeformationGradientGpu(
            const real la, const real mu, const real *deformation_grad, real *stress_tensor_differential)
        {

            const integer dim = 3;
            Eigen::Map<const Matrix3r> F(deformation_grad);
            const Matrix3r C = F.transpose() * F;
            const real J = F.determinant();
            const real Ic = C.trace();
            const real delta = 1;
            const real alpha = (1 - 1 / (dim + delta)) * mu / la + 1;
            // dJ/dF = JF^-T
            // F = [ f0 | f1 | f2 ].
            // J = f0.dot(f1 x f2).
            // dJ/dF = [ f1 x f2 | f2 x f0 | f0 x f1 ]
            const Matrix3r dJdF = DeterminantGradientGpu(Matrix3r(F));
            const Vector3r f0 = F.col(0);
            const Vector3r f1 = F.col(1);
            const Vector3r f2 = F.col(2);
            // P = (1 - 1 / (Ic + delta)) * mu * F + la * (J - alpha) * dJdF.
            // Part I:
            const Vector9r f = F.reshaped();
            Eigen::Map<Matrix9r> dPdF(stress_tensor_differential);
            dPdF = ((1 - 1 / (Ic + delta)) * mu) * Matrix9r::Identity() + (2 * mu / ((Ic + delta) * (Ic + delta))) * (f * f.transpose());
            // Part II: la * (J - alpha) * dJdF.
            const Vector9r djdf = dJdF.reshaped();
            dPdF += la * djdf * djdf.transpose();
            // The trickiest part in part II:
            dPdF += la * (J - alpha) * DeterminantHessianGpu(Matrix3r(F));
            // for (int i = 0; i < 9; ++i) {
            //     for (int j = 0; j < 9; ++j) {
            //         stress_tensor_differential[i + 9 * j] = dPdF(i, j);
            //     }
            // }
            // Regarding delta and alpha:
            // F = 0, J = 0, Ic = 0.
            // eig = mu * (1 - 1 / delta) = 0 => delta = 1.
        }

        __global__ static void ComputeElementElasticEnergyIntegral(const vec3_t *position_gpu, const unsigned *tetvIdx,
                                                                   const real *lambda_gpu, const real *mu_gpu, const real *tetVolume,
                                                                   const mat3_t *tetDmInv,
                                                                   real *elastic_energy_gpu,
                                                                   const integer element_num)
        {
            const integer e = static_cast<integer>(threadIdx.x + blockIdx.x * blockDim.x);
            if (e >= element_num)
                return;

            real energy = 0;
            int vidx_0 = tetvIdx[4 * e];
            int vidx_1 = tetvIdx[4 * e + 1];
            int vidx_2 = tetvIdx[4 * e + 2];
            int vidx_3 = tetvIdx[4 * e + 3];
            Matrix3r DmInvFaInv = Eigen::Map<const Matrix3r>(glm::value_ptr(tetDmInv[e])).transpose();
            Matrix3r Ds;
            for (int i = 0; i < 3; i++)
            {
                Ds(i, 0) = position_gpu[vidx_1][i] - position_gpu[vidx_0][i];
                Ds(i, 1) = position_gpu[vidx_2][i] - position_gpu[vidx_0][i];
                Ds(i, 2) = position_gpu[vidx_3][i] - position_gpu[vidx_0][i];
            }
            Matrix3r F = Ds * DmInvFaInv;
            energy = NeohookeanComputeEnergyDensityFromDeformationGradientGpu(lambda_gpu[e], mu_gpu[e], F.data()) * tetVolume[e];

            elastic_energy_gpu[e] = energy;
        }

        __global__ void ComputeLBSEnergyIntegral(const vec3_t *position_gpu, const vec3_t *vpos_lbs_gpu,
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

        real ComputeElementElasticEnergyGpu(const Eigen::Matrix<real, 3, Eigen::Dynamic> &position, VbdSceneDataGpu3D &scene_data, control::LBSDataGpu3D &lbs_data, cudaStream_t stream)
        {
            PROFILE_CUDA_SCOPE("ComputeElementElasticEnergyGpu", stream);
            // We loop over all elements in parallel.
            const integer element_num = scene_data.tetrahedron_num;
            const integer vNum = scene_data.vertex_num;
            const integer block_size = 256;
            const integer grid_size = (element_num + block_size - 1) / block_size;

            ///////////////////////////////////////////////
            // Prepare data.
            ///////////////////////////////////////////////
            std::vector<vec3_t> tmp_pos(vNum);
            copyEigenToGlm(position, tmp_pos);
            scene_data.itr_pre_pre_positions.uploadAsync(tmp_pos, stream);
            lbs_data.elastic_energy_integral.setZeroAsync(stream);
            lbs_data.lbs_energy_integral.setZeroAsync(stream);

            ///////////////////////////////////////////////
            // Call Cuda kernels.
            ///////////////////////////////////////////////

            ComputeElementElasticEnergyIntegral<<<grid_size, block_size, 0, stream>>>(scene_data.itr_pre_pre_positions.data(), scene_data.tetrahedra_indices.data(),
                                                                                      scene_data.tet_lambda.data(), scene_data.tet_mu.data(), scene_data.tet_volumes.data(),
                                                                                      lbs_data.tet_DmInv.data(),
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
            const vec3_t *position_gpu, const unsigned *tetvIdx,
            const real *lambda_gpu, const real *mu_gpu, const real *tetVolume,
            const mat3_t *actuation_deformation_gradient_inverse_gpu,
            const real *tetAt_gpu,
            real *elastic_gradient_integral_gpu,
            const integer element_num)
        {
            const integer e = static_cast<integer>(threadIdx.x + blockIdx.x * blockDim.x);
            if (e >= element_num)
                return;

            int vidx_0 = tetvIdx[4 * e];
            int vidx_1 = tetvIdx[4 * e + 1];
            int vidx_2 = tetvIdx[4 * e + 2];
            int vidx_3 = tetvIdx[4 * e + 3];
            Matrix3r DmInvFaInv = Eigen::Map<const Matrix3r>(glm::value_ptr(actuation_deformation_gradient_inverse_gpu[e])).transpose();
            Matrix3r Ds;
            for (size_t i = 0; i < 3; i++)
            {
                Ds(i, 0) = position_gpu[vidx_1][i] - position_gpu[vidx_0][i];
                Ds(i, 1) = position_gpu[vidx_2][i] - position_gpu[vidx_0][i];
                Ds(i, 2) = position_gpu[vidx_3][i] - position_gpu[vidx_0][i];
            }

            Matrix3r F = Ds * DmInvFaInv;
            real P[9];
            NeohookeanComputeStressTensorFromDeformationGradientGpu(lambda_gpu[e], mu_gpu[e], F.data(), P);

            // if (e == 123)
            // {
            //     printf("e: %d, Ds: %f %f %f, %f %f %f, %f %f %f\n", e,
            //         Ds(0, 0), Ds(0, 1), Ds(0, 2),
            //         Ds(1, 0), Ds(1, 1), Ds(1, 2),
            //         Ds(2, 0), Ds(2, 1), Ds(2, 2));
            //     printf("e: %d, F: %f %f %f, %f %f %f, %f %f %f\n", e,
            //         F(0, 0), F(0, 1), F(0, 2),
            //         F(1, 0), F(1, 1), F(1, 2),
            //         F(2, 0), F(2, 1), F(2, 2));
            //     printf("e: %d, P: %f %f %f %f %f %f %f %f %f\n", e,
            //         P(0, 0), P(0, 1), P(0, 2),
            //         P(1, 0), P(1, 1), P(1, 2),
            //         P(2, 0), P(2, 1), P(2, 2));
            // }

            const real *Ate = tetAt_gpu + e * 108;
            for (int j = 0; j < 4; j++)
            {
                for (int i = 0; i < 3; i++)
                {
                    int idx = j * 3 + i;
                    real val = 0;
                    for (int k = 0; k < 9; k++)
                    {
                        val += P[k] * Ate[k * 12 + idx];
                    }
                    elastic_gradient_integral_gpu[12 * e + idx] = val * tetVolume[e];
                }
            }
        }

        __global__ static void ElasticGradientValueAssemble(
            const real *elastic_gradient_integral_gpu,
            const integer *elastic_gradient_map_begin_index_gpu,
            const integer *elastic_gradient_map_gpu,
            const integer dof_num,
            real *elastic_gradient_value_ptr_gpu,
            const vec3_t *position_gpu,
            const vec3_t *position_lbs_gpu,
            const real *stiffness)
        {
            const integer k = static_cast<integer>(threadIdx.x + blockIdx.x * blockDim.x);
            if (k >= dof_num)
                return;

            const integer begin_index = elastic_gradient_map_begin_index_gpu[k];
            const integer end_index = elastic_gradient_map_begin_index_gpu[k + 1];
            Eigen::Matrix<real, 3, 1> val;
            val.setZero();
            for (integer i = begin_index; i < end_index; ++i)
            {
                for (integer d = 0; d < 3; ++d)
                    val(d) += elastic_gradient_integral_gpu[3 * elastic_gradient_map_gpu[i] + d];
            }
            for (integer d = 0; d < 3; ++d)
                elastic_gradient_value_ptr_gpu[3 * k + d] = val(d) + 2 * stiffness[k] * (position_gpu[k][d] - position_lbs_gpu[k][d]);
        }

        Eigen::Matrix<real, 3, Eigen::Dynamic> ComputeElasticForceGpu(
            const Eigen::Matrix<real, 3, Eigen::Dynamic> &position, VbdSceneDataGpu3D &scene_data, control::LBSDataGpu3D &lbs_data, cudaStream_t stream)
        {

            PROFILE_CUDA_SCOPE("ComputeElasticForceGpu", stream);
            // Basic size.
            // We loop over all elements in parallel.
            const integer element_num = scene_data.tetrahedron_num;
            const integer dof_num = scene_data.vertex_num;
            const integer block_size = 256;
            const integer grid_size = (element_num + block_size - 1) / block_size;

            ///////////////////////////////////////////////
            // Prepare data.
            ///////////////////////////////////////////////
            std::vector<vec3_t> tmp_pos(dof_num);
            copyEigenToGlm(position, tmp_pos);
            scene_data.itr_pre_positions.uploadAsync(tmp_pos, stream);

            lbs_data.elastic_gradient_integral.setZeroAsync(stream);

            ///////////////////////////////////////////////
            // Call Cuda kernels.
            ///////////////////////////////////////////////
            // Part I: Compute Gradient.
            ComputeElasticGradientIntegral<<<grid_size, block_size, 0, stream>>>(
                scene_data.itr_pre_positions.data(), scene_data.tetrahedra_indices.data(),
                scene_data.tet_lambda.data(), scene_data.tet_mu.data(), scene_data.tet_volumes.data(),
                lbs_data.tet_DmInv.data(),
                scene_data.tet_At.data(),
                lbs_data.elastic_gradient_integral.data(),
                element_num);

            // Wait for GPU programs to finish.
            // cudaDeviceSynchronize();

            // Part II: Assemble local gradients.
            const integer assemble_block_size = 256;
            const integer assemble_grid_size = (dof_num + assemble_block_size - 1) / assemble_block_size;

            ElasticGradientValueAssemble<<<assemble_grid_size, assemble_block_size, 0, stream>>>(
                lbs_data.elastic_gradient_integral.data(),
                lbs_data.elastic_gradient_map_begin_index.data(),
                lbs_data.elastic_gradient_map.data(), dof_num,
                lbs_data.elastic_gradient_value_ptr.data(),
                scene_data.itr_pre_positions.data(),
                lbs_data.position_lbs.data(), lbs_data.stiffness.data());

            Eigen::Matrix<real, 3, Eigen::Dynamic> force = Eigen::Matrix<real, 3, Eigen::Dynamic>::Zero(3, dof_num);
            CUDA_CHECK(cudaMemcpyAsync(force.data(), lbs_data.elastic_gradient_value_ptr.data(), 3 * dof_num * sizeof(real), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            return force;
        }

        __global__ static void ComputeElasticHessianAndProjectionIntegral(
            const vec3_t *position_gpu, const unsigned *tetvIdx,
            const real *lambda_gpu, const real *mu_gpu, const real *tet_volume_gpu,
            const mat3_t *actuation_deformation_gradient_inverse_gpu,
            const real *tetAt_gpu,
            real *elastic_hessian_integral_gpu,
            real *elastic_hessian_projection_integral_gpu,
            const integer element_num,
            const bool compute_hessian,
            const bool compute_projection)
        {
            const integer e = static_cast<integer>(threadIdx.x + blockIdx.x * blockDim.x);
            if (e >= element_num)
                return;

            int vidx_0 = tetvIdx[4 * e];
            int vidx_1 = tetvIdx[4 * e + 1];
            int vidx_2 = tetvIdx[4 * e + 2];
            int vidx_3 = tetvIdx[4 * e + 3];
            Matrix3r DmInvFaInv = Eigen::Map<const Matrix3r>(glm::value_ptr(actuation_deformation_gradient_inverse_gpu[e])).transpose();
            Matrix3r Ds;
            for (size_t i = 0; i < 3; i++)
            {
                Ds(i, 0) = position_gpu[vidx_1][i] - position_gpu[vidx_0][i];
                Ds(i, 1) = position_gpu[vidx_2][i] - position_gpu[vidx_0][i];
                Ds(i, 2) = position_gpu[vidx_3][i] - position_gpu[vidx_0][i];
            }

            Matrix3r F = Ds * DmInvFaInv;
            Matrix9r dPdF;
            NeohookeanComputeStressTensorDifferentialFromDeformationGradientGpu(lambda_gpu[e], mu_gpu[e], F.data(), dPdF.data());
            Matrix9r dPdF_proj = dPdF;

            // if (compute_projection)
            // {
            //     Eigen::SelfAdjointEigenSolver<Eigen::Matrix<real, 9, 9>> eig_solver(dPdF);
            //     const VectorXr &la = eig_solver.eigenvalues();
            //     const MatrixXr &V = eig_solver.eigenvectors();
            //     // V * la.asDiagonal() * V.transpose() = dPdF.
            //     dPdF_proj = V * la.cwiseMax(Eigen::Matrix<real, 9, 1>::Zero()).asDiagonal() * V.transpose();
            // }

            const real *Ate = tetAt_gpu + e * 108;
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int di = 0; di < 3; di++)
                    {
                        for (int dj = 0; dj < 3; dj++)
                        {
                            // Compute dFdx(di, i) and dFdx(dj, j).
                            // Vector9r dFdxdii = (Ate.row(i * 3 + di).reshaped(3, 3)).reshaped();
                            Vector9r dFdxdii;
                            int rowii = i * 3 + di;
                            dFdxdii << Ate[rowii],
                                Ate[rowii + 12],
                                Ate[rowii + 24],
                                Ate[rowii + 36],
                                Ate[rowii + 48],
                                Ate[rowii + 60],
                                Ate[rowii + 72],
                                Ate[rowii + 84],
                                Ate[rowii + 96];
                            // Vector9r dFdxdjj = (Ate.row(j * 3 + dj).reshaped(3, 3)).reshaped();
                            Vector9r dFdxdjj;
                            int rowjj = j * 3 + dj;
                            dFdxdjj << Ate[rowjj],
                                Ate[rowjj + 12],
                                Ate[rowjj + 24],
                                Ate[rowjj + 36],
                                Ate[rowjj + 48],
                                Ate[rowjj + 60],
                                Ate[rowjj + 72],
                                Ate[rowjj + 84],
                                Ate[rowjj + 96];
                            int idx = 144 * e + 12 * rowii + rowjj;
                            if (compute_hessian)
                                elastic_hessian_integral_gpu[idx] = dFdxdii.dot(dPdF * dFdxdjj) * tet_volume_gpu[e];
                            if (compute_projection)
                                elastic_hessian_projection_integral_gpu[idx] = dFdxdii.dot(dPdF_proj * dFdxdjj) * tet_volume_gpu[e];
                        }
                    }
                }
            }
        }

        __global__ static void ElasticHessianValueAssemble(
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
            const Eigen::Matrix<real, 3, Eigen::Dynamic> &position, VbdSceneDataGpu3D &scene_data, control::LBSDataGpu3D &lbs_data,
            const bool compute_hessian, const bool compute_projection, cudaStream_t stream)
        {
            PROFILE_CUDA_SCOPE("ComputeElasticHessianAndProjectionGpu", stream);

            std::pair<SparseMatrixXr, SparseMatrixXr> ret{lbs_data.elastic_hessian_, lbs_data.elastic_hessian_projection_};

            // Basic size.
            // We loop over all elements in parallel.
            const integer element_num = scene_data.tetrahedron_num;
            const integer dof_num = scene_data.vertex_num;
            const integer elastic_hessian_nonzero_num_ = lbs_data.elastic_hessian_nonzero_num_;
            const integer block_size = 256;
            const integer grid_size = (element_num + block_size - 1) / block_size;

            ///////////////////////////////////////////////
            // Prepare data.
            ///////////////////////////////////////////////
            std::vector<vec3_t> tmp_pos(dof_num);
            copyEigenToGlm(position, tmp_pos);
            scene_data.itr_pre_positions.uploadAsync(tmp_pos, stream);
            lbs_data.elastic_hessian_integral.setZeroAsync(stream);
            lbs_data.elastic_hessian_projection_integral.setZeroAsync(stream);

            ///////////////////////////////////////////////
            // Call Cuda kernels.
            ///////////////////////////////////////////////
            // Part I: Compute Hessian.
            ComputeElasticHessianAndProjectionIntegral<<<grid_size, block_size, 0, stream>>>(
                scene_data.itr_pre_positions.data(), scene_data.tetrahedra_indices.data(),
                scene_data.tet_lambda.data(), scene_data.tet_mu.data(), scene_data.tet_volumes.data(),
                lbs_data.tet_DmInv.data(),
                scene_data.tet_At.data(),
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
                ElasticHessianValueAssemble<<<assemble_grid_size, assemble_block_size, 0, stream>>>(
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
                ElasticHessianValueAssemble<<<assemble_grid_size, assemble_block_size, 0, stream>>>(
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