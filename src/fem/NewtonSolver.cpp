// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/fem/NewtonSolver.hpp"
#include "FSI_Simulator/optimizer/NewtonOptimizer.hpp"
#include "FSI_Simulator/common/SparseMatrixUtils.hpp"
#include "FSI_Simulator/common/MathUtils.hpp"
#include "FSI_Simulator/fem/NewtonKernels.cuh"

#include "omp.h"

#include <iostream>

namespace fsi
{
    namespace fem
    {
        const real ComputeElasticEnergyFromDeformationGradient(const Matrix3r &F, const real mu, const real la)
        {
            // Neo-Hookean energy.
            // - J = |F|,
            // - C = Ft * F,
            // - Ic = tr(C).
            // \Psi(F) = 0.5mu * (Ic - dim) + lambda / 2 * (J - alpha)^2 - 0.5mu * log(Ic + delta).
            const real dim = 3;
            const real delta = 1;
            const real alpha = (1 - 1 / (dim + delta)) * mu / la + 1;
            const Matrix3r C = F.transpose() * F;
            const real J = F.determinant();
            const real Ic = C.trace();
            return mu / 2. * (Ic - dim) + la / 2. * (J - alpha) * (J - alpha) - 0.5 * mu * std::log(Ic + delta);
        }

        const Matrix3r ComputeStressTensorFromDeformationGradient(
            const Matrix3r &F, const real mu, const real la)
        {

            const real dim = 3;
            const real delta = 1;
            const real alpha = (1 - 1 / (dim + delta)) * mu / la + 1;
            const Matrix3r C = F.transpose() * F;
            const real J = F.determinant();
            const real Ic = C.trace();
            const Matrix3r dJdF = determinantGradient(F);
            return (1. - 1. / (Ic + delta)) * mu * F + la * (J - alpha) * dJdF;
        }

        const Matrix9r ComputeStressTensorDifferentialFromDeformationGradient(
            const Matrix3r &F, const real mu, const real la)
        {
            const real dim = 3;
            const real delta = 1;
            const real alpha = (1 - 1 / (dim + delta)) * mu / la + 1;
            const Matrix3r C = F.transpose() * F;
            const real J = F.determinant();
            const real Ic = C.trace();
            // dJ/dF = JF^-T
            // F = [ f0 | f1 | f2 ].
            // J = f0.dot(f1 x f2).
            // dJ/dF = [ f1 x f2 | f2 x f0 | f0 x f1 ]
            const Matrix3r dJdF = determinantGradient(F);
            const Vector3r f0 = F.col(0);
            const Vector3r f1 = F.col(1);
            const Vector3r f2 = F.col(2);
            // P = (1 - 1 / (Ic + delta)) * mu * F + la * (J - alpha) * dJdF.
            // Part I:
            const Vector9r f = F.reshaped();
            Matrix9r dPdF = ((1 - 1 / (Ic + delta)) * mu) * Matrix9r::Identity() + (2 * mu / ((Ic + delta) * (Ic + delta))) * (f * f.transpose());
            // Part II: la * (J - alpha) * dJdF.
            const Vector9r djdf = dJdF.reshaped();
            dPdF += la * djdf * djdf.transpose();
            // The trickiest part in part II:
            dPdF += la * (J - alpha) * determinantHessian(F);
            return dPdF;
        }

        const real ComputeLBSEnergy(const Matrix3Xr &position, const Matrix3Xr &position_lbs, const std::vector<real> &stiffness)
        {
            real energy = 0;
            int vNum = position.cols();
#pragma omp parallel for reduction(+ : energy)
            for (int i = 0; i < vNum; ++i)
            {
                energy += stiffness[i] * (position.col(i) - position_lbs.col(i)).squaredNorm();
            }
            return energy;
        }

        // LBS gradient.
        const Matrix3Xr ComputeLBSGradient(
            const Matrix3Xr &position,
            const Matrix3Xr &position_lbs,
            const std::vector<real> &stiffness)
        {
            Eigen::Matrix<real, 3, Eigen::Dynamic> gradient = Eigen::Matrix<real, 3, Eigen::Dynamic>::Zero(3, position.cols());
            int vNum = position.cols();
#pragma omp parallel for
            for (int i = 0; i < vNum; ++i)
            {
                gradient.col(i) = 2 * stiffness[i] * (position.col(i) - position_lbs.col(i));
            }
            return gradient;
        }

        const std::pair<SparseMatrixXr, SparseMatrixXr> ComputeLBSHessianAndProjection(
            const Matrix3Xr &position, const std::vector<real> &stiffness,
            const bool compute_hessian, const bool compute_projection)
        {
            std::vector<Eigen::Triplet<real>> lbs_nonzeros;
            int vNum = position.cols();
            for (int i = 0; i < vNum; ++i)
            {
                lbs_nonzeros.push_back(Eigen::Triplet<real>(i, i, 2 * stiffness[i]));
            }
            const SparseMatrixXr H_lbs = FromTriplet(3 * vNum, 3 * vNum, lbs_nonzeros);

            std::pair<SparseMatrixXr, SparseMatrixXr> H;
            if (compute_hessian)
                H.first = H_lbs;
            if (compute_projection)
                H.second = H_lbs;
            return H;
        }

        const real ComputeSolidEnergy(const Matrix3Xr &position, const NewtonSceneDataCpu &cpu_data)
        {
            // std::cout << "ComputeSolidEnergy begin" << std::endl;
            real energy = 0;
            int tetNum = cpu_data.tetrahedron_num;

#pragma omp parallel for reduction(+ : energy)
            for (int e = 0; e < tetNum; ++e)
            {
                int vidx_0 = cpu_data.tetrahedra_indices(0, e);
                int vidx_1 = cpu_data.tetrahedra_indices(1, e);
                int vidx_2 = cpu_data.tetrahedra_indices(2, e);
                int vidx_3 = cpu_data.tetrahedra_indices(3, e);
                real mu = cpu_data.tetMeshMu[e];
                real la = cpu_data.tetMeshLambda[e];
                Matrix3r DmInvFaInv = cpu_data.tetDmInv[e]; // * cpu_data.tetrahedra_FaInv[e];
                Matrix3r Ds;
                Ds << position.col(vidx_1) - position.col(vidx_0),
                    position.col(vidx_2) - position.col(vidx_0),
                    position.col(vidx_3) - position.col(vidx_0);
                Matrix3r F = Ds * DmInvFaInv;
                energy += ComputeElasticEnergyFromDeformationGradient(F, mu, la) * cpu_data.tetVolume[e];
            }
            // std::cout << "ComputeSolidEnergy end" << std::endl;
            return energy;
        }

        const Matrix3Xr ComputeSolidGradient(const Matrix3Xr &position, const NewtonSceneDataCpu &cpu_data)
        {
            // std::cout << "ComputeSolidGradient begin" << std::endl;
            int tetNum = cpu_data.tetrahedron_num;
            Eigen::Matrix<real, 3, Eigen::Dynamic> gradient = Eigen::Matrix<real, 3, Eigen::Dynamic>::Zero(3, position.cols());
#pragma omp parallel for
            for (int e = 0; e < tetNum; ++e)
            {
                int vidx_0 = cpu_data.tetrahedra_indices(0, e);
                int vidx_1 = cpu_data.tetrahedra_indices(1, e);
                int vidx_2 = cpu_data.tetrahedra_indices(2, e);
                int vidx_3 = cpu_data.tetrahedra_indices(3, e);
                real mu = cpu_data.tetMeshMu[e];
                real la = cpu_data.tetMeshLambda[e];
                // Matrix3r DmInvFaInv = tetDmInv[e] * tetFaInv[e];
                Matrix3r DmInvFaInv = cpu_data.tetDmInv[e];
                Matrix3r Ds;
                Ds << position.col(vidx_1) - position.col(vidx_0),
                    position.col(vidx_2) - position.col(vidx_0),
                    position.col(vidx_3) - position.col(vidx_0);
                Matrix3r F = Ds * DmInvFaInv;
                Matrix3r P = ComputeStressTensorFromDeformationGradient(F, mu, la);
                Matrix12_9 Ate = cpu_data.tetAt[e];
                for (int j = 0; j < 3; j++)
                {
                    for (int i = 0; i < 3; i++)
                    {
                        int idx = j * 3 + i;
                        Matrix3r dFdxdii = Ate.row(idx).reshaped(3, 3); // * cpu_data.tetFaInv[e];
#pragma omp atomic
                        gradient(i, cpu_data.tetrahedra_indices(j, e)) += P.cwiseProduct(dFdxdii).sum() * cpu_data.tetVolume[e];
                    }
                }
            }
            // std::cout << "ComputeSolidGradient end" << std::endl;
            return gradient;
        }

        const std::pair<SparseMatrixXr, SparseMatrixXr> ComputeSolidHessianAndProjection(
            const Matrix3Xr &position,
            const bool compute_hessian, const bool compute_projection,
            const NewtonSceneDataCpu &cpu_data)
        {
            // std::cout << "ComputeSolidHessianAndProjection begin" << std::endl;
            int tetNum = cpu_data.tetrahedron_num;
            int vNum = cpu_data.vertex_num;
            int nonzero_num = tetNum * 3 * 3 * 3 * 3;
            std::vector<Eigen::Triplet<real>> hess_nonzeros(nonzero_num, Eigen::Triplet<real>(0, 0, 0.0));
            std::vector<Eigen::Triplet<real>> hess_proj_nonzeros(nonzero_num, Eigen::Triplet<real>(0, 0, 0.0));
#pragma omp parallel for
            for (int e = 0; e < tetNum; ++e)
            {
                int vidx_0 = cpu_data.tetrahedra_indices(0, e);
                int vidx_1 = cpu_data.tetrahedra_indices(1, e);
                int vidx_2 = cpu_data.tetrahedra_indices(2, e);
                int vidx_3 = cpu_data.tetrahedra_indices(3, e);
                real mu = cpu_data.tetMeshMu[e];
                real la = cpu_data.tetMeshLambda[e];
                Matrix3r DmInvFaInv = cpu_data.tetDmInv[e]; // * tetFaInv[e];
                Matrix3r Ds;
                Ds << position.col(vidx_1) - position.col(vidx_0),
                    position.col(vidx_2) - position.col(vidx_0),
                    position.col(vidx_3) - position.col(vidx_0);
                Matrix3r F = Ds * DmInvFaInv;
                Matrix9r dPdF = ComputeStressTensorDifferentialFromDeformationGradient(F, mu, la);
                Matrix9r dPdF_proj;

                if (compute_projection)
                {
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<real, 9, 9>> eig_solver(dPdF);
                    const VectorXr &la = eig_solver.eigenvalues();
                    const MatrixXr &V = eig_solver.eigenvectors();
                    // V * la.asDiagonal() * V.transpose() = dPdF.
                    dPdF_proj = V * la.cwiseMax(Eigen::Matrix<real, 9, 1>::Zero()).asDiagonal() * V.transpose();
                }

                Matrix12_9 Ate = cpu_data.tetAt[e];
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        for (int di = 0; di < 3; di++)
                        {
                            for (int dj = 0; dj < 3; dj++)
                            {
                                // Compute dFdx(di, i) and dFdx(dj, j).
                                int idx = e * 3 * 3 * 3 * 3 + i * 3 * 3 * 3 + j * 3 * 3 + di * 3 + dj;
                                int row = cpu_data.tetrahedra_indices(i, e) * 3 + di;
                                int col = cpu_data.tetrahedra_indices(j, e) * 3 + dj;
                                Vector9r dFdxdii = (Ate.row(i * 3 + di).reshaped(3, 3)).reshaped(); // * cpu_data.tetFaInv[e];
                                Vector9r dFdxdjj = (Ate.row(j * 3 + dj).reshaped(3, 3)).reshaped(); // * cpu_data.tetFaInv[e];
                                if (compute_hessian)
                                    hess_nonzeros[idx] = Eigen::Triplet<real>(row, col, dFdxdii.dot(dPdF * dFdxdjj) * cpu_data.tetVolume[e]);
                                if (compute_projection)
                                    hess_proj_nonzeros[idx] = Eigen::Triplet<real>(row, col, dFdxdii.dot(dPdF_proj * dFdxdjj) * cpu_data.tetVolume[e]);
                            }
                        }
                    }
                }
            }
            std::pair<SparseMatrixXr, SparseMatrixXr> ret;
            if (compute_hessian)
                ret.first = FromTriplet(3 * vNum, 3 * vNum, hess_nonzeros);
            if (compute_projection)
                ret.second = FromTriplet(3 * vNum, 3 * vNum, hess_proj_nonzeros);
            // std::cout << "ComputeSolidHessianAndProjection end" << std::endl;
            return ret;
        }

        const real ComputeObjectiveEnergy(const Matrix3Xr &position, const real time_step, const NewtonSceneDataCpu &cpu_data)
        {
            const Matrix3Xr &x_next = position;
            const real h = time_step;
            const real inv_h = real(1) / h;
            // x0 refers to the position of the current time step.
            const Matrix3Xr x0 = cpu_data.position;
            const Matrix3Xr v0 = cpu_data.velocity;
            const Matrix3Xr a = cpu_data.external_acceleration;
            const Matrix3Xr y = x0 + v0 * h + a * h * h;
            const real half_inv_h2 = inv_h * inv_h / 2;

            // The kinematic energy.
            real energy_kinematic = 0;
            for (int d = 0; d < 3; ++d)
            {
                const VectorXr x_next_d(x_next.row(d));
                const VectorXr y_d(y.row(d));
                const VectorXr diff_d = x_next_d - y_d;
                energy_kinematic += diff_d.dot(cpu_data.mass_matrix * diff_d);
            }
            energy_kinematic *= half_inv_h2;

            // The elastic energy.
            real energy_solid = ComputeSolidEnergy(x_next, cpu_data);

            real energy = energy_kinematic + energy_solid;

            // debug.

            LOG_TRACE("kinematic energy: {}", energy_kinematic);
            LOG_TRACE("solid energy: {}", energy_solid);
            LOG_TRACE("total energy: {}", energy);

            return energy;
        }

        const Matrix3Xr ComputeObjectiveGradient(
            const Matrix3Xr &position, const real time_step, const NewtonSceneDataCpu &cpu_data)
        {
            // std::cout << "ComputeObjectiveGradient begin" << std::endl;
            const Matrix3Xr &x_next = position;
            const real h = time_step;
            const real inv_h = real(1) / h;
            // x0 refers to the position of the current time step.
            const Matrix3Xr x0 = cpu_data.position;
            const Matrix3Xr v0 = cpu_data.velocity;
            const Matrix3Xr a = cpu_data.external_acceleration;
            const Matrix3Xr y = x0 + v0 * h + a * h * h;

            // The kinematic force.
            Matrix3Xr gradient_kinematic =
                Matrix3Xr::Zero(3, x_next.cols());
            for (int d = 0; d < 3; ++d)
            {
                const VectorXr x_next_d(x_next.row(d));
                const VectorXr y_d(y.row(d));
                const VectorXr diff_d = x_next_d - y_d;
                gradient_kinematic.row(d) += RowVectorXr(cpu_data.mass_matrix * diff_d);
            }
            gradient_kinematic *= inv_h * inv_h;

            // The elastic force.
            Matrix3Xr gradient_elastic = Matrix3Xr::Zero(3, x_next.cols());
            gradient_elastic += ComputeSolidGradient(x_next, cpu_data);

            Matrix3Xr gradient = gradient_kinematic + gradient_elastic;
            // std::cout << "ComputeObjectiveGradient end" << std::endl;
            return gradient;
        }

        const std::pair<SparseMatrixXr, SparseMatrixXr> ComputeObjectiveHessianAndProjection(
            const Matrix3Xr &position, const real time_step,
            const bool compute_hessian, const bool compute_projection,
            const NewtonSceneDataCpu &cpu_data)
        {
            // std::cout << "ComputeObjectiveHessianAndProjection begin" << std::endl;
            const Matrix3Xr &x_next = position;
            const real h = time_step;
            const real inv_h = real(1) / h;

            // Kinematic Hessian.
            std::vector<Eigen::Triplet<real>> kinematic_nonzeros;
            std::vector<Eigen::Triplet<real>> int_nonzeros = ToTriplet(cpu_data.mass_matrix);
            const real scale = inv_h * inv_h;
            for (const auto &triplet : int_nonzeros)
            {
                for (int d = 0; d < 3; ++d)
                {
                    kinematic_nonzeros.push_back(Eigen::Triplet<real>(
                        triplet.row() * 3 + d,
                        triplet.col() * 3 + d,
                        triplet.value() * scale));
                }
            }
            const SparseMatrixXr H_kinematics = FromTriplet(3 * cpu_data.vertex_num, 3 * cpu_data.vertex_num, kinematic_nonzeros);

            std::pair<SparseMatrixXr, SparseMatrixXr> H;
            if (compute_hessian)
                H.first = H_kinematics;
            if (compute_projection)
                H.second = H_kinematics;

            // Elastic Hessian.
            const std::pair<SparseMatrixXr, SparseMatrixXr> H_solid = ComputeSolidHessianAndProjection(x_next, compute_hessian, compute_projection, cpu_data);
            if (compute_hessian)
                H.first += H_solid.first;
            if (compute_projection)
                H.second += H_solid.second;

            // std::cout << "ComputeObjectiveHessianAndProjection end" << std::endl;
            return H;
        }

        const real ComputeObjectiveEnergyStable(const Matrix3Xr &position, const Matrix3Xr &position_lbs, const NewtonSceneDataCpu &cpu_data, const std::vector<real> &stiffness)
        {

            // The elastic energy.
            real energy_solid = ComputeSolidEnergy(position, cpu_data);

            real energy_lbs = ComputeLBSEnergy(position, position_lbs, stiffness);
            real energy = energy_solid + energy_lbs;

            // debug.
            LOG_TRACE("solid energy: {}", energy_solid);
            LOG_TRACE("lbs energy: {}", energy_lbs);
            LOG_TRACE("total energy: {}", energy);
            return energy;
        }

        const Matrix3Xr ComputeObjectiveGradientStable(
            const Matrix3Xr &position, const Matrix3Xr &position_lbs, const NewtonSceneDataCpu &cpu_data, const std::vector<real> &stiffness)
        {

            // The elastic force.
            Matrix3Xr gradient_elastic = Matrix3Xr::Zero(3, position.cols());
            gradient_elastic += ComputeSolidGradient(position, cpu_data);

            gradient_elastic += ComputeLBSGradient(position, position_lbs, stiffness);

            Matrix3Xr gradient = gradient_elastic;
            return gradient;
        }

        const std::pair<SparseMatrixXr, SparseMatrixXr> ComputeObjectiveHessianAndProjectionStable(
            const Matrix3Xr &position, const NewtonSceneDataCpu &cpu_data, const std::vector<real> &stiffness,
            const bool compute_hessian, const bool compute_projection)
        {
            // Kinematic Hessian.
            std::vector<Eigen::Triplet<real>> kinematic_nonzeros;
            const SparseMatrixXr H_kinematics = FromTriplet(3 * cpu_data.vertex_num, 3 * cpu_data.vertex_num, kinematic_nonzeros);

            std::pair<SparseMatrixXr, SparseMatrixXr> H;
            if (compute_hessian)
                H.first = H_kinematics;
            if (compute_projection)
                H.second = H_kinematics;

            // Elastic Hessian.
            const std::pair<SparseMatrixXr, SparseMatrixXr> H_solid = ComputeSolidHessianAndProjection(position, compute_hessian, compute_projection, cpu_data);
            if (compute_hessian)
                H.first += H_solid.first;
            if (compute_projection)
                H.second += H_solid.second;

            const std::pair<SparseMatrixXr, SparseMatrixXr> H_lbs = ComputeLBSHessianAndProjection(position, stiffness, compute_hessian, compute_projection);
            if (compute_hessian)
                H.first += H_lbs.first;
            if (compute_projection)
                H.second += H_lbs.second;
            // std::cout << "ComputeObjectiveHessianAndProjection end" << std::endl;
            return H;
        }

        void solveNewtonFEMStep(NewtonSceneDataCpu &cpu_data, real dt, const FemSolverOptions &options)
        {
            const Matrix3Xr &x0 = cpu_data.position;
            const real h = dt;
            const real inv_h = real(1) / h;
            const int vNum = cpu_data.vertex_num;

            // Functions needed by Newton.
            const std::function<const real(const VectorXr &)> E = [&](const VectorXr &x_next)
            {
                real energy = ComputeObjectiveEnergy(x_next.reshaped(3, vNum), h, cpu_data);
                return energy;
            };
            // Its gradient.
            const std::function<const VectorXr(const VectorXr &)> grad_E = [&](const VectorXr &x_next)
            {
                const Matrix3Xr g = ComputeObjectiveGradient(x_next.reshaped(3, vNum), h, cpu_data);
                return g.reshaped();
            };
            // The Hessian matrix.
            const std::function<const std::pair<SparseMatrixXr, SparseMatrixXr>(
                const VectorXr &, const bool, const bool)>
                Hess_E_and_proj = [&](const VectorXr &x_next, const bool compute_hessian, const bool compute_projection)
            {
                const auto ret = ComputeObjectiveHessianAndProjection(x_next.reshaped(3, vNum), h, compute_hessian, compute_projection, cpu_data);
                return ret;
            };
            // For gradient and Hessian check.
            const std::function<const SparseMatrixXr(const VectorXr &)> Hess_E = [&](const VectorXr &x_next)
            {
                const auto ret = ComputeObjectiveHessianAndProjection(x_next.reshaped(3, vNum), h, true, false, cpu_data);
                return ret.first;
            };

            // Check gradient and Hessian.
            int grad_check = options.grad_check;
            if (grad_check > 0)
            {
                VectorXr random_vec = VectorXr::Random(3 * vNum) * 0.1;
                VectorXr x_test = x0.reshaped() + random_vec;
                checkGradient(E, grad_E, x_test);
                checkHessian(grad_E, Hess_E, x_test);
            }

            int thread_ct = options.thread_ct;
            omp_set_num_threads(thread_ct);

            // The line search step size.
            const std::function<const real(const VectorXr &, const VectorXr &)> max_step_size = [&](const VectorXr &xk, const VectorXr &pk)
            {
                real ss = 1.;
                return ss;
            };
            // The convergence condition.
            const real force_density_abs_tol = options.force_density_abs_tol;
            // It is fair to assume every entry in vol_int_matrix is positive (not always true though, but since the semantical meaning of vol_int_matrix
            // is the integral of weight functions between [0, 1], it seems fair to assume so).
            // Therefore, we can require that |gk|_i <= |vol_int_matrix * 1 * force_density_abs_tol|.
            // Or something like |gk|_max <= |vol_int_matrix * 1 * force_density_abs_tol|_max.
            const real gk_abs_tol = (cpu_data.mass_matrix * VectorXr::Constant(cpu_data.mass_matrix.cols(), force_density_abs_tol)).cwiseAbs().maxCoeff() * 10.;
            const std::function<const bool(const VectorXr &)> converged = [&](const VectorXr &gk)
            {
                return gk.cwiseAbs().maxCoeff() <= gk_abs_tol;
            };
            const std::function<const bool(const VectorXr &, const VectorXr &)> converged_var = [&](const VectorXr &xk, const VectorXr &xk_last)
            {
                return (xk - xk_last).cwiseAbs().maxCoeff() <= 1e-5;
            };

            optimizers::NewtonOptimizer<SparseMatrixXr> newton(E, grad_E, Hess_E_and_proj, max_step_size);
            // Optimize() will throw out errors if it fails.
            // Note that the initial guess must take care of boundary conditions.
            const optimizers::OptimizerReturnData result = newton.Optimize(x0.reshaped(), converged, converged_var, options);
            // Update.
            cpu_data.next_position = result.solution.reshaped(3, vNum);
            cpu_data.next_velocity = (cpu_data.next_position - x0) * inv_h;
            cpu_data.position = cpu_data.next_position;
            cpu_data.velocity = cpu_data.next_velocity;
        }

        void solveNewtonDynamicCorrectionStep(NewtonSceneDataCpu &cpu_data, control::LBSDataGpu3D &lbs_control_data, const FemSolverOptions &options)
        {
            PROFILE_FUNCTION();
            auto x_rest = lbs_control_data.position_rest.download();
            auto x_control = lbs_control_data.position_lbs.download();
            auto stiffness = lbs_control_data.stiffness.download();
            Matrix3Xr x0, x_lbs;
            copyGlmToEigen(x_rest, x0);
            copyGlmToEigen(x_control, x_lbs);

            const int vNum = cpu_data.vertex_num;

            // Functions needed by Newton.
            const std::function<const real(const VectorXr &)> E = [&](const VectorXr &x_next)
            {
                real energy = ComputeObjectiveEnergyStable(x_next.reshaped(3, vNum), x_lbs, cpu_data, stiffness);
                return energy;
            };
            // Its gradient.
            const std::function<const VectorXr(const VectorXr &)> grad_E = [&](const VectorXr &x_next)
            {
                const Matrix3Xr g = ComputeObjectiveGradientStable(x_next.reshaped(3, vNum), x_lbs, cpu_data, stiffness);
                return g.reshaped();
            };
            // The Hessian matrix.
            const std::function<const std::pair<SparseMatrixXr, SparseMatrixXr>(
                const VectorXr &, const bool, const bool)>
                Hess_E_and_proj = [&](const VectorXr &x_next, const bool compute_hessian, const bool compute_projection)
            {
                const auto ret = ComputeObjectiveHessianAndProjectionStable(x_next.reshaped(3, vNum), cpu_data, stiffness, compute_hessian, compute_projection);
                return ret;
            };
            // For gradient and Hessian check.
            const std::function<const SparseMatrixXr(const VectorXr &)> Hess_E = [&](const VectorXr &x_next)
            {
                const auto ret = ComputeObjectiveHessianAndProjectionStable(x_next.reshaped(3, vNum), cpu_data, stiffness, true, false);
                return ret.first;
            };

            // Check gradient and Hessian.
            int grad_check = options.grad_check;
            if (grad_check > 0)
            {
                VectorXr random_vec = VectorXr::Random(3 * vNum) * 0.1;
                VectorXr x_test = x0.reshaped() + random_vec;
                checkGradient(E, grad_E, x_test);
                checkHessian(grad_E, Hess_E, x_test);
            }

            int thread_ct = options.thread_ct;
            omp_set_num_threads(thread_ct);

            // The line search step size.
            const std::function<const real(const VectorXr &, const VectorXr &)> max_step_size = [&](const VectorXr &xk, const VectorXr &pk)
            {
                real ss = 1.;
                return ss;
            };
            // The convergence condition.
            const real force_density_abs_tol = options.force_density_abs_tol;
            // It is fair to assume every entry in vol_int_matrix is positive (not always true though, but since the semantical meaning of vol_int_matrix
            // is the integral of weight functions between [0, 1], it seems fair to assume so).
            // Therefore, we can require that |gk|_i <= |vol_int_matrix * 1 * force_density_abs_tol|.
            // Or something like |gk|_max <= |vol_int_matrix * 1 * force_density_abs_tol|_max.
            const real gk_abs_tol = (cpu_data.mass_matrix * VectorXr::Constant(cpu_data.mass_matrix.cols(), force_density_abs_tol)).cwiseAbs().maxCoeff() * 10.;
            const std::function<const bool(const VectorXr &)> converged = [&](const VectorXr &gk)
            {
                return gk.cwiseAbs().maxCoeff() <= gk_abs_tol;
            };
            const std::function<const bool(const VectorXr &, const VectorXr &)> converged_var = [&](const VectorXr &xk, const VectorXr &xk_last)
            {
                return (xk - xk_last).cwiseAbs().maxCoeff() <= 1e-5;
            };

            optimizers::NewtonOptimizer<SparseMatrixXr> newton(E, grad_E, Hess_E_and_proj, max_step_size);
            // Optimize() will throw out errors if it fails.
            // Note that the initial guess must take care of boundary conditions.
            const optimizers::OptimizerReturnData result = newton.Optimize(x0.reshaped(), converged, converged_var, options);
            // Update.

            std::vector<vec3_t> x_lbs_glm;
            Matrix3Xr x_target_eigen = result.solution.reshaped(3, vNum);
            copyEigenToGlm(x_target_eigen, x_lbs_glm);
            lbs_control_data.position_target.upload(x_lbs_glm);
        }

        void solveNewtonDynamicCorrectionStepGpu(NewtonSceneDataCpu &cpu_data, VbdSceneDataGpu3D &gpu_data, control::LBSDataGpu3D &lbs_control_data, const FemSolverOptions &options, cudaStream_t stream)
        {
            PROFILE_FUNCTION();
            auto x_rest = lbs_control_data.position_rest.download();
            Matrix3Xr x0, x_lbs;
            copyGlmToEigen(x_rest, x0);

            const int vNum = gpu_data.vertex_num;

            // Functions needed by Newton.
            const std::function<const real(const VectorXr &)> E = [&](const VectorXr &x_next)
            {
                real energy = ComputeElementElasticEnergyGpu(x_next.reshaped(3, vNum), gpu_data, lbs_control_data, stream);
                // real energy_cpu = ComputeObjectiveEnergyStable(x_next.reshaped(3, vNum), x_lbs, cpu_data, stiffness);
                // printf("elastic energy: %f, cpu: %f\n", energy, energy_cpu);
                return energy;
            };
            // Its gradient.
            const std::function<const VectorXr(const VectorXr &)> grad_E = [&](const VectorXr &x_next)
            {
                const Matrix3Xr g = ComputeElasticForceGpu(x_next.reshaped(3, vNum), gpu_data, lbs_control_data, stream);
                // const Matrix3Xr g_cpu = ComputeObjectiveGradientStable(x_next.reshaped(3, vNum), x_lbs, cpu_data, stiffness);
                // printf("elastic force diff norm: %f\n", (g-g_cpu).norm());
                return VectorXr(g.reshaped());
            };
            // The Hessian matrix.
            const std::function<const std::pair<SparseMatrixXr, SparseMatrixXr>(
                const VectorXr &, const bool, const bool)>
                Hess_E_and_proj = [&](const VectorXr &x_next, const bool compute_hessian, const bool compute_projection)
            {
                const auto ret = ComputeElasticHessianAndProjectionGpu(x_next.reshaped(3, vNum), gpu_data, lbs_control_data, compute_hessian, compute_projection, stream);
                return ret;
            };
            // For gradient and Hessian check.
            const std::function<const SparseMatrixXr(const VectorXr &)> Hess_E = [&](const VectorXr &x_next)
            {
                const auto ret = ComputeElasticHessianAndProjectionGpu(x_next.reshaped(3, vNum), gpu_data, lbs_control_data, true, false, stream);
                return ret.first;
            };

            // Check gradient and Hessian.
            if (options.grad_check)
            {
                VectorXr random_vec = VectorXr::Random(3 * vNum) * 0.01;
                VectorXr x_test = x0.reshaped() + random_vec;
                checkGradient(E, grad_E, x_test);
                checkHessian(grad_E, Hess_E, x_test);
            }

            int thread_ct = options.thread_ct;
            omp_set_num_threads(thread_ct);

            // The line search step size.
            const std::function<const real(const VectorXr &, const VectorXr &)> max_step_size = [&](const VectorXr &xk, const VectorXr &pk)
            {
                real ss = 1.;
                return ss;
            };
            // The convergence condition.
            const real force_density_abs_tol = options.force_density_abs_tol;
            // It is fair to assume every entry in vol_int_matrix is positive (not always true though, but since the semantical meaning of vol_int_matrix
            // is the integral of weight functions between [0, 1], it seems fair to assume so).
            // Therefore, we can require that |gk|_i <= |vol_int_matrix * 1 * force_density_abs_tol|.
            // Or something like |gk|_max <= |vol_int_matrix * 1 * force_density_abs_tol|_max.
            const real gk_abs_tol = (cpu_data.mass_matrix * VectorXr::Constant(cpu_data.mass_matrix.cols(), force_density_abs_tol)).cwiseAbs().maxCoeff() * 10.;
            const std::function<const bool(const VectorXr &)> converged = [&](const VectorXr &gk)
            {
                return gk.cwiseAbs().maxCoeff() <= gk_abs_tol;
            };
            const std::function<const bool(const VectorXr &, const VectorXr &)> converged_var = [&](const VectorXr &xk, const VectorXr &xk_last)
            {
                return (xk - xk_last).cwiseAbs().maxCoeff() <= 1e-5;
            };

            optimizers::NewtonOptimizer<SparseMatrixXr> newton(E, grad_E, Hess_E_and_proj, max_step_size);
            // Optimize() will throw out errors if it fails.
            // Note that the initial guess must take care of boundary conditions.
            const optimizers::OptimizerReturnData result = newton.Optimize(x0.reshaped(), converged, converged_var, options);
            // Update.

            std::vector<vec3_t> x_lbs_glm;
            Matrix3Xr x_target_eigen = result.solution.reshaped(3, vNum);
            copyEigenToGlm(x_target_eigen, x_lbs_glm);
            lbs_control_data.position_target.upload(x_lbs_glm);
        }

        // 2D version

        void solveNewtonDynamicCorrectionStepGpu(NewtonSceneDataCpu &cpu_data, VbdSceneDataGpu2D &gpu_data, control::LBSDataGpu2D &lbs_control_data, const FemSolverOptions &options, cudaStream_t stream)
        {
            PROFILE_FUNCTION();
            auto x_rest = lbs_control_data.position_rest.download();
            auto stiffness = lbs_control_data.stiffness.download();
            Matrix2Xr x0, x_lbs;
            copyGlmToEigen(x_rest, x0);

            const int vNum = gpu_data.vertex_num;

            // Functions needed by Newton.
            const std::function<const real(const VectorXr &)> E = [&](const VectorXr &x_next)
            {
                real energy = ComputeElementElasticEnergyGpu(x_next.reshaped(2, vNum), gpu_data, lbs_control_data, stream);
                return energy;
            };
            // Its gradient.
            const std::function<const VectorXr(const VectorXr &)> grad_E = [&](const VectorXr &x_next)
            {
                const Matrix2Xr g = ComputeElasticForceGpu(x_next.reshaped(2, vNum), gpu_data, lbs_control_data, stream);
                return VectorXr(g.reshaped());
            };
            // The Hessian matrix.
            const std::function<const std::pair<SparseMatrixXr, SparseMatrixXr>(
                const VectorXr &, const bool, const bool)>
                Hess_E_and_proj = [&](const VectorXr &x_next, const bool compute_hessian, const bool compute_projection)
            {
                const auto ret = ComputeElasticHessianAndProjectionGpu(x_next.reshaped(2, vNum), gpu_data, lbs_control_data, compute_hessian, compute_projection, stream);
                return ret;
            };
            // For gradient and Hessian check.
            const std::function<const SparseMatrixXr(const VectorXr &)> Hess_E = [&](const VectorXr &x_next)
            {
                const auto ret = ComputeElasticHessianAndProjectionGpu(x_next.reshaped(2, vNum), gpu_data, lbs_control_data, true, false, stream);
                return ret.first;
            };

            // Check gradient and Hessian.
            if (options.grad_check)
            {
                VectorXr random_vec = VectorXr::Random(2 * vNum) * 0.1;
                VectorXr x_test = x0.reshaped() + random_vec;
                checkGradient(E, grad_E, x_test);
                checkHessian(grad_E, Hess_E, x_test);
            }

            int thread_ct = options.thread_ct;
            omp_set_num_threads(thread_ct);

            // The line search step size.
            const std::function<const real(const VectorXr &, const VectorXr &)> max_step_size = [&](const VectorXr &xk, const VectorXr &pk)
            {
                real ss = 1.;
                return ss;
            };
            // The convergence condition.
            const real force_density_abs_tol = options.force_density_abs_tol;
            // It is fair to assume every entry in vol_int_matrix is positive (not always true though, but since the semantical meaning of vol_int_matrix
            // is the integral of weight functions between [0, 1], it seems fair to assume so).
            // Therefore, we can require that |gk|_i <= |vol_int_matrix * 1 * force_density_abs_tol|.
            // Or something like |gk|_max <= |vol_int_matrix * 1 * force_density_abs_tol|_max.
            const real gk_abs_tol = (cpu_data.mass_matrix * VectorXr::Constant(cpu_data.mass_matrix.cols(), force_density_abs_tol)).cwiseAbs().maxCoeff() * 10.;
            const std::function<const bool(const VectorXr &)> converged = [&](const VectorXr &gk)
            {
                return gk.cwiseAbs().mean() <= gk_abs_tol;
            };
            const std::function<const bool(const VectorXr &, const VectorXr &)> converged_var = [&](const VectorXr &xk, const VectorXr &xk_last)
            {
                return (xk - xk_last).cwiseAbs().maxCoeff() <= 1e-5;
            };

            optimizers::NewtonOptimizer<SparseMatrixXr> newton(E, grad_E, Hess_E_and_proj, max_step_size);
            // Optimize() will throw out errors if it fails.
            // Note that the initial guess must take care of boundary conditions.
            const optimizers::OptimizerReturnData result = newton.Optimize(x0.reshaped(), converged, converged_var, options);
            // Update.

            if (result.success)
            {
                std::vector<vec2_t> x_lbs_glm;
                Matrix2Xr x_target_eigen = result.solution.reshaped(2, vNum);
                copyEigenToGlm(x_target_eigen, x_lbs_glm);
                lbs_control_data.position_target.upload(x_lbs_glm);
            }
        }

    } // namespace fem
} // namespace fsi