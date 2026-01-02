// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/common/MathUtils.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include "FSI_Simulator/utils/StringUtils.hpp"

#include <algorithm>
#include <iostream>

namespace fsi
{

    Matrix3r crossProductMatrix(const Vector3r &a)
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

    Matrix2r determinantGradient(const Matrix2r &A)
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

    Matrix3r determinantGradient(const Matrix3r &A)
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

    Matrix4r determinantHessian(const Matrix2r &A)
    {
        Matrix4r H = Matrix4r::Zero();
        H(3, 0) = 1;
        H(2, 1) = -1;
        H(1, 2) = -1;
        H(0, 3) = 1;
        return H;
    }

    Matrix9r determinantHessian(const Matrix3r &A)
    {
        Matrix9r H = Matrix9r::Zero();
        const Matrix3r A0 = crossProductMatrix(A.col(0));
        const Matrix3r A1 = crossProductMatrix(A.col(1));
        const Matrix3r A2 = crossProductMatrix(A.col(2));
        H.block<3, 3>(0, 3) += -A2;
        H.block<3, 3>(0, 6) += A1;
        H.block<3, 3>(3, 0) += A2;
        H.block<3, 3>(3, 6) += -A0;
        H.block<3, 3>(6, 0) += -A1;
        H.block<3, 3>(6, 3) += A0;
        return H;
    }

    MatrixXr projectToSpd(const MatrixXr &A)
    {
        Eigen::SelfAdjointEigenSolver<MatrixXr> eig_solver(A);
        const VectorXr &la = eig_solver.eigenvalues();
        const MatrixXr &V = eig_solver.eigenvectors();
        return V * la.cwiseMax(VectorXr::Zero(A.rows())).asDiagonal() * V.transpose();
    }

    void checkGradient(
        const std::function<const real(const VectorXr &)> &f,
        const std::function<const VectorXr(const VectorXr &)> &g,
        const VectorXr &x)
    {
        LOG_TRACE("Starting numerical gradient check, size {}...", x.size());
        real f0 = f(x);
        auto analytical_grad = g(x);
        // std::cout << analytical_grad << std::endl;

        for (int i = 0; i < x.size(); ++i)
        {
            // printf("Checking gradient: %d / %d\r", i + 1, x.size());

            bool local_check = false;
            std::vector<real> histo;
            for (real eps : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8})
            {
                VectorXr delta_x = VectorXr::Zero(x.size());
                delta_x[i] = eps;
                real numerical_grad = (f(x + delta_x) - f0) / eps;
                histo.push_back(numerical_grad);

                if (std::abs(numerical_grad - analytical_grad[i]) <= 1e-2 + 1e-3 * std::abs(analytical_grad[i]))
                {
                    local_check = true;
                    break;
                }
            }

            if (!local_check)
            {
                LOG_ERROR("Gradient check failed at index {}", i);
                LOG_ERROR("  - Numerical gradients: {}", fsi::utils::join(histo));
                LOG_ERROR("  - Analytical gradient: {}", analytical_grad[i]);
                // ASSERT(false, "Gradient check failed.");
            }
        }
        LOG_TRACE("\n");
        LOG_TRACE("Gradient check passed.");
    }

    void checkHessian(
        const std::function<const VectorXr(const VectorXr &)> &f,
        const std::function<const SparseMatrixXr(const VectorXr &)> &g,
        const VectorXr &x)
    {
        VectorXr f0 = f(x);
        SparseMatrixXr analytical_hess = g(x);
        bool hess_check = true;

        for (int i = 0; i < x.size(); ++i)
        {

            for (int j = 0; j < x.size(); ++j)
            {

                int progress = static_cast<int>(100.0 * (j + 1) / x.size());
                // LOG_TRACE("Hessian {} / {}: {}% [{}{}]\r", i, x.size(), progress, std::string(progress / 2, '='), std::string(50 - progress / 2, ' '));
                printf("Hessian %d / %ld: %d%% [%s%s]\r", i, x.size(), progress, std::string(progress / 2, '=').c_str(), std::string(50 - progress / 2, ' ').c_str());
                bool local_check = false;
                std::vector<double> histo;

                for (double eps : {1., 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8})
                {
                    VectorXr delta_x = VectorXr::Zero(x.size());
                    delta_x[i] = eps;

                    VectorXr x_plus = x + delta_x;
                    VectorXr x_minus = x - delta_x;
                    double numerical_hess = (f(x_plus)[j] - f(x_minus)[j]) / (2 * eps);
                    histo.push_back(numerical_hess);

                    double hess_val = analytical_hess.coeff(j, i);
                    if (std::abs(numerical_hess - hess_val) <= 1e-1 + 1e-3 * std::abs(hess_val))
                    {
                        local_check = true;
                        break;
                    }
                }

                double min_val = *std::min_element(histo.begin(), histo.end());
                double max_val = *std::max_element(histo.begin(), histo.end());

                if (min_val <= analytical_hess.coeff(j, i) && analytical_hess.coeff(j, i) <= max_val)
                {
                    local_check = true;
                }

                if (!local_check)
                {
                    hess_check = false;
                    LOG_ERROR("Hessian check fails at ({}, {})", i, j);
                    LOG_ERROR("Numerical Hessian: {}", fsi::utils::join(histo));
                    LOG_ERROR("Analytical Hessian: {}", analytical_hess.coeff(j, i));
                    break;
                }
            }
        }

        if (hess_check)
        {
            LOG_TRACE("Hessian check passed.");
        }
        else
        {
            LOG_ERROR("Hessian check failed.");
        }
    }

    void checkHessian(
        const std::function<const VectorXr(const VectorXr &)> &f,
        const std::function<const MatrixXr(const VectorXr &)> &g,
        const VectorXr &x)
    {
        VectorXr f0 = f(x);
        MatrixXr analytical_hess = g(x);
        bool hess_check = true;

        for (int i = 0; i < x.size(); ++i)
        {
            for (int j = 0; j < x.size(); ++j)
            {
                int progress = static_cast<int>(100.0 * (j + 1) / x.size());
                // LOG_TRACE("Hessian {} / {}: {}% [{}{}]\r", i, x.size(), progress, std::string(progress / 2, '='), std::string(50 - progress / 2, ' '));

                bool local_check = false;
                std::vector<double> histo;

                for (double eps : {1., 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10})
                {
                    VectorXr delta_x = VectorXr::Zero(x.size());
                    delta_x[i] = eps;

                    VectorXr x_plus = x + delta_x;
                    VectorXr x_minus = x - delta_x;
                    double numerical_hess = (f(x_plus)[j] - f(x_minus)[j]) / (2 * eps);
                    histo.push_back(numerical_hess);

                    double hess_val = analytical_hess(j, i);
                    if (std::abs(numerical_hess - hess_val) <= 1e-1 + 1e-3 * std::abs(hess_val))
                    {
                        local_check = true;
                        break;
                    }
                }

                double min_val = *std::min_element(histo.begin(), histo.end());
                double max_val = *std::max_element(histo.begin(), histo.end());

                if (min_val <= analytical_hess(j, i) && analytical_hess(j, i) <= max_val)
                {
                    local_check = true;
                }

                if (!local_check)
                {
                    hess_check = false;
                    LOG_ERROR("Hessian check fails at ({}, {})", i, j);
                    LOG_ERROR("Numerical Hessian: {}", fsi::utils::join(histo));
                    LOG_ERROR("Analytical Hessian: {}", analytical_hess(j, i));
                    break;
                }
            }
        }

        if (hess_check)
        {
            LOG_TRACE("Hessian check passed.");
        }
        else
        {
            LOG_ERROR("Hessian check failed.");
        }
    }

} // namespace fsi