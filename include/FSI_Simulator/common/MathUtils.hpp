// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/common/Types.hpp"
#include <functional>

namespace fsi
{

    Matrix3r crossProductMatrix(const Vector3r &a);
    Matrix2r determinantGradient(const Matrix2r &A);
    Matrix3r determinantGradient(const Matrix3r &A);
    Matrix4r determinantHessian(const Matrix2r &A);
    Matrix9r determinantHessian(const Matrix3r &A);
    MatrixXr projectToSpd(const MatrixXr &A);

    // Numerical differentiation for checking gradient and Hessian
    void checkGradient(
        const std::function<const real(const VectorXr &)> &f,
        const std::function<const VectorXr(const VectorXr &)> &g,
        const VectorXr &x);

    void checkHessian(
        const std::function<const VectorXr(const VectorXr &)> &f,
        const std::function<const SparseMatrixXr(const VectorXr &)> &g,
        const VectorXr &x);

    void checkHessian(
        const std::function<const VectorXr(const VectorXr &)> &f,
        const std::function<const MatrixXr(const VectorXr &)> &g,
        const VectorXr &x);

} // namespace fsi