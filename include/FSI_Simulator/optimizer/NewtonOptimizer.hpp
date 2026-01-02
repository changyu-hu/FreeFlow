// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/common/MathUtils.hpp"
#include "FSI_Simulator/fem/FemConfig.hpp"
#include "FSI_Simulator/optimizer/LineSearch.hpp"

namespace fsi
{
    namespace optimizers
    {

        struct OptimizerReturnData
        {
        public:
            VectorXr solution;
            double obj;
            VectorXr grad;
            bool success;
            fem::FemSolverOptions info;
        };

        template <class MatrixType>
        class NewtonOptimizer;

        template <>
        class NewtonOptimizer<MatrixXr>
        {
        public:
            // The functional max_step_size computes the upper bound of the step size used by the line search
            // algorithm. It is usually 1 in Newton's method. However, in simulation, we may need to crop this
            // value due to flipped finite elements, collisions with obstacles, and so on. Therefore, we provide
            // this function to make this a flexible, user-controlled value. The two inputs are xk and pk.
            NewtonOptimizer(const std::function<const double(const VectorXr &)> &obj,
                            const std::function<const VectorXr(const VectorXr &)> &grad,
                            const std::function<const MatrixXr(const VectorXr &)> &hess,
                            const std::function<const double(const VectorXr &, const VectorXr &)> &max_step_size);
            ~NewtonOptimizer() {}

            // Convergence is defined on grad(x_k), (x_k, x_{k - 1}), or (f_k, f_{k - 1}).
            // If any of them indicates convergence, we terminate.
            const OptimizerReturnData Optimize(const VectorXr &initial_guess,
                                               const std::function<const bool(const VectorXr &)> &converge_grad,
                                               const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var,
                                               const std::function<const bool(const double, const double)> &converge_obj,
                                               const fem::FemSolverOptions &opt) const;

            const OptimizerReturnData Optimize(const VectorXr &initial_guess,
                                               const std::function<const bool(const VectorXr &)> &converge_grad,
                                               const fem::FemSolverOptions &opt) const;

            const OptimizerReturnData Optimize(const VectorXr &initial_guess,
                                               const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var,
                                               const fem::FemSolverOptions &opt) const;

            const OptimizerReturnData Optimize(const VectorXr &initial_guess,
                                               const std::function<const bool(const double, const double)> &converge_obj,
                                               const fem::FemSolverOptions &opt) const;

            static const std::function<const bool(const VectorXr &)> GetDummyGradientConvergenceFunction();
            static const std::function<const bool(const VectorXr &, const VectorXr &)> GetDummyVariableConvergenceFunction();
            static const std::function<const bool(const double, const double)> GetDummyObjectiveConvergenceFunction();

        private:
            const std::shared_ptr<LineSearch> CreateLineSearch(const std::string &method) const;

            const std::function<const double(const VectorXr &)> &obj_;
            const std::function<const VectorXr(const VectorXr &)> &grad_;
            const std::function<const MatrixXr(const VectorXr &)> &hess_;

            const std::function<const double(const VectorXr &, const VectorXr &)> &max_step_size_;
        };

        template <>
        class NewtonOptimizer<SparseMatrixXr>
        {
        public:
            NewtonOptimizer(const std::function<const double(const VectorXr &)> &obj,
                            const std::function<const VectorXr(const VectorXr &)> &grad,
                            const std::function<const std::pair<SparseMatrixXr, SparseMatrixXr>(
                                const VectorXr &, const bool, const bool)> &hess_and_proj,
                            const std::function<const double(const VectorXr &, const VectorXr &)> &max_step_size);
            ~NewtonOptimizer() {}

            // Convergence is defined on grad(x_k), (x_k, x_{k - 1}), or (f_k, f_{k - 1}).
            // If any of them indicates convergence, we terminate.
            const OptimizerReturnData Optimize(const VectorXr &initial_guess,
                                               const std::function<const bool(const VectorXr &)> &converge_grad,
                                               const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var,
                                               const std::function<const bool(const double, const double)> &converge_obj,
                                               const fem::FemSolverOptions &opt) const;

            const OptimizerReturnData Optimize(const VectorXr &initial_guess,
                                               const std::function<const bool(const VectorXr &)> &converge_grad,
                                               const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var,
                                               const fem::FemSolverOptions &opt) const;

            const OptimizerReturnData Optimize(const VectorXr &initial_guess,
                                               const std::function<const bool(const VectorXr &)> &converge_grad,
                                               const fem::FemSolverOptions &opt) const;

            const OptimizerReturnData Optimize(const VectorXr &initial_guess,
                                               const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var,
                                               const fem::FemSolverOptions &opt) const;

            const OptimizerReturnData Optimize(const VectorXr &initial_guess,
                                               const std::function<const bool(const double, const double)> &converge_obj,
                                               const fem::FemSolverOptions &opt) const;

            static const std::function<const bool(const VectorXr &)> GetDummyGradientConvergenceFunction();
            static const std::function<const bool(const VectorXr &, const VectorXr &)> GetDummyVariableConvergenceFunction();
            static const std::function<const bool(const double, const double)> GetDummyObjectiveConvergenceFunction();

        private:
            const std::shared_ptr<LineSearch> CreateLineSearch(const std::string &method) const;

            const std::function<const double(const VectorXr &)> &obj_;
            const std::function<const VectorXr(const VectorXr &)> &grad_;
            // The three inputs are (x, compute_hessian, compute_hessian_projection)
            const std::function<const std::pair<SparseMatrixXr, SparseMatrixXr>(
                const VectorXr &, const bool, const bool)> &hess_and_proj_;

            const std::function<const double(const VectorXr &, const VectorXr &)> &max_step_size_;
        };

    } // namespace optimizers
} // namespace fsi