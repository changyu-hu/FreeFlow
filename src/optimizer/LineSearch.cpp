// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/optimizer/LineSearch.hpp"

namespace fsi
{
    namespace optimizers
    {

        // Possible methods: "backtracking", "wofle".
        LineSearch::LineSearch(const std::string &method,
                               const std::function<const double(const VectorXr &)> &obj,
                               const std::function<const VectorXr(const VectorXr &)> &grad)
            : method_(method), obj_(obj), grad_(grad) {}

        const std::pair<double, real> LineSearch::GetStepSize(const VectorXr &variable, const VectorXr &direction,
                                                              const double initial_obj, const VectorXr &initial_grad, const double initial_step_size,
                                                              const fem::FemSolverOptions &opt) const
        {

            return GetStepSize(variable, direction, initial_obj, initial_grad, initial_step_size,
                               GetFixedVariableConvergenceFunction(), opt);
        }

        const std::function<const bool(const VectorXr &, const VectorXr &)> LineSearch::GetDummyVariableConvergenceFunction()
        {
            const std::function<const bool(const VectorXr &, const VectorXr &)> converge_var = [](
                                                                                                   const VectorXr &, const VectorXr &)
            {
                return false;
            };
            return converge_var;
        }

        const std::function<const bool(const VectorXr &, const VectorXr &)> LineSearch::GetFixedVariableConvergenceFunction()
        {
            const std::function<const bool(const VectorXr &, const VectorXr &)> converge_var = [](
                                                                                                   const VectorXr &x_update, const VectorXr &x)
            {
                double eps = 1e-5;
                if ((x_update - x).cwiseAbs().maxCoeff() < eps)
                {
                    return true;
                }
                return false;
            };
            return converge_var;
        }

    } // namespace optimizers
} // namespace fsi