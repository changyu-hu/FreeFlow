// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/optimizer/BacktrackingLineSearch.hpp"

namespace fsi
{
    namespace optimizers
    {

        BacktrackingLineSearch::BacktrackingLineSearch(const std::function<const double(const VectorXr &)> &obj,
                                                       const std::function<const VectorXr(const VectorXr &)> &grad)
            : LineSearch("backtracking", obj, grad) {}

        const std::pair<double, real> BacktrackingLineSearch::GetStepSize(const VectorXr &variable, const VectorXr &direction,
                                                                          const double initial_obj, const VectorXr &initial_grad, const double initial_step_size,
                                                                          const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var, const fem::FemSolverOptions &opt) const
        {
            // Alg. 9.2 in "Convex Optimization".

            const std::string error_location = "opt::BacktrackingLineSearch::GetStepSize";

            const integer max_iter = opt.ls_max_iter;
            const double beta = opt.ls_beta;
            const double ls_alpha = opt.ls_alpha;
            // In practice, ls_alpha is typically quite small, e.g., 1e-4.
            // Reference: "Numerical Optimization", above Eqn. (3.5).
            ASSERT(max_iter > 0 && 0 < beta && beta < 1 && 0 < ls_alpha && ls_alpha < 1, "Hyperparameters out of bound.");

            // Rename.
            const VectorXr &x = variable;
            const VectorXr &p = direction;
            const double f = initial_obj;
            const VectorXr g = initial_grad;

            LOG_TRACE("Initial value at line search: f = {}, |p| = {}", f, p.norm());

            double s = initial_step_size;
            for (integer k = 0; k < max_iter; ++k)
            {
                const double fk = obj()(x + s * p);
                const double fline = f + ls_alpha * s * g.dot(p);
                LOG_TRACE("Line search iter {}: fk = {}, fline = {}, s = {}", k, fk, fline, s);
                if (std::isnan(fk) || std::isinf(fk) || std::isnan(fline) || std::isinf(fline))
                {
                    // Case 1: bad numbers.
                    LOG_ERROR("Inf or NaN encountered.");
                    s *= beta;
                }
                else if (converge_var(x + s * p, x))
                {
                    // Case 2: negligible udpate.
                    LOG_TRACE("Negligible update.");
                    return {s, fk};
                }
                else if (fk <= fline)
                {
                    // Case 3: sufficient decrease.
                    LOG_TRACE("Sufficient decrease.");
                    return {s, fk};
                }
                else
                {
                    // Case 4: good numbers, but it hasn't converged or encountered sufficient decrease.
                    LOG_TRACE("Shrink s and retry.");
                    s *= beta;
                }
            }
            LOG_ERROR("Line search failed to find sufficient decrease.");
            const double fk = obj()(x + s * p);
            return {s, fk};
        }

    } // namespace optimizers
} // namespace fsi