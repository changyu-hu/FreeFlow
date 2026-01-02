// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/common/MathUtils.hpp"
#include "FSI_Simulator/fem/FemConfig.hpp"

namespace fsi
{
    namespace optimizers
    {

        class LineSearch
        {
        public:
            // Possible methods: "backtracking", "wofle".
            LineSearch(const std::string &method,
                       const std::function<const double(const VectorXr &)> &obj,
                       const std::function<const VectorXr(const VectorXr &)> &grad);
            virtual ~LineSearch() {}

            const std::string &method() const { return method_; }
            const std::function<const double(const VectorXr &)> &obj() const { return obj_; }
            const std::function<const VectorXr(const VectorXr &)> &grad() const { return grad_; }

            // x = variable;
            // p = direction;
            // obj(x) = initial_obj;
            // grad(x) = initial_grad;
            // s = initial_step_size.
            // Goal: find s* \in [0, s] so that f(x + s* p) triggers sufficient decrease or sees negligible updates.
            // Return pair: (step size found, other information).
            virtual const std::pair<double, real> GetStepSize(const VectorXr &variable, const VectorXr &direction,
                                                              const double initial_obj, const VectorXr &initial_grad, const double initial_step_size,
                                                              const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var,
                                                              const fem::FemSolverOptions &opt) const = 0;

            const std::pair<double, real> GetStepSize(const VectorXr &variable, const VectorXr &direction,
                                                      const double initial_obj, const VectorXr &initial_grad, const double initial_step_size,
                                                      const fem::FemSolverOptions &opt) const;

            static const std::function<const bool(const VectorXr &, const VectorXr &)> GetDummyVariableConvergenceFunction();
            static const std::function<const bool(const VectorXr &, const VectorXr &)> GetFixedVariableConvergenceFunction();

        private:
            const std::string method_;

            const std::function<const double(const VectorXr &)> &obj_;
            const std::function<const VectorXr(const VectorXr &)> &grad_;
        };

    } // namespace optimizers
} // namespace fsi