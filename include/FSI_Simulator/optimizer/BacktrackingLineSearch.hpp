// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/common/MathUtils.hpp"
#include "FSI_Simulator/optimizer/LineSearch.hpp"

namespace fsi
{
    namespace optimizers
    {

        // According to "Numerical Optimization", this is well suited for Newton but less ideal for quasi-Newton.
        class BacktrackingLineSearch : public LineSearch
        {
        public:
            BacktrackingLineSearch(const std::function<const double(const VectorXr &)> &obj,
                                   const std::function<const VectorXr(const VectorXr &)> &grad);

            const std::pair<double, real> GetStepSize(const VectorXr &variable, const VectorXr &direction,
                                                      const double initial_obj, const VectorXr &initial_grad, const double initial_step_size,
                                                      const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var,
                                                      const fem::FemSolverOptions &opt) const override;
        };

    } // namespace optimizers
} // namespace fsi