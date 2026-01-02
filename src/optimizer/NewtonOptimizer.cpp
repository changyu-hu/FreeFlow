// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/optimizer/NewtonOptimizer.hpp"
#include "FSI_Simulator/optimizer/BacktrackingLineSearch.hpp"
#include "FSI_Simulator/optimizer/DirectSolver.hpp"
#include "FSI_Simulator/utils/Profiler.hpp"
#include <iostream>

namespace fsi
{
    namespace optimizers
    {

        NewtonOptimizer<MatrixXr>::NewtonOptimizer(const std::function<const double(const VectorXr &)> &obj,
                                                   const std::function<const VectorXr(const VectorXr &)> &grad,
                                                   const std::function<const MatrixXr(const VectorXr &)> &hess,
                                                   const std::function<const double(const VectorXr &, const VectorXr &)> &max_step_size)
            : obj_(obj), grad_(grad), hess_(hess), max_step_size_(max_step_size) {}

        const OptimizerReturnData NewtonOptimizer<MatrixXr>::Optimize(const VectorXr &initial_guess,
                                                                      const std::function<const bool(const VectorXr &)> &converge_grad,
                                                                      const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var,
                                                                      const std::function<const bool(const double, const double)> &converge_obj,
                                                                      const fem::FemSolverOptions &opt) const
        {
            const std::string error_location = "opt::NewtonOptimizer::Optimize";

            OptimizerReturnData ret;
            ret.success = false;

            // Setting up the line search algorithm.
            const std::string ls_method = "backtracking";
            const std::shared_ptr<LineSearch> ls = CreateLineSearch(ls_method);
            // Fetch verbose.
            integer verbose = 0;

            const integer max_iter = opt.iterations;

            // Initial guess.
            VectorXr xk = initial_guess;
            double Ek = obj_(xk);
            VectorXr gk = grad_(xk);
            LOG_TRACE("Initial guess: E0 = {}, |g0| = {}", Ek, gk.norm());
            // Check convergence.

            if (converge_grad(gk))
            {
                // Done.
                ret.solution = xk;
                ret.obj = Ek;
                ret.grad = gk;
                ret.info = {};
                ret.success = true;
                return ret;
            }

            for (integer k = 0; k < max_iter; ++k)
            {
                const MatrixXr Hk = hess_(xk);
                // Solve Hk * pk = -gk.
                // Since this is a dense matrix, we will assume it is small and will simply call its eigen decomposition.
                Eigen::SelfAdjointEigenSolver<MatrixXr> eig_solver(Hk);
                const VectorXr &la = eig_solver.eigenvalues();
                const MatrixXr &V = eig_solver.eigenvectors();
                // V * la.asDiagonal() * V.transpose() = Hk.

                // Hessian projection: See the discussion around (3.41) in "Numerical Optimization", which suggested
                // using sqrt(machine precision).
                const double delta = std::sqrt(std::numeric_limits<double>::epsilon());
                // If la > delta, we keep it; otherwise, we replace it with |la| + delta.
                const VectorXr projected_la = (la.array() > delta).select(la, la.cwiseAbs().array() + delta);
                // Projected Hessian = V * projected_la * V.transpose().
                // V * projected_la * V.transpose * pk = -gk.
                // pk = V * projected_la.inv() * V.t * -gk.
                const VectorXr pk = V * (projected_la.cwiseInverse().asDiagonal()) * V.transpose() * -gk;

                // Line search.
                const double initial_step_size = max_step_size_(xk, pk);
                const auto ls_ret = ls->GetStepSize(xk, pk, Ek, gk, initial_step_size, opt);
                const double ls_step = ls_ret.first;

                LOG_TRACE("Newton iteration {}: |p{}| = {}, s{} = {}, |g{}| = {}, pk.dot(gk) = {}", k, k, pk.norm(), k, ls_step, k, gk.norm(), pk.dot(gk));

                // Update.
                const VectorXr xk_last = xk;
                const double Ek_last = Ek;
                xk = xk + ls_step * pk;
                Ek = ls_ret.second;
                gk = grad_(xk);

                LOG_TRACE("E{} = {}, |g{}| = {}", k + 1, Ek, k + 1, gk.norm());

                // Check convergence.
                if (converge_grad(gk) || converge_var(xk, xk_last) || converge_obj(Ek, Ek_last))
                {
                    // Done.
                    ret.solution = xk;
                    ret.obj = Ek;
                    ret.grad = gk;
                    ret.info = {};
                    ret.success = true;
                    return ret;
                }
            }

            LOG_ERROR("Newton's method failed to converge. Check whether your hyperparameters are sensible, or something must be seriously wrong.");
            return ret;
        }

        const OptimizerReturnData NewtonOptimizer<MatrixXr>::Optimize(const VectorXr &initial_guess,
                                                                      const std::function<const bool(const VectorXr &)> &converge_grad,
                                                                      const fem::FemSolverOptions &opt) const
        {

            return Optimize(initial_guess,
                            converge_grad,
                            GetDummyVariableConvergenceFunction(),
                            GetDummyObjectiveConvergenceFunction(),
                            opt);
        }

        const OptimizerReturnData NewtonOptimizer<MatrixXr>::Optimize(const VectorXr &initial_guess,
                                                                      const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var,
                                                                      const fem::FemSolverOptions &opt) const
        {

            return Optimize(initial_guess,
                            GetDummyGradientConvergenceFunction(),
                            converge_var,
                            GetDummyObjectiveConvergenceFunction(),
                            opt);
        }

        const OptimizerReturnData NewtonOptimizer<MatrixXr>::Optimize(const VectorXr &initial_guess,
                                                                      const std::function<const bool(const double, const double)> &converge_obj,
                                                                      const fem::FemSolverOptions &opt) const
        {

            return Optimize(initial_guess,
                            GetDummyGradientConvergenceFunction(),
                            GetDummyVariableConvergenceFunction(),
                            converge_obj,
                            opt);
        }

        const std::function<const bool(const VectorXr &)> NewtonOptimizer<MatrixXr>::GetDummyGradientConvergenceFunction()
        {
            const std::function<const bool(const VectorXr &)> converge_grad = [](
                                                                                  const VectorXr &)
            {
                return false;
            };
            return converge_grad;
        }

        const std::function<const bool(const VectorXr &, const VectorXr &)> NewtonOptimizer<MatrixXr>::GetDummyVariableConvergenceFunction()
        {
            const std::function<const bool(const VectorXr &, const VectorXr &)> converge_var = [](
                                                                                                   const VectorXr &, const VectorXr &)
            {
                return false;
            };
            return converge_var;
        }

        const std::function<const bool(const double, const double)> NewtonOptimizer<MatrixXr>::GetDummyObjectiveConvergenceFunction()
        {
            const std::function<const bool(const double, const double)> converge_obj = [](
                                                                                           const double, const double)
            {
                return false;
            };
            return converge_obj;
        }

        const std::shared_ptr<LineSearch> NewtonOptimizer<MatrixXr>::CreateLineSearch(const std::string &method) const
        {
            std::shared_ptr<LineSearch> line_search = nullptr;
            if (method == "backtracking")
            {
                line_search = std::make_shared<BacktrackingLineSearch>(obj_, grad_);
            }
            else
            {
                LOG_ERROR("Unsupported line search method: {}.", method);
            }
            return line_search;
        }

        NewtonOptimizer<SparseMatrixXr>::NewtonOptimizer(const std::function<const double(const VectorXr &)> &obj,
                                                         const std::function<const VectorXr(const VectorXr &)> &grad,
                                                         const std::function<const std::pair<SparseMatrixXr, SparseMatrixXr>(
                                                             const VectorXr &, const bool, const bool)> &hess_and_proj,
                                                         const std::function<const double(const VectorXr &, const VectorXr &)> &max_step_size)
            : obj_(obj), grad_(grad), hess_and_proj_(hess_and_proj), max_step_size_(max_step_size) {}

        const OptimizerReturnData NewtonOptimizer<SparseMatrixXr>::Optimize(const VectorXr &initial_guess,
                                                                            const std::function<const bool(const VectorXr &)> &converge_grad,
                                                                            const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var,
                                                                            const std::function<const bool(const double, const double)> &converge_obj,
                                                                            const fem::FemSolverOptions &opt) const
        {

            const std::string error_location = "opt::NewtonOptimizer::Optimize";

            OptimizerReturnData ret;

            // Fetch newton iteration.
            const integer max_iter = opt.iterations;
            // Fetch linear solver information.
            const std::string solver_type = "direct";

            // Setting up the line search algorithm.
            const std::string ls_method = "backtracking";
            const std::shared_ptr<LineSearch> ls = CreateLineSearch(ls_method);

            // Initial guess.
            VectorXr xk = initial_guess;
            double Ek = obj_(xk);
            VectorXr gk = grad_(xk);
            DirectSolver direct_solver(opt.linear_solver_type);
            LOG_TRACE("Initial guess: E0 = {}, |g0| = {}", Ek, gk.norm());
            bool conv = converge_grad(gk);

            if (conv)
            {
                // Done.
                ret.solution = xk;
                ret.obj = Ek;
                ret.grad = gk;
                ret.info = {};
                ret.success = true;
                return ret;
            }

            for (integer k = 0; k < max_iter; ++k)
            {
                // For now, we will use the strategy that simply computes the Hessian projection every time.
                // We might want to be more flexible in the future, so we keep the function hess_and_proj.
                // It is the caller's responsibility to ensure hess_and_proj returns SPD projection.
                SparseMatrixXr Hk_proj = hess_and_proj_(xk, false, true).second;
                // Solve Hk_proj * pk = -gk.
                VectorXr pk = VectorXr::Zero(gk.size());
                if (solver_type == "direct")
                {
                    PROFILE_SCOPE("DirectSolver::Solve");
                    Hk_proj.makeCompressed();
                    pk = direct_solver.Solve(Hk_proj, -gk, opt);
                }
                else
                {
                    LOG_ERROR("Unsupported solver type.");
                }
                // A sanity check: Hk_proj must be strictly SPD.
                // Note that gk must be nonzero. Otherwise, the iteration would have converged.
                // So pk.dot(gk) < 0, i.e., it must be always strictly negative.
                // CheckCondition(pk.dot(gk) < 0, error_location, "The Hessian projection is not strictly positive definite: pk.dot(gk) = "
                //     + std::to_string(pk.dot(gk)) + ".");
                if (pk.dot(gk) > 0)
                {
                    pk = -gk;
                }

                // Line search.
                const double initial_step_size = max_step_size_(xk, pk);
                const auto ls_ret = ls->GetStepSize(xk, pk, Ek, gk, initial_step_size, opt);
                const double ls_step = ls_ret.first;

                LOG_TRACE("Newton iteration {}: |p{}| = {}, s{}={}, |g{}| = {}, pk.dot(gk) = {}", k, k, pk.norm(), k, ls_step, k, gk.norm(), pk.dot(gk));

                // Update.
                const VectorXr xk_last = xk;
                const double Ek_last = Ek;
                xk = xk + ls_step * pk;
                Ek = ls_ret.second;
                gk = grad_(xk);

                LOG_TRACE("E{} = {}, |g{}| = {}", k + 1, Ek, k + 1, gk.norm());

                // Check convergence.
                conv = converge_grad(gk) || converge_var(xk, xk_last) || converge_obj(Ek, Ek_last);
                if (conv)
                {
                    // Done.
                    ret.solution = xk;
                    ret.obj = Ek;
                    ret.grad = gk;
                    ret.info = {};
                    ret.success = true;
                    return ret;
                }
            }

            // CheckCondition(false, error_location, "Newton's method failed to converge. "
            //     "Check whether your hyperparameters are sensible, or something must be seriously wrong.");
            LOG_ERROR("Newton's method failed to converge. |g| = {}", gk.norm());
            ret.solution = initial_guess;
            ret.obj = Ek;
            ret.grad = gk;
            ret.info = {};
            ret.success = false;
            return ret;
        }

        const OptimizerReturnData NewtonOptimizer<SparseMatrixXr>::Optimize(const VectorXr &initial_guess,
                                                                            const std::function<const bool(const VectorXr &)> &converge_grad,
                                                                            const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var,
                                                                            const fem::FemSolverOptions &opt) const
        {

            return Optimize(initial_guess,
                            converge_grad,
                            converge_var,
                            GetDummyObjectiveConvergenceFunction(),
                            opt);
        }

        const OptimizerReturnData NewtonOptimizer<SparseMatrixXr>::Optimize(const VectorXr &initial_guess,
                                                                            const std::function<const bool(const VectorXr &)> &converge_grad,
                                                                            const fem::FemSolverOptions &opt) const
        {

            return Optimize(initial_guess,
                            converge_grad,
                            GetDummyVariableConvergenceFunction(),
                            GetDummyObjectiveConvergenceFunction(),
                            opt);
        }

        const OptimizerReturnData NewtonOptimizer<SparseMatrixXr>::Optimize(const VectorXr &initial_guess,
                                                                            const std::function<const bool(const VectorXr &, const VectorXr &)> &converge_var,
                                                                            const fem::FemSolverOptions &opt) const
        {

            return Optimize(initial_guess,
                            GetDummyGradientConvergenceFunction(),
                            converge_var,
                            GetDummyObjectiveConvergenceFunction(),
                            opt);
        }

        const OptimizerReturnData NewtonOptimizer<SparseMatrixXr>::Optimize(const VectorXr &initial_guess,
                                                                            const std::function<const bool(const double, const double)> &converge_obj,
                                                                            const fem::FemSolverOptions &opt) const
        {

            return Optimize(initial_guess,
                            GetDummyGradientConvergenceFunction(),
                            GetDummyVariableConvergenceFunction(),
                            converge_obj,
                            opt);
        }

        const std::function<const bool(const VectorXr &)> NewtonOptimizer<SparseMatrixXr>::GetDummyGradientConvergenceFunction()
        {
            const std::function<const bool(const VectorXr &)> converge_grad = [](
                                                                                  const VectorXr &)
            {
                return false;
            };
            return converge_grad;
        }

        const std::function<const bool(const VectorXr &, const VectorXr &)> NewtonOptimizer<SparseMatrixXr>::GetDummyVariableConvergenceFunction()
        {
            const std::function<const bool(const VectorXr &, const VectorXr &)> converge_var = [](
                                                                                                   const VectorXr &, const VectorXr &)
            {
                return false;
            };
            return converge_var;
        }

        const std::function<const bool(const double, const double)> NewtonOptimizer<SparseMatrixXr>::GetDummyObjectiveConvergenceFunction()
        {
            const std::function<const bool(const double, const double)> converge_obj = [](
                                                                                           const double, const double)
            {
                return false;
            };
            return converge_obj;
        }

        const std::shared_ptr<LineSearch> NewtonOptimizer<SparseMatrixXr>::CreateLineSearch(const std::string &method) const
        {
            std::shared_ptr<LineSearch> line_search = nullptr;
            if (method == "backtracking")
            {
                line_search = std::make_shared<BacktrackingLineSearch>(obj_, grad_);
            }
            else
            {
                LOG_ERROR("Unsupported line search method: {}.", method);
            }
            return line_search;
        }
    }
}