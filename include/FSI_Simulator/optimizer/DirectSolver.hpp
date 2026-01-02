// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/common/MathUtils.hpp"
#include "FSI_Simulator/fem/FemConfig.hpp"
// Available matrix solvers.
#include "Eigen/SparseCholesky"
#include "Eigen/IterativeLinearSolvers"

#ifdef USE_SUITESPARSE
#include "Eigen/CholmodSupport"
#endif
// #include "Eigen/PardisoSupport"

#include <cusolverSp.h>
#include <cusparse.h>

#ifdef USE_CUDSS
#include "cudss.h"
#endif

namespace fsi
{
    namespace optimizers
    {
        class DirectSolver
        {
        public:
            DirectSolver(const std::string &method);
            ~DirectSolver();

            const VectorXr Solve(const SparseMatrixXr &lhs, const VectorXr &rhs, const fem::FemSolverOptions &opt);

        private:
            // Helper function to set up CUDA resources
            void setup_cuda_solver(const SparseMatrixXr &lhs);
            void free_cuda_resources();

            const std::string method_;
            bool solver_initialized_ = false;

            Eigen::SimplicialLDLT<SparseMatrixXr> eigen_ldlt_solver_;
#ifdef USE_SUITESPARSE
            Eigen::CholmodSimplicialLDLT<SparseMatrixXr> cholmod_ldlt_solver_;
#endif
            // Eigen::PardisoLDLT<SparseMatrixXr> pardiso_ldlt_solver_;
            // Eigen::BiCGSTAB<SparseMatrixXr, Eigen::IncompleteCholesky<double>> eigen_bicgstab_ilu_solver_;

            // --- CUDA/cusolverRf related members ---

            cusolverSpHandle_t cusolver_handle_ = nullptr;
            cusparseMatDescr_t mat_descr_ = nullptr;

            double *d_vals_ = nullptr;
            int *d_col_indices_ = nullptr;
            int *d_row_offsets_ = nullptr;
            double *d_b_ = nullptr; // Renamed from d_x_ for clarity
            double *d_x_ = nullptr; // Renamed from d_y_ for clarity

            int singularity_ = 0;

#ifdef USE_CUDSS
            void setup_cudss_solver(const SparseMatrixXr &lhs);
            bool is_first_factorization_ = true; //  Factorization or Refactorization

            cudssHandle_t cudss_handle_ = nullptr;
            cudssConfig_t cudss_config_ = nullptr;
            cudssData_t cudss_data_ = nullptr;
            int matrix_size_ = 0;
            int nnz_ = 0;
#endif
        };

    } // namespace optimizers
} // namespace fsi