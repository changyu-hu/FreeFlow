// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/optimizer/DirectSolver.hpp"
#include "FSI_Simulator/utils/CudaErrorCheck.cuh"

#include <cuda_runtime.h>
#include <unsupported/Eigen/SparseExtra>
#define CUSOLVER_CHECK(call)                                    \
    do                                                          \
    {                                                           \
        cusolverStatus_t err = call;                            \
        if (err != CUSOLVER_STATUS_SUCCESS)                     \
        {                                                       \
            LOG_CRITICAL("CUSOLVER Error in {}:{}",             \
                         __FILE__, __LINE__);                   \
            throw std::runtime_error("CUSOLVER runtime error"); \
        }                                                       \
    } while (0)

#define CUSPARSE_CHECK(call)                                               \
    do                                                                     \
    {                                                                      \
        cusparseStatus_t err = call;                                       \
        if (err != CUSPARSE_STATUS_SUCCESS)                                \
        {                                                                  \
            LOG_CRITICAL("CUSPARSE Error in {}:{} : {}",                   \
                         __FILE__, __LINE__, cusparseGetErrorString(err)); \
            throw std::runtime_error("CUSPARSE runtime error");            \
        }                                                                  \
    } while (0)

#ifdef USE_CUDSS
#define CUDSS_CHECK(call)                                    \
    do                                                       \
    {                                                        \
        cudssStatus_t err = call;                            \
        if (err != CUDSS_STATUS_SUCCESS)                     \
        {                                                    \
            LOG_CRITICAL("CUDSS Error in {}:{}",             \
                         __FILE__, __LINE__);                \
            throw std::runtime_error("CUDSS runtime error"); \
        }                                                    \
    } while (0)
#endif

namespace fsi
{
    namespace optimizers
    {

        DirectSolver::DirectSolver(const std::string &method)
            : method_(method)
        {

            // Supported methods.
            if (method_ == "eigen_ldlt")
            {
                // eigen_ldlt_solver_.analyzePattern(lhs_);
            }
#ifdef USE_SUITESPARSE
            else if (method_ == "cholmod_ldlt")
            {
                // cholmod_ldlt_solver_.analyzePattern(lhs_);
                // if(cholmod_ldlt_solver_.info() != Eigen::Success) { LOG_ERROR("CholmodSimplicialLDLT analyzePattern failed."); /* 处理错误 */ }
            }
#endif
            else if (method_ == "cuda_qr")
            {
            }
#ifdef USE_CUDSS
            else if (method_ == "cuda_lu")
            {
            }
#endif
            else
            {
                LOG_ERROR("Unsupported method: " + method + ".");
            }
        }

        DirectSolver::~DirectSolver()
        {
            if (method_ == "cuda_qr" || method_ == "cuda_lu")
            {
                free_cuda_resources();
            }
        }

        const VectorXr DirectSolver::Solve(const SparseMatrixXr &lhs, const VectorXr &rhs, const fem::FemSolverOptions &opt)
        {

            // For now we do not have any int/real options, but we keep them for future usage.
            if (method_ == "eigen_ldlt")
            {
                if (!solver_initialized_)
                {
                    eigen_ldlt_solver_.analyzePattern(lhs);
                    solver_initialized_ = true;
                }
                eigen_ldlt_solver_.factorize(lhs);
                const VectorXr x = eigen_ldlt_solver_.solve(rhs);
                if (eigen_ldlt_solver_.info() != Eigen::Success)
                    LOG_ERROR("SimplicialLDLT fails to solve the rhs vector.");
                return x;
            }
#ifdef USE_SUITESPARSE
            else if (method_ == "cholmod_ldlt")
            {
                if (!solver_initialized_)
                {
                    cholmod_ldlt_solver_.analyzePattern(lhs);
                    if (cholmod_ldlt_solver_.info() != Eigen::Success)
                        LOG_ERROR("CholmodSimplicialLDLT fails to analyzePattern.");
                    solver_initialized_ = true;
                }
                cholmod_ldlt_solver_.factorize(lhs);
                if (cholmod_ldlt_solver_.info() != Eigen::Success)
                    LOG_ERROR("CholmodSimplicialLDLT fails to factorize the lhs matrix.");
                const VectorXr x = cholmod_ldlt_solver_.solve(rhs);
                if (cholmod_ldlt_solver_.info() != Eigen::Success)
                    LOG_ERROR("CholmodSimplicialLDLT fails to solve the rhs vector.");
                return x;
            }
#endif
            // else if (method_ == "eigen_bicgstab") {
            //     eigen_bicgstab_ilu_solver_.compute(lhs);
            //     const VectorXr x = eigen_bicgstab_ilu_solver_.solve(rhs);
            //     if (eigen_sparse_lu_solver_.info() != Eigen::Success) LOG_ERROR("SPARSELU fails to solve the rhs vector.");
            //     return x;
            // } else if (method_ == "mkl_ldlt") {
            //     pardiso_ldlt_solver_.factorize(lhs);
            //     if (pardiso_ldlt_solver_.info() != Eigen::Success) LOG_ERROR("PardisoLDLT fails to factorize the lhs matrix.");
            //     const VectorXr x = pardiso_ldlt_solver_.solve(rhs);
            //     if (pardiso_ldlt_solver_.info() != Eigen::Success) LOG_ERROR("PardisoLDLT fails to solve the rhs vector.");
            //     return x;
            // }
            if (method_ == "cuda_qr")
            {
                if (!solver_initialized_)
                {
                    setup_cuda_solver(lhs);
                    solver_initialized_ = true;
                }

                Eigen::SparseMatrix<real, Eigen::RowMajor> lhs_csr = lhs;
                lhs_csr.makeCompressed();
                const real *values = lhs_csr.valuePtr();
                const int nnz = lhs_csr.nonZeros();

                CUDA_CHECK(cudaMemcpy(d_vals_, values, nnz * sizeof(real), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_b_, rhs.data(), rhs.size() * sizeof(real), cudaMemcpyHostToDevice));

                CUSOLVER_CHECK(cusolverSpDcsrlsvqr(
                    cusolver_handle_,
                    lhs.rows(),
                    nnz,
                    mat_descr_,
                    d_vals_,
                    d_row_offsets_,
                    d_col_indices_,
                    d_b_,  // Input: RHS vector b
                    1e-12, // Tolerance
                    0,     // Reordering (0 for AMD)
                    d_x_,  // Output: Solution vector x
                    &singularity_));

                if (singularity_ >= 0)
                {
                    LOG_WARN("Matrix singularity detected by QR solver at row: {}.", singularity_);
                }

                VectorXr result(rhs.size());
                CUDA_CHECK(cudaMemcpy(result.data(), d_x_, result.size() * sizeof(real), cudaMemcpyDeviceToHost));

                return result;
            }
#ifdef USE_CUDSS
            else if (method_ == "cuda_lu")
            {
                if (!solver_initialized_)
                {
                    setup_cudss_solver(lhs); // Implement this function if needed
                    solver_initialized_ = true;
                }
                Eigen::SparseMatrix<real, Eigen::RowMajor> lhs_csr = lhs;
                lhs_csr.makeCompressed();
                const real *values = lhs_csr.valuePtr();

                CUDA_CHECK(cudaMemcpy(d_vals_, values, nnz_ * sizeof(real), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_b_, rhs.data(), matrix_size_ * sizeof(real), cudaMemcpyHostToDevice));

                cudssMatrix_t mat_A, vec_x, vec_b;
                CUDSS_CHECK(cudssMatrixCreateCsr(&mat_A, matrix_size_, matrix_size_, nnz_,
                                                 d_row_offsets_, nullptr, d_col_indices_, d_vals_,
                                                 CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_GENERAL,
                                                 CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
                CUDSS_CHECK(cudssMatrixCreateDn(&vec_x, matrix_size_, 1, matrix_size_, d_x_, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
                CUDSS_CHECK(cudssMatrixCreateDn(&vec_b, matrix_size_, 1, matrix_size_, d_b_, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

                if (is_first_factorization_)
                {
                    CUDSS_CHECK(cudssExecute(cudss_handle_, CUDSS_PHASE_FACTORIZATION, cudss_config_, cudss_data_, mat_A, vec_x, vec_b));
                    is_first_factorization_ = false;
                }
                else
                {
                    CUDSS_CHECK(cudssExecute(cudss_handle_, CUDSS_PHASE_REFACTORIZATION, cudss_config_, cudss_data_, mat_A, vec_x, vec_b));
                }

                CUDSS_CHECK(cudssExecute(cudss_handle_, CUDSS_PHASE_SOLVE, cudss_config_, cudss_data_, mat_A, vec_x, vec_b));

                CUDSS_CHECK(cudssMatrixDestroy(mat_A));
                CUDSS_CHECK(cudssMatrixDestroy(vec_x));
                CUDSS_CHECK(cudssMatrixDestroy(vec_b));

                VectorXr result(matrix_size_);
                CUDA_CHECK(cudaMemcpy(result.data(), d_x_, matrix_size_ * sizeof(real), cudaMemcpyDeviceToHost));

                return result;
            }
#endif
            else
            {
                LOG_ERROR("Unsupported method: {}.", method_);
                return VectorXr::Zero(lhs.cols());
            }
        }

        void DirectSolver::setup_cuda_solver(const SparseMatrixXr &lhs)
        {
            Eigen::SparseMatrix<real, Eigen::RowMajor> lhs_csr = lhs;
            lhs_csr.makeCompressed();

            const int rows = lhs_csr.rows();
            const int nnz = lhs_csr.nonZeros();

            const int *row_offsets = lhs_csr.outerIndexPtr();
            const int *col_indices = lhs_csr.innerIndexPtr();

            CUSOLVER_CHECK(cusolverSpCreate(&cusolver_handle_));
            CUSPARSE_CHECK(cusparseCreateMatDescr(&mat_descr_));
            CUSPARSE_CHECK(cusparseSetMatType(mat_descr_, CUSPARSE_MATRIX_TYPE_GENERAL));
            CUSPARSE_CHECK(cusparseSetMatIndexBase(mat_descr_, CUSPARSE_INDEX_BASE_ZERO));

            CUDA_CHECK(cudaMalloc((void **)&d_row_offsets_, (rows + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc((void **)&d_col_indices_, nnz * sizeof(int)));
            CUDA_CHECK(cudaMalloc((void **)&d_vals_, nnz * sizeof(real)));
            CUDA_CHECK(cudaMalloc((void **)&d_b_, rows * sizeof(real))); // RHS
            CUDA_CHECK(cudaMalloc((void **)&d_x_, rows * sizeof(real))); // Solution

            CUDA_CHECK(cudaMemcpy(d_row_offsets_, row_offsets, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_col_indices_, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
        }

#ifdef USE_CUDSS
        void DirectSolver::setup_cudss_solver(const SparseMatrixXr &lhs)
        {
            Eigen::SparseMatrix<real, Eigen::RowMajor> lhs_csr = lhs;
            lhs_csr.makeCompressed();

            matrix_size_ = lhs_csr.rows();
            nnz_ = lhs_csr.nonZeros();

            const int *row_offsets = lhs_csr.outerIndexPtr();
            const int *col_indices = lhs_csr.innerIndexPtr();

            CUDSS_CHECK(cudssCreate(&cudss_handle_));
            CUDSS_CHECK(cudssConfigCreate(&cudss_config_));
            CUDSS_CHECK(cudssDataCreate(cudss_handle_, &cudss_data_));

            CUDA_CHECK(cudaMalloc((void **)&d_row_offsets_, (matrix_size_ + 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc((void **)&d_col_indices_, nnz_ * sizeof(int)));
            CUDA_CHECK(cudaMalloc((void **)&d_vals_, nnz_ * sizeof(real)));
            CUDA_CHECK(cudaMalloc((void **)&d_b_, matrix_size_ * sizeof(real)));
            CUDA_CHECK(cudaMalloc((void **)&d_x_, matrix_size_ * sizeof(real)));

            CUDA_CHECK(cudaMemcpy(d_row_offsets_, row_offsets, (matrix_size_ + 1) * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_col_indices_, col_indices, nnz_ * sizeof(int), cudaMemcpyHostToDevice));

            cudssMatrix_t mat_A_structure, vec_x_structure, vec_b_structure;
            CUDSS_CHECK(cudssMatrixCreateCsr(&mat_A_structure, matrix_size_, matrix_size_, nnz_,
                                             d_row_offsets_, nullptr, d_col_indices_, d_vals_,
                                             CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_GENERAL,
                                             CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
            CUDSS_CHECK(cudssMatrixCreateDn(&vec_x_structure, matrix_size_, 1, matrix_size_, d_x_, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
            CUDSS_CHECK(cudssMatrixCreateDn(&vec_b_structure, matrix_size_, 1, matrix_size_, d_b_, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

            CUDSS_CHECK(cudssExecute(cudss_handle_, CUDSS_PHASE_ANALYSIS, cudss_config_, cudss_data_, mat_A_structure, vec_x_structure, vec_b_structure));

            CUDSS_CHECK(cudssMatrixDestroy(mat_A_structure));
            CUDSS_CHECK(cudssMatrixDestroy(vec_x_structure));
            CUDSS_CHECK(cudssMatrixDestroy(vec_b_structure));
        }
#endif

        void DirectSolver::free_cuda_resources()
        {
            if (!solver_initialized_)
                return;

            if (d_x_)
                cudaFree(d_x_);
            if (d_b_)
                cudaFree(d_b_);
            if (d_vals_)
                cudaFree(d_vals_);
            if (d_col_indices_)
                cudaFree(d_col_indices_);
            if (d_row_offsets_)
                cudaFree(d_row_offsets_);

            if (mat_descr_)
                cusparseDestroyMatDescr(mat_descr_);
            if (cusolver_handle_)
                cusolverSpDestroy(cusolver_handle_);

#ifdef USE_CUDSS
            if (cudss_data_)
                CUDSS_CHECK(cudssDataDestroy(cudss_handle_, cudss_data_));
            if (cudss_config_)
                CUDSS_CHECK(cudssConfigDestroy(cudss_config_));
            if (cudss_handle_)
                CUDSS_CHECK(cudssDestroy(cudss_handle_));
#endif

            solver_initialized_ = false;
        }
    } // namespace optimizers
} // namespace fsi