// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#ifndef SPARSE_MATRIX_UTILS_HPP
#define SPARSE_MATRIX_UTILS_HPP

#include "FSI_Simulator/common/Types.hpp"

#include <vector>
#include <stdexcept>

namespace fsi
{
    inline const SparseMatrixXr FromTriplet(
        const double row_num,
        const int col_num,
        const std::vector<Eigen::Triplet<double>> &nonzeros)
    {
        SparseMatrixXr mat(row_num, col_num);
        mat.setFromTriplets(nonzeros.begin(), nonzeros.end());
        mat.makeCompressed();
        return mat;
    }

    inline const std::vector<Eigen::Triplet<double>> ToTriplet(const SparseMatrixXr &mat)
    {
        SparseMatrixXr mat_compressed = mat;
        mat_compressed.makeCompressed();
        std::vector<Eigen::Triplet<double>> nonzeros;
        for (int k = 0; k < static_cast<int>(mat_compressed.outerSize()); ++k)
        {
            for (SparseMatrixXr::InnerIterator it(mat_compressed, k); it; ++it)
            {
                nonzeros.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
            }
        }
        return nonzeros;
    }

    inline const SparseMatrixXr FromDiagonal(const VectorXr &diagonal)
    {
        const int matrix_size = static_cast<int>(diagonal.size());
        std::vector<Eigen::Triplet<double>> nonzeros;
        for (int i = 0; i < matrix_size; ++i)
        {
            nonzeros.push_back(Eigen::Triplet<double>(i, i, diagonal(i)));
        }
        return FromTriplet(matrix_size, matrix_size, nonzeros);
    }

    inline const SparseMatrixXr FromDiagonal(const double *diagonal, int size)
    {
        std::vector<Eigen::Triplet<double>> nonzeros;
        for (int i = 0; i < size; ++i)
        {
            nonzeros.push_back(Eigen::Triplet<double>(i, i, diagonal[i]));
        }
        return FromTriplet(size, size, nonzeros);
    }

    inline const SparseMatrixXr ApplyDirichlet(const SparseMatrixXr &A, const VectorXi &dirichlet_dof)
    {
        const int size = static_cast<int>(A.rows());
        const std::vector<Eigen::Triplet<double>> A_nonzeros = ToTriplet(A);
        std::vector<Eigen::Triplet<double>> A_modified_nonzeros;
        for (const auto &triplet : A_nonzeros)
        {
            const int row = triplet.row();
            const int col = triplet.col();
            if (dirichlet_dof(row) == 1 || dirichlet_dof(col) == 1)
                continue;
            A_modified_nonzeros.push_back(triplet);
        }
        for (int i = 0; i < size; ++i)
        {
            if (dirichlet_dof(i) == 1)
            {
                A_modified_nonzeros.push_back(Eigen::Triplet<double>(i, i, 1));
            }
        }
        return FromTriplet(size, size, A_modified_nonzeros);
    }

} // namespace fsi
#endif // SPARSE_MATRIX_UTILS_HPP
