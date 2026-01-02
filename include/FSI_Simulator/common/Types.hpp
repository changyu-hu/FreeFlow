// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include <FSI_Simulator/utils/Logger.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace fsi
{
    using real = double; // float or double.
    using integer = int; // int32_t or int64_t.

    using vec2_t = std::conditional_t<std::is_same_v<real, double>, glm::dvec2, glm::fvec2>;
    using vec3_t = std::conditional_t<std::is_same_v<real, double>, glm::dvec3, glm::fvec3>;
    using vec4_t = std::conditional_t<std::is_same_v<real, double>, glm::dvec4, glm::fvec4>;
    using mat2_t = std::conditional_t<std::is_same_v<real, double>, glm::dmat2, glm::fmat2>;
    using mat3_t = std::conditional_t<std::is_same_v<real, double>, glm::dmat3, glm::fmat3>;
    using mat4_t = std::conditional_t<std::is_same_v<real, double>, glm::dmat4, glm::fmat4>;

    using vec9_t = glm::vec<9, real, glm::defaultp>;
    using mat9_t = glm::mat<9, 9, real, glm::defaultp>;
    using mat3x9_t = glm::mat<3, 9, real, glm::defaultp>;

    using Vector2r = Eigen::Matrix<real, 2, 1>;
    using Vector3r = Eigen::Matrix<real, 3, 1>;
    using Vector4r = Eigen::Matrix<real, 4, 1>;
    using Vector6r = Eigen::Matrix<real, 6, 1>;
    using Vector8r = Eigen::Matrix<real, 8, 1>;
    using Vector9r = Eigen::Matrix<real, 9, 1>;
    using VectorXr = Eigen::Matrix<real, Eigen::Dynamic, 1>;
    using Vector3i = Eigen::Matrix<int, 3, 1>;
    using VectorXi = Eigen::Matrix<int, Eigen::Dynamic, 1>;

    using RowVectorXr = Eigen::Matrix<real, 1, Eigen::Dynamic>;

    using Matrix2r = Eigen::Matrix<real, 2, 2>;
    using Matrix3r = Eigen::Matrix<real, 3, 3>;
    using Matrix4r = Eigen::Matrix<real, 4, 4>;
    using Matrix6r = Eigen::Matrix<real, 6, 6>;
    using Matrix8r = Eigen::Matrix<real, 8, 8>;
    using Matrix9r = Eigen::Matrix<real, 9, 9>;
    using Matrix12r = Eigen::Matrix<real, 12, 12>;
    using Matrix23 = Eigen::Matrix<real, 2, 3>;
    using Matrix32 = Eigen::Matrix<real, 3, 2>;
    using Matrix39 = Eigen::Matrix<real, 3, 9>;

    using Matrix3_12 = Eigen::Matrix<real, 3, 12>;
    using Matrix43 = Eigen::Matrix<real, 4, 3>;
    using Matrix49 = Eigen::Matrix<real, 4, 9>;
    using Matrix64 = Eigen::Matrix<real, 6, 4>;
    using Matrix9_27 = Eigen::Matrix<real, 9, 27>;
    using Matrix12_9 = Eigen::Matrix<real, 12, 9>;
    using Matrix27_9 = Eigen::Matrix<real, 27, 9>;
    using MatrixXr = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;
    using Matrix3Xr = Eigen::Matrix<real, 3, Eigen::Dynamic>;
    using Matrix2Xr = Eigen::Matrix<real, 2, Eigen::Dynamic>;
    using Matrix3Xi = Eigen::Matrix<unsigned int, 3, Eigen::Dynamic>;
    using Matrix4Xi = Eigen::Matrix<unsigned int, 4, Eigen::Dynamic>;
    using MatrixXi = Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic>;

    using SparseMatrixXr = Eigen::SparseMatrix<real>;

    /**
     * @brief [eifficient copy] copy the data of a std::vector<glm::vec3> to a Eigen::Matrix<double, 3, Eigen::Dynamic>.
     * @param glm_vectors the source data.
     * @param eigen_matrix the target matrix. It will be resized to the correct size.
     */

    template <typename T, int R, int C, int O, int MR, int MC>
    inline void copyGlmToEigen(
        const std::vector<glm::vec<R, T, glm::defaultp>> &glm_vectors,
        Eigen::Matrix<T, R, C, O, MR, MC> &eigen_matrix)
    {
        const size_t num_vectors = glm_vectors.size();
        eigen_matrix.resize(R, num_vectors);

        if (num_vectors == 0)
            return;

        std::memcpy(
            eigen_matrix.data(),
            glm::value_ptr(glm_vectors[0]),
            num_vectors * R * sizeof(T));
    }

    /**
     * @brief [eifficient copy] copy the data of a Eigen::Matrix<double, 3, Eigen::Dynamic> to a std::vector<glm::vec3>.
     * @param eigen_matrix source matrix.
     * @param glm_vectors target data. It will be resized to the correct size.
     */
    template <typename T, int R, int C, int O, int MR, int MC>
    inline void copyEigenToGlm(
        const Eigen::Matrix<T, R, C, O, MR, MC> &eigen_matrix,
        std::vector<glm::vec<R, T, glm::defaultp>> &glm_vectors)
    {
        const size_t num_vectors = eigen_matrix.cols();
        glm_vectors.resize(num_vectors);

        if (num_vectors == 0)
            return;

        std::memcpy(
            glm::value_ptr(glm_vectors[0]),
            eigen_matrix.data(),
            num_vectors * R * sizeof(T));
    }

    template <typename T>
    inline void copyVecToEigen4X(
        const std::vector<T> &indices,
        Eigen::Matrix<T, 4, Eigen::Dynamic> &eigen_matrix)
    {
        const size_t num_elements = indices.size() / 4;
        ASSERT(indices.size() % 4 == 0, "Index vector size is not a multiple of 4.");

        eigen_matrix.resize(4, num_elements);
        if (num_elements == 0)
            return;

        std::memcpy(
            eigen_matrix.data(),
            indices.data(),
            num_elements * 4 * sizeof(T));
    }

    template <typename T>
    inline void copyVecToEigen3X(
        const std::vector<T> &indices,
        Eigen::Matrix<T, 3, Eigen::Dynamic> &eigen_matrix)
    {
        const size_t num_elements = indices.size() / 3;
        ASSERT(indices.size() % 3 == 0, "Index vector size is not a multiple of 3.");

        eigen_matrix.resize(3, num_elements);
        if (num_elements == 0)
            return;

        std::memcpy(
            eigen_matrix.data(),
            indices.data(),
            num_elements * 3 * sizeof(T));
    }

    template <typename T>
    inline void copyVecToEigen2X(
        const std::vector<T> &indices,
        Eigen::Matrix<T, 2, Eigen::Dynamic> &eigen_matrix)
    {
        const size_t num_elements = indices.size() / 2;
        ASSERT(indices.size() % 2 == 0, "Index vector size is not a multiple of 2.");

        eigen_matrix.resize(2, num_elements);
        if (num_elements == 0)
            return;

        std::memcpy(
            eigen_matrix.data(),
            indices.data(),
            num_elements * 2 * sizeof(T));
    }

    template <typename T>
    inline void copyVecToEigen1X(
        const std::vector<T> &indices,
        Eigen::Matrix<T, Eigen::Dynamic, 1> &eigen_matrix)
    {
        const size_t num_elements = indices.size();
        eigen_matrix.resize(num_elements, 1);
        if (num_elements == 0)
            return;

        std::memcpy(
            eigen_matrix.data(),
            indices.data(),
            num_elements * 1 * sizeof(T));
    }

    inline Eigen::Matrix<double, 3, 3> convertGlmToEigen(const mat3_t &glm_matrix)
    {
        Eigen::Matrix<double, 3, 3> eigen_matrix;
        std::memcpy(eigen_matrix.data(), glm::value_ptr(glm_matrix), 9 * sizeof(double));
        return eigen_matrix;
    }

    inline mat3_t convertEigenToGlm(const Eigen::Matrix<double, 3, 3> &eigen_matrix)
    {
        return glm::make_mat3(eigen_matrix.data());
    }
}
