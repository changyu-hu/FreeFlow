// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include "FSI_Simulator/utils/CudaErrorCheck.cuh"
#include <vector>
#include <numeric> // For std::iota

namespace fsi
{

    template <typename T>
    class CudaArray
    {
    public:
        CudaArray() = default;

        explicit CudaArray(size_t size) : m_size(size)
        {
            if (m_size > 0)
            {
                CUDA_CHECK(cudaMalloc(&d_ptr, byteSize()));
            }
        }

        explicit CudaArray(const std::vector<T> &host_vector) : m_size(host_vector.size())
        {
            if (m_size > 0)
            {
                CUDA_CHECK(cudaMalloc(&d_ptr, byteSize()));
                upload(host_vector);
            }
        }

        ~CudaArray()
        {
            free();
        }

        CudaArray(CudaArray &&other) noexcept
            : d_ptr(other.d_ptr), m_size(other.m_size)
        {
            other.d_ptr = nullptr;
            other.m_size = 0;
        }

        CudaArray &operator=(CudaArray &&other) noexcept
        {
            if (this != &other)
            {
                free();
                d_ptr = other.d_ptr;
                m_size = other.m_size;
                other.d_ptr = nullptr;
                other.m_size = 0;
            }
            return *this;
        }

        CudaArray(const CudaArray &) = delete;
        CudaArray &operator=(const CudaArray &) = delete;

        void resize(size_t new_size)
        {
            if (m_size == new_size)
            {
                return;
            }
            free();
            m_size = new_size;
            if (m_size > 0)
            {
                CUDA_CHECK(cudaMalloc(&d_ptr, byteSize()));
            }
        }

        void upload(const std::vector<T> &host_vector)
        {
            ASSERT(host_vector.size() == m_size, "Host vector size {} does not match CudaArray size {}.", host_vector.size(), m_size);
            if (m_size > 0)
            {
                CUDA_CHECK(cudaMemcpy(d_ptr, host_vector.data(), byteSize(), cudaMemcpyHostToDevice));
            }
        }

        void uploadAsync(const std::vector<T> &host_vector, cudaStream_t stream)
        {
            ASSERT(host_vector.size() == m_size, "Host vector size {} does not match CudaArray size {}.", host_vector.size(), m_size);
            if (m_size > 0)
            {
                CUDA_CHECK(cudaMemcpyAsync(d_ptr, host_vector.data(), byteSize(), cudaMemcpyHostToDevice, stream));
            }
        }

        std::vector<T> download() const
        {
            std::vector<T> host_vector(m_size);
            if (m_size > 0)
            {
                CUDA_CHECK(cudaMemcpy(host_vector.data(), d_ptr, byteSize(), cudaMemcpyDeviceToHost));
            }
            return host_vector;
        }

        void download(std::vector<T> &host_vector) const
        {
            ASSERT(host_vector.size() == m_size, "Host vector size {} does not match CudaArray size {}.", host_vector.size(), m_size);
            if (m_size > 0)
            {
                CUDA_CHECK(cudaMemcpy(host_vector.data(), d_ptr, byteSize(), cudaMemcpyDeviceToHost));
            }
        }

        std::vector<glm::vec<3, T>> downloadToVec3() const
        {
            ASSERT(m_size % 3 == 0, "Array size is not a multiple of 3, cannot convert to vec3.");
            std::vector<T> flat_data = download();
            std::vector<glm::vec<3, T>> vec_data(m_size / 3);
            for (size_t i = 0; i < vec_data.size(); ++i)
            {
                vec_data[i] = glm::vec<3, T>(
                    flat_data[i * 3 + 0],
                    flat_data[i * 3 + 1],
                    flat_data[i * 3 + 2]);
            }
            return vec_data;
        }

        void copyFrom(const CudaArray<T> &other)
        {
            resize(other.size());
            if (m_size > 0)
            {
                CUDA_CHECK(cudaMemcpy(d_ptr, other.d_ptr, byteSize(), cudaMemcpyDeviceToDevice));
            }
        }

        void copyFromAsync(const CudaArray<T> &other, cudaStream_t stream)
        {
            resize(other.size());
            if (m_size > 0)
            {
                CUDA_CHECK(cudaMemcpyAsync(d_ptr, other.d_ptr, byteSize(), cudaMemcpyDeviceToDevice, stream));
            }
        }

        void setZero()
        {
            if (m_size > 0)
            {
                CUDA_CHECK(cudaMemset(d_ptr, 0, byteSize()));
            }
        }

        void setZeroAsync(cudaStream_t stream)
        {
            if (m_size > 0)
            {
                CUDA_CHECK(cudaMemsetAsync(d_ptr, 0, byteSize(), stream));
            }
        }

        T *data() { return d_ptr; }
        const T *data() const { return d_ptr; }

        size_t size() const { return m_size; }

        size_t byteSize() const { return m_size * sizeof(T); }

        bool empty() const { return m_size == 0; }

    private:
        void free()
        {
            if (d_ptr != nullptr)
            {
                cudaFree(d_ptr);
                d_ptr = nullptr;
                m_size = 0;
            }
        }

        T *d_ptr = nullptr;
        size_t m_size = 0;
    };

} // namespace fsi