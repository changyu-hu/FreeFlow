// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once
#include "FSI_Simulator/common/CudaCommon.cuh"
#include "FSI_Simulator/utils/NumericUtils.cuh"

namespace fsi {
// --- 通用 IntersectionInfo 结构体 ---
struct IntersectionInfo {
    float hit_point[3]; // 总是存储3D坐标，2D时z=0
    float depth;
    int element_index;
    float barycentric_coords[3]; // 三角形/四面体的重心坐标
};

// --- 主模板 (保持未定义或为空) ---
// 如果有人尝试实例化 SolidGeometryProxy_Device<4> 等不支持的维度，会导致编译错误
template <int DIM>
struct SolidGeometryProxy_Device;


// =========================================================================
// --- 2D 特化版本 (DIM = 2) ---
// =========================================================================
template <>
struct SolidGeometryProxy_Device<2> {
    // --- 成员数据 ---
    const float* vertices;   // (x1,y1, x2,y2, ...)
    const float* velocities; // (vx1,vy1, ...)
    const unsigned int* elements;   // 2D元素 (如线段) (v1,v2, v3,v4, ...)
    float* forces;     // (fx1,fy1, ...)

    int num_vertices;
    int num_elements;

    // --- 设备端方法 (2D 专用接口) ---

    __device__ bool isInside(glm::vec2 p) const;
    __device__ bool intersect(glm::vec2 p1, glm::vec2 p2, IntersectionInfo& info) const;
    __device__ void getVelocityAt(const IntersectionInfo& info, glm::vec2& v) const;
    __device__ void addForceAt(const IntersectionInfo& info, glm::vec2 F);
    __device__ void getBoundingBox(int idx, glm::vec2& min, glm::vec2& max) const;
    __device__ void intersectAtIndex(glm::vec2 p1, glm::vec2 p2, IntersectionInfo& info) const;
};


// =========================================================================
// --- 3D 特化版本 (DIM = 3) ---
// =========================================================================
template <>
struct SolidGeometryProxy_Device<3> {
    // --- 成员数据 ---
    const float* vertices;   // (x1,y1,z1, ...)
    const float* velocities; // (vx1,vy1,vz1, ...)
    const unsigned int* elements;   // 3D元素 (如三角形、四面体) (v1,v2,v3, ...)
    float* forces;     // (fx1,fy1,fz1, ...)

    int num_vertices;
    int num_elements;

    // --- 设备端方法 (3D 专用接口) ---

    __device__ bool isInside(glm::vec3 p) const;
    __device__ void intersect(glm::vec3 p1, glm::vec3 p2, IntersectionInfo& info) const;
    __device__ void intersectAtIndex(glm::vec3 p1, glm::vec3 p2, IntersectionInfo& info) const;
    __device__ void getVelocityAt(const IntersectionInfo& info, glm::vec3& v) const;
    __device__ void addForceAt(const IntersectionInfo& info, glm::vec3 F);
    __device__ void getBoundingBox(int idx, glm::vec3& min, glm::vec3& max) const;
};

// --- 设备端函数实现 ---
// (将实现放在这里，因为它们是 __device__ inline 函数)

// --- 深度索引压缩 ---
inline __device__ uint32_t depthIndexCompact(float depth, int index)
{
	uint32_t ret = 0;
	ret |= ((uint32_t)(depth * (1<<15))) << 16;
	ret |= (uint16_t)(index);
	return ret;
}

// --- 2D 实现 ---

__device__ inline float line_intersect(float x1, float y1, float x2, float y2,
                   float x3, float y3, float x4, float y4, float &xo, float &yo)
{
    // float a1 = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    // float b1 = (x4 - x2) * (y3 - y4) - (y4 - y2) * (x3 - x4);
    // if (NumericUtils::isApproxLessEqual(fabs(a1), 0.0f))
    //     return -1.0f;
    // float t1 = b1 / a1;
    // if (NumericUtils::isApproxLess(t1, 0.0f) || NumericUtils::isApproxGreater(t1, 1.0f))
    //     return -1.0f;
    // xo = t1 * x1 + (1.0f - t1) * x2;
    // yo = t1 * y1 + (1.0f - t1) * y2;
    // float mark = ((x3 - xo) * (x4 - xo) + (y3 - yo) * (y4 - yo)) / ((x3 - x4) * (x3 - x4) + (y3 - y4) * (y3 - y4));
    // if (NumericUtils::isApproxLessEqual(mark, 0.0f))
    // {
    //     return 1.0f - t1;
    // }
    // else
    //     return -1.0f;

    float a1 = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    float b1 = (x4 - x2) * (y3 - y4) - (y4 - y2) * (x3 - x4);
    if (abs(a1) <= 0.)
        return -1.0;
    float t1 = b1 / a1;
    if (t1 <= 0.0 || t1 >= 1.0)
        return -1.0;
    xo = t1 * x1 + (1 - t1) * x2;
    if ((x3 - xo) * (x4 - xo) < 0.0)
    {
        yo = t1 * y1 + (1 - t1) * y2;
        return 1.0 - t1;
    }
    else
        return -1.0;
}

__device__ inline bool SolidGeometryProxy_Device<2>::isInside(glm::vec2 p) const {
    bool isInPolygon = false;
    for (size_t i = 0; i < num_elements; i++)
    {
        int p1 = elements[2 * i], p2 = elements[2 * i + 1];
        glm::vec2 s = glm::vec2(vertices[2 * p1], vertices[2 * p1 + 1]);
        glm::vec2 t = glm::vec2(vertices[2 * p2], vertices[2 * p2 + 1]);

        /*if (sy == ty) {
            isInPolygon = (py == sy && (px >= sx && px <= tx || px <= sx && px >= tx));
        }*/
        if ((s[1] < p[1] && t[1] >= p[1]) || (s[1] >= p[1] && t[1] < p[1]))
        {
            isInPolygon ^= (s[0] + (p[1] - s[1]) * (t[0] - s[0]) / (t[1] - s[1]) > p[0]);
        }
    }

    return isInPolygon;
}

__device__ inline bool SolidGeometryProxy_Device<2>::intersect(glm::vec2 p1, glm::vec2 p2, IntersectionInfo& info) const {
    float depth = 2;
    float xo, yo;
    bool isIntersection = false;
    // printf("fuck.\n");
    for (size_t i = 0; i < num_elements; i++)
    {
        // printf("fuck2.\n");
        int s = elements[2 * i], t = elements[2 * i + 1];
        glm::vec2 p3 = glm::vec2(vertices[2 * s], vertices[2 * s + 1]);
        glm::vec2 p4 = glm::vec2(vertices[2 * t], vertices[2 * t + 1]);
        float d = line_intersect(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1], xo, yo);
        if (d > 0 && d < depth)
        {
            info.hit_point[0] = xo;
            info.hit_point[1] = yo;
            info.element_index = i;
            info.barycentric_coords[0] = 1.0f - d;
            info.barycentric_coords[1] = d;
            depth = d;
            isIntersection = true;
        }
    }

    return isIntersection;
}

__device__ inline void SolidGeometryProxy_Device<2>::getVelocityAt(const IntersectionInfo& info, glm::vec2& v) const {
    // 根据info中的重心坐标和元素索引，插值速度
    int idx = info.element_index;
    int v1 = elements[2 * idx];
    int v2 = elements[2 * idx + 1];
    v[0] = velocities[2 * v1] * info.barycentric_coords[0] + velocities[2 * v2] * info.barycentric_coords[1];
    v[1] = velocities[2 * v1 + 1] * info.barycentric_coords[0] + velocities[2 * v2 + 1] * info.barycentric_coords[1];
}

#if defined(__CUDA_ARCH__)

__device__ inline void SolidGeometryProxy_Device<2>::addForceAt(const IntersectionInfo& info, glm::vec2 F) {
    // 根据info中的重心坐标和元素索引，将力分配到节点上
    // 注意：这里需要原子操作！
    int idx = info.element_index;
    int v1 = elements[2 * idx];
    int v2 = elements[2 * idx + 1];
    atomicAdd(&forces[2 * v1], F[0] * info.barycentric_coords[0]);
    atomicAdd(&forces[2 * v1 + 1], F[1] * info.barycentric_coords[0]);
    atomicAdd(&forces[2 * v2], F[0] * info.barycentric_coords[1]);
    atomicAdd(&forces[2 * v2 + 1], F[1] * info.barycentric_coords[1]);
}

#endif

__device__ inline void SolidGeometryProxy_Device<2>::getBoundingBox(int idx, glm::vec2& min, glm::vec2& max) const {
    min = glm::vec2(FLT_MAX, FLT_MAX);
    max = glm::vec2(FLT_MIN, FLT_MIN);
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 2; j++)
        {
            min[i] = glm::min(min[i], vertices[2 * elements[2 * idx + j] + i]);
            max[i] = glm::max(max[i], vertices[2 * elements[2 * idx + j] + i]);
        }
    }
}

__device__ inline void SolidGeometryProxy_Device<2>::intersectAtIndex(glm::vec2 p1, glm::vec2 p2, IntersectionInfo& info) const {
    int idx = info.element_index;
    glm::vec2 p3 = glm::vec2(vertices[2 * elements[2 * idx]], vertices[2 * elements[2 * idx] + 1]);
    glm::vec2 p4 = glm::vec2(vertices[2 * elements[2 * idx + 1]], vertices[2 * elements[2 * idx + 1] + 1]);
    auto d = line_intersect(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1], info.hit_point[0], info.hit_point[1]);
    if (d > 0 && d < 1)
    {
        info.element_index = idx;
        info.depth = d;
        info.barycentric_coords[0] = glm::distance(glm::vec2(info.hit_point[0], info.hit_point[1]), p4) / glm::distance(p3, p4);
        info.barycentric_coords[1] = 1 - info.barycentric_coords[0];
    }
    else
    {
        info.element_index = -1;
    }
}


// --- 3D 实现 ---
__device__ inline bool SolidGeometryProxy_Device<3>::isInside(glm::vec3 p) const {
    bool isInPolygon = false;
    glm::vec3 o{0, 0, 0};
    for (size_t i = 0; i < num_elements; i++)
    {
        glm::vec3 v[3] = {};
        for (size_t j = 0; j < 3; j++)
        {
            for (size_t k = 0; k < 3; k++)
            {
                v[j][k] = vertices[elements[3 * i + j] * 3 + k];
            }
        }
        glm::vec3 n = glm::cross(v[1] - v[0], v[2] - v[0]);
        // check if the line is parallel to the plane
        glm::vec3 d = glm::normalize(p - o);
        if (NumericUtils::isApproxLessEqual(fabs(glm::dot(n, d)), 0.0f))
        {
            // TODO: check if point is in the plane
            continue;
        }

        float t = glm::dot(n, v[0] - p) / glm::dot(n, o - p);
        if (NumericUtils::isApproxLessEqual(t, 0.0f) || NumericUtils::isApproxGreater(t, 1.0f))
            continue;

        // check if the intersection point is inside the triangle
        glm::vec3 hitP = p + t * (o - p);
        bool sameSide = true;
        for (int j = 0; j < 3; j++)
        {
            glm::vec3 e[3] = {};
            for (size_t k = 0; k < 3; k++)
            {
                for (size_t l = 0; l < 3; l++)
                {
                    e[k][l] = vertices[elements[3 * i + (j + k) % 3] * 3 + l];
                }
            }

            glm::vec3 v1 = glm::cross(e[1] - e[0], e[2] - e[0]);
            float vn = glm::length(v1);
            v1 = v1 / vn;
            glm::vec3 v2 = glm::cross(e[1] - e[0], hitP - e[0]) / vn;

            if (NumericUtils::isApproxLessEqual(glm::dot(v1, v2), 0.0f))
            {
                sameSide = false;
                break;
            }
        }
        if (sameSide)
        {
            isInPolygon = !isInPolygon;
        }
    }

    return isInPolygon;
}

__device__ inline void barycentric_coordinates(glm::vec3 point, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float &u, float &v)
{
    auto edge1 = v1 - v0;
    auto edge2 = v2 - v0;
    auto w = point - v0;
    float d00 = glm::dot(edge1, edge1);
    float d01 = glm::dot(edge1, edge2);
    float d11 = glm::dot(edge2, edge2);
    float d20 = glm::dot(w, edge1);
    float d21 = glm::dot(w, edge2);
    float denominator = d00 * d11 - d01 * d01;
    u = (d11 * d20 - d01 * d21) / denominator;
    v = (d00 * d21 - d01 * d20) / denominator;
}

__device__ inline bool line_triangle_intersect(glm::vec3 p1, glm::vec3 p2, glm::vec3 v[3], IntersectionInfo& info)
{
    glm::vec3 n = glm::normalize(glm::cross(v[1] - v[0], v[2] - v[0]));
    glm::vec3 d = glm::normalize(p2 - p1);
    // check if the line is parallel to the plane
    if (NumericUtils::isApproxZero(glm::dot(n, d)))
    {
        info.element_index = -1;
        return false;
    }
    float t = glm::dot(n, v[0] - p1) / glm::dot(n, p2 - p1);
    if (t < 0. || t > 1.) //(NumericUtils::isApproxLess(t, 0.0f) || NumericUtils::isApproxGreater(t, 1.0f))
    {
        info.element_index = -1;
        return false;
    }
    float u0, v0;
    glm::vec3 intersection_point = p1 + t * (p2 - p1);
    barycentric_coordinates(intersection_point, v[0], v[1], v[2], u0, v0);
    if (NumericUtils::isApproxGreaterEqual(u0, 0.0f) && NumericUtils::isApproxGreaterEqual(v0, 0.0f) && NumericUtils::isApproxLessEqual(u0 + v0, 1.0f))
    {
        // if (info.element_index == -1 || t < info.depth)
        // {
        info.hit_point[0] = intersection_point.x;
        info.hit_point[1] = intersection_point.y;
        info.hit_point[2] = intersection_point.z;
        info.depth = t;
        info.barycentric_coords[0] = 1.0f - u0 - v0;
        info.barycentric_coords[1] = u0;
        info.barycentric_coords[2] = v0;
        return true;
        // }
    }
    info.element_index = -1;
    return false;
}

// WARNING: This function is wrong. DO NOT USE IT.
__device__ inline void SolidGeometryProxy_Device<3>::intersect(glm::vec3 p1, glm::vec3 p2, IntersectionInfo& info) const {
    info.element_index = -1;
    info.depth = 2;
    for (size_t i = 0; i < num_elements; i++)
    {
        glm::vec3 v[3] = {};
        for (size_t j = 0; j < 3; j++)
        {
            for (size_t k = 0; k < 3; k++)
            {
                v[j][k] = vertices[elements[3 * i + j] * 3 + k];
            }
        }
        if (line_triangle_intersect(p1, p2, v, info))
        {
            info.element_index = i;
        }
    }
}

__device__ inline void SolidGeometryProxy_Device<3>::intersectAtIndex(glm::vec3 p1, glm::vec3 p2, IntersectionInfo& info) const {
    int idx = info.element_index;
    if (idx < 0 || idx >= num_elements)
    {
        printf("Error: idx out of range in intersectAtIndex: %d\n", idx);
        info.element_index = -1;
        return;
    }
    

    glm::vec3 v[3] = {};
    for (size_t j = 0; j < 3; j++)
    {
        for (size_t k = 0; k < 3; k++)
        {
            v[j][k] = vertices[elements[3 * idx + j] * 3 + k];
        }
    }
    line_triangle_intersect(p1, p2, v, info);
}

__device__ inline void SolidGeometryProxy_Device<3>::getVelocityAt(const IntersectionInfo& info, glm::vec3& v) const {
    int idx = info.element_index;
    if (idx < 0 || idx >= num_elements)
    {
        printf("Error: idx out of range in getVelocityAt: %d\n", idx);
        return;
    }
    int v1 = elements[3 * idx];
    int v2 = elements[3 * idx + 1];
    int v3 = elements[3 * idx + 2];
    v[0] = velocities[3 * v1] * info.barycentric_coords[0] + velocities[3 * v2] * info.barycentric_coords[1] + velocities[3 * v3] * info.barycentric_coords[2];
    v[1] = velocities[3 * v1 + 1] * info.barycentric_coords[0] + velocities[3 * v2 + 1] * info.barycentric_coords[1] + velocities[3 * v3 + 1] * info.barycentric_coords[2];
    v[2] = velocities[3 * v1 + 2] * info.barycentric_coords[0] + velocities[3 * v2 + 2] * info.barycentric_coords[1] + velocities[3 * v3 + 2] * info.barycentric_coords[2];
}

#if defined(__CUDA_ARCH__)
__device__ inline void SolidGeometryProxy_Device<3>::addForceAt(const IntersectionInfo& info, glm::vec3 F) {
    int idx = info.element_index;
    if (idx < 0 || idx >= num_elements)
    {
        printf("Error: idx out of range in addForceAt: %d\n", idx);
        return;
    }
    int v1 = elements[3 * idx];
    int v2 = elements[3 * idx + 1];
    int v3 = elements[3 * idx + 2];
    // if (glm::length(F) > 1.0)
    // {
    //     printf("idx: %d, v1: %d, v2: %d, v3: %d, F: %f %f %f\n", idx, v1, v2, v3, F[0], F[1], F[2]);
    // }
    for (size_t i = 0; i < 3; i++)
    {
        atomicAdd(&forces[3 * v1 + i], F[i] * info.barycentric_coords[0]);
        atomicAdd(&forces[3 * v2 + i], F[i] * info.barycentric_coords[1]);
        atomicAdd(&forces[3 * v3 + i], F[i] * info.barycentric_coords[2]);
    }
}
#endif

__device__ inline void SolidGeometryProxy_Device<3>::getBoundingBox(int idx, glm::vec3& min, glm::vec3& max) const {
    if (idx < 0 || idx >= num_elements)
    {
        printf("Error: idx out of range in getBoundingBox: %d\n", idx);
        return;
    }
    min = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    max = glm::vec3(FLT_MIN, FLT_MIN, FLT_MIN);
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            min[j] = glm::min(min[j], vertices[elements[3 * idx + i] * 3 + j]);
            max[j] = glm::max(max[j], vertices[elements[3 * idx + i] * 3 + j]);
        }
    }
}
} // namespace fsi