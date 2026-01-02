// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include "FSI_Simulator/control/LBSSampling.hpp"
#include "FSI_Simulator/utils/Logger.hpp"
#include <random>
#include <algorithm>
#include <queue>

namespace fsi
{
    namespace control
    {

        using PrioQueueNode = std::pair<real, int>; // <distance, vertex_index>

        /**
         * @brief adjacency list from tet mesh
         * @param vnum vertex number.
         * @param vpos vertex position array.
         * @param tetvIdx tet index array.
         * @return adjacency list. adj[i] = { {neighbor1, dist1}, {neighbor2, dist2}, ... }
         */
        std::vector<std::vector<std::pair<int, real>>> build_adjacency_list(
            int vnum,
            const std::vector<vec3_t> &vpos,
            const std::vector<unsigned int> &tetvIdx)
        {
            std::vector<std::vector<std::pair<int, real>>> adj(vnum);
            int tet_num = tetvIdx.size() / 4;

            for (int i = 0; i < tet_num; ++i)
            {
                const unsigned int *tet = &tetvIdx[i * 4];
                const int edges[6][2] = {
                    {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

                for (const auto &edge : edges)
                {
                    int u_idx = tet[edge[0]];
                    int v_idx = tet[edge[1]];

                    real weight = glm::distance(vpos[u_idx], vpos[v_idx]);

                    adj[u_idx].push_back({v_idx, weight});
                    adj[v_idx].push_back({u_idx, weight});
                }
            }

            for (int i = 0; i < vnum; ++i)
            {
                std::sort(adj[i].begin(), adj[i].end());
                adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
            }

            return adj;
        }

        std::vector<std::vector<std::pair<int, real>>> build_adjacency_list(
            int vnum,
            const std::vector<vec2_t> &vpos,
            const std::vector<unsigned int> &trivIdx)
        {
            std::vector<std::vector<std::pair<int, real>>> adj(vnum);
            int tri_num = trivIdx.size() / 3;

            for (int i = 0; i < tri_num; ++i)
            {
                const unsigned int *tri = &trivIdx[i * 3];
                const int edges[3][2] = {
                    {0, 1},
                    {0, 2},
                    {1, 2},
                };

                for (const auto &edge : edges)
                {
                    int u_idx = tri[edge[0]];
                    int v_idx = tri[edge[1]];

                    real weight = glm::distance(vpos[u_idx], vpos[v_idx]);

                    adj[u_idx].push_back({v_idx, weight});
                    adj[v_idx].push_back({u_idx, weight});
                }
            }

            for (int i = 0; i < vnum; ++i)
            {
                std::sort(adj[i].begin(), adj[i].end());
                adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
            }

            return adj;
        }

        /**
         * @brief use Dijkstra algorithm to calculate the geodesic distance from a single source point to all other points.
         * @param start_node_idx index of the source point.
         * @param vnum total number of vertices.
         * @param adj adjacency list. adj[i] = { {neighbor1, dist1}, {neighbor2, dist2}, ... }
         * @return a vector containing the geodesic distance from the source point to each vertex.
         */
        std::vector<real> dijkstra(
            int start_node_idx,
            int vnum,
            const std::vector<std::vector<std::pair<int, real>>> &adj)
        {
            std::vector<real> dist(vnum, std::numeric_limits<real>::max());

            // priority queue, store <distance, vertex_index>, use std::greater to make it a min heap
            std::priority_queue<PrioQueueNode, std::vector<PrioQueueNode>, std::greater<PrioQueueNode>> pq;

            // initialize source point
            dist[start_node_idx] = 0.0;
            pq.push({0.0, start_node_idx});

            while (!pq.empty())
            {
                real d = pq.top().first;
                int u = pq.top().second;
                pq.pop();

                // if a shorter path is found, skip
                if (d > dist[u])
                {
                    continue;
                }

                // traverse all neighbors
                for (const auto &edge : adj[u])
                {
                    int v = edge.first;
                    real weight = edge.second;

                    // relax operation
                    if (dist[u] + weight < dist[v])
                    {
                        dist[v] = dist[u] + weight;
                        pq.push({dist[v], v});
                    }
                }
            }
            return dist;
        }

        void farthest_point_sampling(
            int vnum, const std::vector<vec3_t> &vpos, const std::vector<unsigned int> &tetvIdx,
            int cnum, std::vector<int> &v_ctrl, std::vector<real> &lbs_dist,
            std::string lbs_distance_type,
            bool random_first)
        {
            ASSERT(cnum > 0 && cnum <= vnum, "Number of control points is invalid.");
            v_ctrl.resize(cnum);
            lbs_dist.resize(cnum * vnum);

            // --- 1. choose the first control point ---
            int first_point_idx = -1;
            if (random_first)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, vnum - 1);
                first_point_idx = dis(gen);
            }
            else
            { // max x distance
                auto max_x_iter = std::max_element(vpos.begin(), vpos.end(),
                                                   [](const vec3_t &a, const vec3_t &b)
                                                   { return a.x < b.x; });
                first_point_idx = std::distance(vpos.begin(), max_x_iter);
            }
            v_ctrl[0] = first_point_idx;

            // --- 2. iterate to choose remaining control points ---
            std::vector<real> dist(vnum, std::numeric_limits<real>::max());

            // pre-build adjacency list if needed
            std::vector<std::vector<std::pair<int, real>>> adj;
            if (lbs_distance_type == "geodesic")
            {
                LOG_INFO("Building adjacency list for geodesic distance calculation...");
                adj = build_adjacency_list(vnum, vpos, tetvIdx);
                LOG_INFO("Adjacency list built.");
            }

            int last_selected_idx = first_point_idx;

            for (int i = 1; i <= cnum; i++)
            {

                std::vector<real> current_dist;
                if (lbs_distance_type == "euclidean")
                {
                    for (int j = 0; j < vnum; j++)
                    {
                        real d = glm::distance(vpos[j], vpos[v_ctrl[i - 1]]);
                        lbs_dist[j * cnum + i - 1] = d;
                        if (d < dist[j])
                        {
                            dist[j] = d;
                        }
                    }
                }
                else if (lbs_distance_type == "geodesic")
                {
                    // --- use Dijkstra to calculate geodesic distance ---
                    LOG_INFO("Calculating geodesic distances from point {}...", i - 1);
                    current_dist = dijkstra(last_selected_idx, vnum, adj);

                    // update each point to "selected point set" minimum geodesic distance
                    for (int j = 0; j < vnum; j++)
                    {
                        lbs_dist[j * cnum + i - 1] = current_dist[j];
                        if (current_dist[j] < dist[j])
                        {
                            dist[j] = current_dist[j];
                        }
                    }
                }
                else
                {
                    ASSERT(false, "Unknown LBS distance type: {}", lbs_distance_type);
                }

                if (i == cnum)
                {
                    break;
                }

                // --- b. find the point with maximum distance ---
                auto max_dist_iter = std::max_element(dist.begin(), dist.end());
                int max_dist_idx = std::distance(dist.begin(), max_dist_iter);

                // --- c. select this point as new control point ---
                v_ctrl[i] = max_dist_idx;
                last_selected_idx = max_dist_idx;

                dist[max_dist_idx] = 0.0;

                LOG_INFO("FPS iteration {}: Selected point {} as new control point.", i, max_dist_idx);
            }
        }

        void farthest_point_sampling(
            int vnum, const std::vector<vec2_t> &vpos, const std::vector<unsigned int> &trivIdx,
            int cnum, std::vector<int> &v_ctrl, std::vector<real> &lbs_dist,
            std::string lbs_distance_type,
            bool random_first)
        {
            ASSERT(cnum > 0 && cnum <= vnum, "Number of control points is invalid.");
            v_ctrl.resize(cnum);
            lbs_dist.resize(cnum * vnum);

            int first_point_idx = -1;
            if (random_first)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, vnum - 1);
                first_point_idx = dis(gen);
            }
            else
            { // max x distance
                auto max_x_iter = std::max_element(vpos.begin(), vpos.end(),
                                                   [](const vec2_t &a, const vec2_t &b)
                                                   { return a.x < b.x; });
                first_point_idx = std::distance(vpos.begin(), max_x_iter);
            }
            v_ctrl[0] = first_point_idx;

            std::vector<real> dist(vnum, std::numeric_limits<real>::max());

            std::vector<std::vector<std::pair<int, real>>> adj;
            if (lbs_distance_type == "geodesic")
            {
                LOG_INFO("Building adjacency list for geodesic distance calculation...");
                adj = build_adjacency_list(vnum, vpos, trivIdx);
                LOG_INFO("Adjacency list built.");
            }

            int last_selected_idx = first_point_idx;

            for (int i = 1; i <= cnum; i++)
            {

                std::vector<real> current_dist;
                if (lbs_distance_type == "euclidean")
                {
                    for (int j = 0; j < vnum; j++)
                    {
                        real d = glm::distance(vpos[j], vpos[v_ctrl[i - 1]]);
                        lbs_dist[j * cnum + i - 1] = d;
                        if (d < dist[j])
                        {
                            dist[j] = d;
                        }
                    }
                }
                else if (lbs_distance_type == "geodesic")
                {
                    LOG_INFO("Calculating geodesic distances from point {}...", i - 1);
                    current_dist = dijkstra(last_selected_idx, vnum, adj);

                    for (int j = 0; j < vnum; j++)
                    {
                        lbs_dist[j * cnum + i - 1] = current_dist[j];
                        if (current_dist[j] < dist[j])
                        {
                            dist[j] = current_dist[j];
                        }
                    }
                }
                else
                {
                    ASSERT(false, "Unknown LBS distance type: {}", lbs_distance_type);
                }

                if (i == cnum)
                {
                    break;
                }

                auto max_dist_iter = std::max_element(dist.begin(), dist.end());
                int max_dist_idx = std::distance(dist.begin(), max_dist_iter);

                v_ctrl[i] = max_dist_idx;
                last_selected_idx = max_dist_idx;

                dist[max_dist_idx] = 0.0;

                LOG_INFO("FPS iteration {}: Selected point {} as new control point.", i, max_dist_idx);
            }
        }

    }
}