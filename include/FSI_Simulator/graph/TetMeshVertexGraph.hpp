// Copyright 2023 Anka He Chen
// 
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. 
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, 
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions 
// and limitations under the License.

#pragma once
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include "FSI_Simulator/graph/Graph.hpp"

namespace GAIA
{
	namespace GraphColoring
	{
		struct TetMeshVertexGraph : Graph
		{
			void fromMesh2d(unsigned int *trivIdx, int vNum, int triNum)
			{
				numNodes = vNum;
				edges.clear();
				std::set<Edge> edgeSet;
				for (int triidx = 0; triidx < triNum; triidx++)
				{
					for (int i = 0; i < 3; i++)
					{
						Edge e;
						if (trivIdx[3 * triidx + i] < trivIdx[3 * triidx + (i + 1) % 3])
						{
							e = {trivIdx[3 * triidx + i], trivIdx[3 * triidx + (i + 1) % 3]};
						}
						else
						{
							e = {trivIdx[3 * triidx + (i + 1) % 3], trivIdx[3 * triidx + i]};
						}
						edgeSet.insert(e);
					}
				}

				for (auto &e : edgeSet)
				{
					edges.push_back(e);
				}
			}

			void fromCoMesh2d(unsigned int *trivIdx, unsigned int *rodvIdx, int vNum, int triNum, int rodNum)
			{
				numNodes = vNum;
				edges.clear();
				std::set<Edge> edgeSet;
				for (int triidx = 0; triidx < triNum; triidx++)
				{
					for (int i = 0; i < 3; i++)
					{
						Edge e;
						if (trivIdx[3 * triidx + i] < trivIdx[3 * triidx + (i + 1) % 3])
						{
							e = {trivIdx[3 * triidx + i], trivIdx[3 * triidx + (i + 1) % 3]};
						}
						else
						{
							e = {trivIdx[3 * triidx + (i + 1) % 3], trivIdx[3 * triidx + i]};
						}
						edgeSet.insert(e);
					}
				}

				for (int rodidx = 0; rodidx < rodNum; rodidx++)
				{
					Edge e;
					if (rodvIdx[2 * rodidx] < rodvIdx[2 * rodidx + 1])
					{
						e = {rodvIdx[2 * rodidx], rodvIdx[2 * rodidx + 1]};
					}
					else
					{
						e = {rodvIdx[2 * rodidx + 1], rodvIdx[2 * rodidx]};
					}
					edgeSet.insert(e);
				}

				for (auto &e : edgeSet)
				{
					edges.push_back(e);
				}
			}

			void fromCoMesh3d(unsigned int *tetvIdx, unsigned int *trivIdx, int vNum, int tetNum, int triNum)
			{
				numNodes = vNum;
				edges.clear();
				std::set<Edge> edgeSet;
				for (int triidx = 0; triidx < triNum; triidx++)
				{
					for (int i = 0; i < 3; i++)
					{
						Edge e;
						if (trivIdx[3 * triidx + i] < trivIdx[3 * triidx + (i + 1) % 3])
						{
							e = {trivIdx[3 * triidx + i], trivIdx[3 * triidx + (i + 1) % 3]};
						}
						else
						{
							e = {trivIdx[3 * triidx + (i + 1) % 3], trivIdx[3 * triidx + i]};
						}
						edgeSet.insert(e);
					}
				}

				for (int tetidx = 0; tetidx < tetNum; tetidx++)
				{
					for (int i = 0; i < 3; i++)
					{
						for (int j = i + 1; j < 4; j++)
						{
							Edge e;
							if (tetvIdx[4 * tetidx + i] < tetvIdx[4 * tetidx + j])
							{
								e = {tetvIdx[4 * tetidx + i], tetvIdx[4 * tetidx + j]};
							}
							else
							{
								e = {tetvIdx[4 * tetidx + j], tetvIdx[4 * tetidx + i]};
							}
							edgeSet.insert(e);
						}
					}
				}

				for (auto &e : edgeSet)
				{
					edges.push_back(e);
				}
			}
		};

	}
}