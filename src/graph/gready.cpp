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

#include "FSI_Simulator/graph/Gready.hpp"
#include <iostream>
#include <algorithm>
#include <set>
#include <queue>
#include <list>
#include <chrono> // std::chrono::system_clock
#include <random>

using std::cerr;
using std::cout;
using std::endl;
using std::queue;
using std::vector;
int GAIA::GraphColoring::OrderedGreedy::nextNode()
{
	int nodeMinDegrees = -1;
	int minDegree = graph.size() + 1;
	for (size_t iNode = 0; iNode < degrees.size(); iNode++)
	{
		if (degrees[iNode] == -1)
		{
			continue;
		}
		if (minDegree > degrees[iNode])
		{
			minDegree = degrees[iNode];
			nodeMinDegrees = iNode;
		}
	}
	return nodeMinDegrees;
}

void GAIA::GraphColoring::OrderedGreedy::reduceDegree(int iNode)
{
	degrees[iNode] = -1;
	for (size_t iNei = 0; iNei < graph[iNode].size(); iNei++)
	{
		int neiId = graph[iNode][iNei];

		if (degrees[neiId] != -1)
		{
			degrees[neiId]--;
		}
	}
}

vector<int> &GAIA::GraphColoring::OrderedGreedy::color()
{
	// initialize the degree
	degrees.clear();
	degrees.resize(graph.size(), 0);
	for (int iNode = 0; iNode < this->graph.size(); iNode++)
	{
		degrees[iNode] = graph[iNode].size();
		graph_colors[iNode] = -1;
	}

	// greedy coloring

	int maxColor = -1;
	int nColored = 0;
	std::vector<bool> colorUsed;
	colorUsed.reserve(128);

	while (nColored < graph.size())
	{
		int node = nextNode();

		// first one
		if (maxColor == -1)
		{
			++maxColor;
			graph_colors[node] = maxColor;
		}
		else
		{
			colorUsed.resize(maxColor + 1);

			for (int iColor = 0; iColor < colorUsed.size(); iColor++)
			{
				colorUsed[iColor] = false;
			}

			// see its neighbor's color
			for (int iNei = 0; iNei < graph[node].size(); iNei++)
			{
				int neiId = graph[node][iNei];
				if (graph_colors[neiId] >= 0)
				{
					colorUsed[graph_colors[neiId]] = true;
				}
			}

			// find the minimal usable color
			int minUsableColor = -1;
			for (int iColor = 0; iColor < colorUsed.size(); iColor++)
			{
				if (!colorUsed[iColor])
				{
					minUsableColor = iColor;
					break;
				}
			}
			if (minUsableColor == -1)
			{
				++maxColor;
				graph_colors[node] = maxColor;
			}
			else
			{
				graph_colors[node] = minUsableColor;
			}
		}

		reduceDegree(node);
		nColored++;
	}

	return this->graph_colors;
}
