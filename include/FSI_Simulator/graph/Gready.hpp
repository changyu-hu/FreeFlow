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
#include "FSI_Simulator/graph/ColoringAlgorithms.hpp"

namespace GAIA
{
	namespace GraphColoring
	{

		/*
		 * Coloring algorithm provided by
		 * Ton-That, Quoc-Minh, Paul G. Kry, and Sheldon Andrews. "Parallel block Neo-Hookean XPBD using graph clustering." Computers & Graphics 110 (2023): 1-10.
		 * Faster than MCS but results are inferior
		 */

		class OrderedGreedy : public GraphColor
		{
		public:
			/* Constructors */
			OrderedGreedy(const Graph &graph) : GraphColor(graph) {}
			OrderedGreedy(vector<vector<int>> &graph) : GraphColor(graph) {}

			virtual int nextNode();
			void reduceDegree(int iNode);
			/* Mutators */
			vector<int> &color();

			/* Accessors */
			string get_algorithm() { return "OrderedGreedy"; }

			std::vector<int> degrees;
			std::vector<bool> colored;
		};
	}
}
