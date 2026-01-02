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
		 * MCS (Register Allocation via Coloring of Chordal Graphs - Magno et al.)
		 */
		class Mcs : public GraphColor
		{
		public:
			/* Constructors */
			Mcs(const Graph &graph) : GraphColor(graph) {}
			Mcs(vector<vector<int>> &graph) : GraphColor(graph) {}

			/* Mutators */
			vector<int> &color();

			/* Accessors */
			string get_algorithm() { return "MCS"; }
		};
	}
}