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
#include <string>
#include <fstream>
#include <iostream>
#include <array>

namespace GAIA
{
	namespace GraphColoring
	{

		struct Graph
		{
			size_t numNodes;
			typedef std::array<unsigned int, 2> Edge;
			std::vector<Edge> edges;
			std::vector<float> edgeWeights;

			void saveColFile(std::string outFile)
			{
				std::ofstream ofs(outFile);
				if (ofs.is_open())
				{
					ofs << "p edge " << numNodes << " " << edges.size() << "\n";
					for (size_t iEdge = 0; iEdge < edges.size(); iEdge++)
					{
						ofs << "e " << edges[iEdge][0] + 1 << " " << edges[iEdge][1] + 1 << "\n";
					}
					ofs.close();
				}
				else
				{
					std::cout << "Fail to open: " << outFile;
				}
			}
		};
	}
}
