// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <map>

#include <glm/glm.hpp>

#include "FSI_Simulator/utils/DataStructures.hpp"
#include "FSI_Simulator/utils/Logger.hpp"

#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPoints.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkTetra.h>
#include <vtkTriangle.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLUnstructuredGridWriter.h>

namespace fsi
{

    namespace io
    {

        class VtkWriter
        {
        public:
            // --- for LBM fluid (Structured Data -> vtkImageData) ---

            template <typename T>
            static bool writeImageDataVectorField(
                const std::string &filepath,
                const T *data_u, const T *data_v, const T *data_w,
                int nx, int ny, int nz,
                const std::string &data_name,
                const glm::vec3 &spacing,
                const glm::vec3 &origin)
            {
                LOG_INFO("Writing VTK ImageData to: {}", filepath);

                auto imageData = vtkSmartPointer<vtkImageData>::New();
                imageData->SetDimensions(nx, ny, nz);
                imageData->SetSpacing(spacing.x, spacing.y, spacing.z);
                imageData->SetOrigin(origin.x, origin.y, origin.z);

                size_t num_points = static_cast<size_t>(nx) * ny * nz;
                std::vector<T> interleaved_vectors(num_points * 3);
                for (size_t i = 0; i < num_points; ++i)
                {
                    interleaved_vectors[i * 3 + 0] = data_u[i];
                    interleaved_vectors[i * 3 + 1] = data_v[i];
                    interleaved_vectors[i * 3 + 2] = data_w[i];
                }

                auto vectors = createVtkDataArray(interleaved_vectors.data(), num_points, 3, data_name);
                imageData->GetPointData()->SetVectors(vectors);

                auto writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
                writer->SetFileName(filepath.c_str());
                writer->SetInputData(imageData);
                writer->SetDataModeToAppended(); // using compression mode, file is smaller
                writer->EncodeAppendedDataOn();
                writer->Write();

                return true;
            }

            template <typename T>
            static bool writeUnstructuredGrid(
                const std::string &filepath,
                const std::vector<glm::vec<3, T>> &vertices,
                const std::vector<std::pair<VtkCellType, std::vector<uint32_t>>> &elements_groups)
            {
                LOG_INFO("Writing VTK UnstructuredGrid to: {}", filepath);

                auto points = vtkSmartPointer<vtkPoints>::New();
                points->SetDataType(std::is_same_v<T, float> ? VTK_FLOAT : VTK_DOUBLE);
                points->SetNumberOfPoints(vertices.size());
                for (size_t i = 0; i < vertices.size(); ++i)
                {
                    points->SetPoint(i, static_cast<double>(vertices[i].x), static_cast<double>(vertices[i].y), static_cast<double>(vertices[i].z));
                }

                auto uGrid = vtkSmartPointer<vtkUnstructuredGrid>::New();
                uGrid->SetPoints(points);

                for (const auto &group : elements_groups)
                {
                    const VtkCellType cell_type = group.first;
                    const auto &connectivity = group.second;

                    int verts_per_cell = 0;
                    VTKCellType vtk_enum_type;

                    switch (cell_type)
                    {
                    case VtkCellType::Line:
                        verts_per_cell = 2;
                        vtk_enum_type = VTK_LINE;
                        break;
                    case VtkCellType::Triangle:
                        verts_per_cell = 3;
                        vtk_enum_type = VTK_TRIANGLE;
                        break;
                    case VtkCellType::Tetra:
                        verts_per_cell = 4;
                        vtk_enum_type = VTK_TETRA;
                        break;
                    default:
                        LOG_WARN("Unsupported VtkCellType encountered in VtkWriter. Skipping.");
                        continue;
                    }

                    if (connectivity.empty())
                    {
                        continue;
                    }
                    ASSERT(connectivity.size() % verts_per_cell == 0, "Element connectivity data size is not a multiple of verts_per_cell.");

                    const size_t num_cells_in_group = connectivity.size() / verts_per_cell;

                    for (size_t i = 0; i < num_cells_in_group; ++i)
                    {
                        std::vector<vtkIdType> ptIds(verts_per_cell);
                        for (int j = 0; j < verts_per_cell; ++j)
                        {
                            ptIds[j] = static_cast<vtkIdType>(connectivity[i * verts_per_cell + j]);
                        }
                        uGrid->InsertNextCell(vtk_enum_type, verts_per_cell, ptIds.data());
                    }
                }

                auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
                writer->SetFileName(filepath.c_str());
                writer->SetInputData(uGrid);
                writer->SetDataModeToAppended();
                writer->EncodeAppendedDataOn();
                writer->Write();

                return true;
            }

        private:
            template <typename T>
            static vtkSmartPointer<vtkAOSDataArrayTemplate<T>> createVtkDataArray(
                const T *data, size_t num_tuples, int num_components, const std::string &name)
            {
                auto vtk_array = vtkSmartPointer<vtkAOSDataArrayTemplate<T>>::New();
                vtk_array->SetNumberOfComponents(num_components);
                vtk_array->SetNumberOfTuples(num_tuples);
                vtk_array->SetName(name.c_str());
                std::copy(data, data + num_tuples * num_components, vtk_array->GetPointer(0));
                return vtk_array;
            }
        };

    } // namespace io

} // namespace fsi