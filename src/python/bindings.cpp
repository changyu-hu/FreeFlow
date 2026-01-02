// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Changyu Hu
//
// Commons Clause addition:
// This software is provided for non-commercial use only. See LICENSE file for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "FSI_Simulator/core/Simulator3D.hpp"
#include "FSI_Simulator/core/Simulator2D.hpp"
#include "FSI_Simulator/utils/Config.hpp"
#include "FSI_Simulator/common/Types.hpp"
#include "FSI_Simulator/utils/Profiler.hpp"
#include "FSI_Simulator/fem/FemConfig.hpp"

namespace pybind11
{
    namespace detail
    {

        // --- add support for glm::vec3 type---
        template <>
        struct type_caster<fsi::vec3_t>
        {
        public:
            bool load(handle src, bool)
            {
                if (!isinstance<array>(src))
                {
                    return false;
                }
                auto arr_untyped = reinterpret_borrow<array>(src);

                if (arr_untyped.ndim() != 1 || arr_untyped.size() != 3)
                {
                    return false;
                }

                auto arr = array_t<fsi::real>::ensure(arr_untyped);
                if (!arr)
                {
                    return false;
                }
                value = fsi::vec3_t(arr.at(0), arr.at(1), arr.at(2));
                return true;
            }

            static handle cast(const fsi::vec3_t &src, return_value_policy, handle)
            {
                return pybind11::array_t<fsi::real>(3, &src.x).release();
            }

            PYBIND11_TYPE_CASTER(fsi::vec3_t, _("numpy.ndarray[fsi::real[3]]"));
        };

        template <>
        struct type_caster<fsi::vec2_t>
        {
        public:
            bool load(handle src, bool)
            {
                if (!isinstance<array>(src))
                {
                    return false;
                }
                auto arr_untyped = reinterpret_borrow<array>(src);

                if (arr_untyped.ndim() != 1 || arr_untyped.size() != 2)
                {
                    return false;
                }
                auto arr = array_t<fsi::real>::ensure(arr_untyped);
                if (!arr)
                {
                    return false;
                }
                value = fsi::vec2_t(arr.at(0), arr.at(1));
                return true;
            }

            static handle cast(const fsi::vec2_t &src, return_value_policy, handle)
            {
                return pybind11::array_t<fsi::real>(2, &src.x).release();
            }

            PYBIND11_TYPE_CASTER(fsi::vec2_t, _("numpy.ndarray[fsi::real[2]]"));
        };

        template <>
        struct type_caster<fsi::mat3_t>
        {
        public:
            bool load(handle src, bool)
            {
                if (!isinstance<array>(src))
                    return false;
                auto arr_untyped = reinterpret_borrow<array>(src);

                if (arr_untyped.ndim() != 2 || arr_untyped.shape(0) != 3 || arr_untyped.shape(1) != 3)
                {
                    return false;
                }

                auto arr = array_t<fsi::real>::ensure(arr_untyped);
                if (!arr)
                    return false;
                value = fsi::mat3_t(
                    arr.at(0, 0), arr.at(0, 1), arr.at(0, 2),
                    arr.at(1, 0), arr.at(1, 1), arr.at(1, 2),
                    arr.at(2, 0), arr.at(2, 1), arr.at(2, 2));
                return true;
            }

            static handle cast(const fsi::mat3_t &src, return_value_policy, handle)
            {
                array_t<fsi::real> arr({3, 3});
                auto req = arr.request();
                fsi::real *ptr = static_cast<fsi::real *>(req.ptr);
                fsi::mat3_t T = glm::transpose(src);
                std::memcpy(ptr, &T[0][0], 9 * sizeof(fsi::real));
                return arr.release();
            }
            PYBIND11_TYPE_CASTER(fsi::mat3_t, _("numpy.ndarray[fsi::real[3, 3]]"));
        };

    }
} // namespace pybind11::detail

PYBIND11_MODULE(fsi_simulator, m)
{
    m.doc() = "Fluid-Structure Interaction Simulator with LBM and FEM";

    pybind11::module_ fem_m = m.def_submodule("fem", "FEM related classes");

    pybind11::class_<fsi::fem::FemSolverOptions>(fem_m, "FemSolverOptions")
        .def(pybind11::init<>())
        .def_readwrite("optimizer_type", &fsi::fem::FemSolverOptions::optimizer_type)
        .def_readwrite("iterations", &fsi::fem::FemSolverOptions::iterations)
        // ... 绑定所有其他你希望在Python中访问的FemSolverOptions成员 ...
        .def_readwrite("force_density_abs_tol", &fsi::fem::FemSolverOptions::force_density_abs_tol)
        .def_readwrite("grad_check", &fsi::fem::FemSolverOptions::grad_check)
        .def_readwrite("thread_ct", &fsi::fem::FemSolverOptions::thread_ct)
        .def_readwrite("substeps", &fsi::fem::FemSolverOptions::substeps)
        .def_readwrite("omega", &fsi::fem::FemSolverOptions::omega)
        .def_readwrite("vbd_iterations", &fsi::fem::FemSolverOptions::vbd_iterations)
        .def_readwrite("ls_max_iter", &fsi::fem::FemSolverOptions::ls_max_iter)
        .def_readwrite("ls_beta", &fsi::fem::FemSolverOptions::ls_beta)
        .def_readwrite("ls_alpha", &fsi::fem::FemSolverOptions::ls_alpha)
        .def_readwrite("linear_solver_type", &fsi::fem::FemSolverOptions::linear_solver_type);

    pybind11::class_<fsi::fem::SolidBodyConfig2D>(fem_m, "SolidBodyConfig2D")
        .def(pybind11::init<>())
        .def_readwrite("mesh_path", &fsi::fem::SolidBodyConfig2D::mesh_path)
        .def_readwrite("density", &fsi::fem::SolidBodyConfig2D::density)
        .def_readwrite("youngs_modulus", &fsi::fem::SolidBodyConfig2D::youngs_modulus)
        .def_readwrite("poisson_ratio", &fsi::fem::SolidBodyConfig2D::poisson_ratio)
        .def_readwrite("translate", &fsi::fem::SolidBodyConfig2D::translate)
        .def_readwrite("rotate", &fsi::fem::SolidBodyConfig2D::rotate)
        .def_readwrite("scale", &fsi::fem::SolidBodyConfig2D::scale)
        .def_readwrite("initial_velocity", &fsi::fem::SolidBodyConfig2D::initial_velocity)
        .def_readwrite("lbs_control_config", &fsi::fem::SolidBodyConfig2D::lbs_control_config);

    pybind11::class_<fsi::fem::SolidBodyConfig3D>(fem_m, "SolidBodyConfig3D")
        .def(pybind11::init<>())
        .def_readwrite("mesh_path", &fsi::fem::SolidBodyConfig3D::mesh_path)
        .def_readwrite("density", &fsi::fem::SolidBodyConfig3D::density)
        .def_readwrite("youngs_modulus", &fsi::fem::SolidBodyConfig3D::youngs_modulus)
        .def_readwrite("poisson_ratio", &fsi::fem::SolidBodyConfig3D::poisson_ratio)
        .def_readwrite("translate", &fsi::fem::SolidBodyConfig3D::translate)
        .def_readwrite("rotate", &fsi::fem::SolidBodyConfig3D::rotate)
        .def_readwrite("scale", &fsi::fem::SolidBodyConfig3D::scale)
        .def_readwrite("initial_velocity", &fsi::fem::SolidBodyConfig3D::initial_velocity)
        .def_readwrite("lbs_control_config", &fsi::fem::SolidBodyConfig3D::lbs_control_config);

    pybind11::class_<fsi::SimulationParameters3D>(m, "SimulationParameters3D")
        .def(pybind11::init<>())
        .def_readwrite("solid_solver_type", &fsi::SimulationParameters3D::solid_solver_type)
        .def_readwrite("total_time", &fsi::SimulationParameters3D::total_time)
        .def_readwrite("dt", &fsi::SimulationParameters3D::dt)
        .def_readwrite("solids", &fsi::SimulationParameters3D::solids)
        .def_readwrite("global_fem_options", &fsi::SimulationParameters3D::global_fem_options)
        .def_readwrite("use_newton", &fsi::SimulationParameters3D::use_newton)
        .def_readwrite("output_path", &fsi::SimulationParameters3D::output_path)
        .def_readwrite("log_level", &fsi::SimulationParameters3D::log_level)
        .def_readwrite("log_file", &fsi::SimulationParameters3D::log_file)
        .def_readwrite("boundary_velocities", &fsi::SimulationParameters3D::boundary_velocities)
        .def_readwrite("boundaries", &fsi::SimulationParameters3D::boundaries)
        .def_readwrite("fluid_viscosity", &fsi::SimulationParameters3D::fluid_viscosity)
        .def_readwrite("fluid_density", &fsi::SimulationParameters3D::fluid_density)
        .def_readwrite("fluid_nx", &fsi::SimulationParameters3D::fluid_nx)
        .def_readwrite("fluid_ny", &fsi::SimulationParameters3D::fluid_ny)
        .def_readwrite("fluid_nz", &fsi::SimulationParameters3D::fluid_nz)
        .def_readwrite("fluid_dx", &fsi::SimulationParameters3D::fluid_dx);

    pybind11::class_<fsi::SimulationParameters2D>(m, "SimulationParameters2D")
        .def(pybind11::init<>())
        .def_readwrite("solid_solver_type", &fsi::SimulationParameters2D::solid_solver_type)
        .def_readwrite("total_time", &fsi::SimulationParameters2D::total_time)
        .def_readwrite("dt", &fsi::SimulationParameters2D::dt)
        .def_readwrite("solids", &fsi::SimulationParameters2D::solids)
        .def_readwrite("global_fem_options", &fsi::SimulationParameters2D::global_fem_options)
        .def_readwrite("use_newton", &fsi::SimulationParameters2D::use_newton)
        .def_readwrite("output_path", &fsi::SimulationParameters2D::output_path)
        .def_readwrite("log_level", &fsi::SimulationParameters2D::log_level)
        .def_readwrite("log_file", &fsi::SimulationParameters2D::log_file)
        .def_readwrite("boundary_velocities", &fsi::SimulationParameters2D::boundary_velocities)
        .def_readwrite("boundaries", &fsi::SimulationParameters2D::boundaries)
        .def_readwrite("fluid_viscosity", &fsi::SimulationParameters2D::fluid_viscosity)
        .def_readwrite("fluid_density", &fsi::SimulationParameters2D::fluid_density)
        .def_readwrite("fluid_nx", &fsi::SimulationParameters2D::fluid_nx)
        .def_readwrite("fluid_ny", &fsi::SimulationParameters2D::fluid_ny)
        .def_readwrite("fluid_dx", &fsi::SimulationParameters2D::fluid_dx);

    pybind11::class_<fsi::Config2D>(m, "Config2D")
        .def(pybind11::init<>(), "Default constructor")
        .def("load", &fsi::Config2D::load, pybind11::arg("filepath"), "Load configuration from a JSON file.")
        .def("get_params", &fsi::Config2D::getParams,
             pybind11::return_value_policy::reference_internal,
             "Get a reference to the loaded simulation parameters.");

    pybind11::class_<fsi::Config3D>(m, "Config3D")
        .def(pybind11::init<>(), "Default constructor")
        .def("load", &fsi::Config3D::load, pybind11::arg("filepath"), "Load configuration from a JSON file.")
        .def("get_params", &fsi::Config3D::getParams,
             pybind11::return_value_policy::reference_internal,
             "Get a reference to the loaded simulation parameters.");

    pybind11::class_<fsi::Simulator3D>(m, "Simulator3D")
        .def(pybind11::init<const fsi::SimulationParameters3D &>(),
             pybind11::arg("params"),
             "Constructor that takes simulation parameters.")
        .def("initialize", &fsi::Simulator3D::initialize, "Initialize the simulation environment.")
        .def("reset", &fsi::Simulator3D::reset, "Reset the simulation environment.")
        .def("step", &fsi::Simulator3D::step, "Advance one time step.")
        .def("enableFluidSolver", &fsi::Simulator3D::enableFluidSolver, pybind11::arg("enable"), "enable/disable fluid solver.")
        .def("apply_lbs_control", &fsi::Simulator3D::applyLBSControl,
             pybind11::arg("mesh_id"),
             pybind11::arg("lbs_shift"),
             pybind11::arg("lbs_rotation"),
             "Apply LBS control data to a specific mesh.")
        .def("apply_active_strain", &fsi::Simulator3D::applyActiveStrain,
             pybind11::arg("mesh_id"),
             pybind11::arg("tet_id"),
             pybind11::arg("dir"),
             pybind11::arg("magnitude"),
             "Apply active strain to a specific tetrahedron of a mesh.")
        .def("apply_kinematic_control", &fsi::Simulator3D::applyKinematicControl,
             pybind11::arg("mesh_id"),
             pybind11::arg("target_pos"),
             pybind11::arg("vel"),
             "Apply kinematic control to a specific mesh.")
        .def("save_frame_data", &fsi::Simulator3D::saveFrameData,
             pybind11::arg("frame_index"), pybind11::arg("save_fluid"), pybind11::arg("save_solid"),
             "Save simulation data for a given frame.")
        .def("get_fluid_moments", [](fsi::Simulator3D &self)
             {
            const auto& moments = self.getFluidMoment();
            int nx = self.m_params.fluid_nx;
            int ny = self.m_params.fluid_ny;
            int nz = self.m_params.fluid_nz;
            int shape[4] = {nz, ny, nx, 10}; 
            return pybind11::array_t<float>(shape, moments.data()); }, "Get the current fluid moment fields.")
        .def("getVertices", &fsi::Simulator3D::getVertices, "Get the position of solid vertices")
        .def("getVelocity", &fsi::Simulator3D::getVelocity, "Get the velocity of solid vertices")
        .def("getForce", &fsi::Simulator3D::getForce, "Get the force of fluid nodes")
        .def("getTetrahedra", &fsi::Simulator3D::getTetrahedra, "Get the indices of tetrahedra of solid meshes")
        .def("getBoundaryVertices", &fsi::Simulator3D::getBoundaryVertices, "Get the position of boundary vertices")
        .def("getBoundaryElements", &fsi::Simulator3D::getBoundaryElements, "Get the indices of surface elements of solid meshes")
        .def("getControlPointIdx", &fsi::Simulator3D::getControlPointIdx, "Get the indices of lbs control points")
        .def("getLBSVertices", &fsi::Simulator3D::getLBSVertices, "Get the position of lbs modulated vertices")
        .def("getBoundaryPointIdx", &fsi::Simulator3D::getBoundaryPointIdx, "Get the indices of boundary points")
        .def("begin_profiler", [](fsi::Simulator3D &self, std::string name)
             { fsi::PROFILE_SESSION(name); }, "Activate the built-in profiler.")
        .def("end_profiler", [](fsi::Simulator3D &self)
             { fsi::PROFILE_END_SESSION(); }, "End the profiling session and output results.");

    pybind11::class_<fsi::Simulator2D>(m, "Simulator2D")
        .def(pybind11::init<const fsi::SimulationParameters2D &>(),
             pybind11::arg("params"),
             "Constructor that takes simulation parameters.")
        .def("initialize", &fsi::Simulator2D::initialize, "Initialize the simulation environment.")
        .def("reset", &fsi::Simulator2D::reset, "Reset the simulation environment.")
        .def("step", &fsi::Simulator2D::step, "Advance one time step.")
        .def("fillSolid", &fsi::Simulator2D::fillSolid, "Set the flags of fluid nodes that lie in the interior of solid meshes.")
        .def("enableFluidSolver", &fsi::Simulator2D::enableFluidSolver, pybind11::arg("enable"), "enable/disable fluid solver.")
        .def("apply_lbs_control", &fsi::Simulator2D::applyLBSControl,
             pybind11::arg("mesh_id"),
             pybind11::arg("lbs_shift"),
             pybind11::arg("lbs_rotation"),
             "Apply LBS control data to a specific mesh.")
        .def("apply_kinematic_control", &fsi::Simulator2D::applyKinematicControl,
             pybind11::arg("mesh_id"),
             pybind11::arg("target_pos"),
             pybind11::arg("vel"),
             "Apply kinematic control to a specific mesh.")
        .def("apply_active_strain", &fsi::Simulator2D::applyActiveStrain,
             pybind11::arg("mesh_id"),
             pybind11::arg("tri_id"),
             pybind11::arg("dir"),
             pybind11::arg("magnitude"),
             "Apply active strain to a specific triangle of a mesh.")
        .def("save_frame_data", &fsi::Simulator2D::saveFrameData,
             pybind11::arg("frame_index"),
             "Save simulation data for a given frame.")
        .def("get_fluid_moments", [](fsi::Simulator2D &self)
             {
            const auto& moments = self.getFluidMoment();
            int nx = self.m_params.fluid_nx;
            int ny = self.m_params.fluid_ny;
            int shape[3] = {ny, nx, 6}; 
            return pybind11::array_t<float>(shape, moments.data()); }, "Get the current fluid moment fields.")
        .def("getVertices", &fsi::Simulator2D::getVertices, "Get the position of solid vertices")
        .def("getVelocity", &fsi::Simulator2D::getVelocity, "Get the velocity of solid vertices")
        .def("getForce", &fsi::Simulator2D::getForce, "Get the force of fluid nodes")
        .def("getTriangles", &fsi::Simulator2D::getTriangles, "Get the indices of triangles of solid meshes")
        .def("getBoundaryVertices", &fsi::Simulator2D::getBoundaryVertices, "Get the position of boundary vertices")
        .def("getBoundaryElements", &fsi::Simulator2D::getBoundaryElements, "Get the indices of surface elements of solid meshes")
        .def("getControlPointIdx", &fsi::Simulator2D::getControlPointIdx, "Get the indices of lbs control points")
        .def("getLBSVertices", &fsi::Simulator2D::getLBSVertices, "Get the position of lbs modulated vertices")
        .def("getBoundaryPointIdx", &fsi::Simulator2D::getBoundaryPointIdx, "Get the indices of boundary points")
        .def("begin_profiler", [](fsi::Simulator2D &self, std::string name)
             { fsi::PROFILE_SESSION(name); }, "Activate the built-in profiler.")
        .def("end_profiler", [](fsi::Simulator2D &self)
             { fsi::PROFILE_END_SESSION(); }, "End the profiling session and output results.");
}