# Learning to Control Free Form Soft Swimmers

![MIT License](https://img.shields.io/badge/license-MIT-green)

This is the official repository for the paper [Learning to Control Free Form Soft Swimmers](https://neurips.cc/virtual/2025/poster/117384).

We provide a high-performance research-oriented simulator for fluid-structure interaction (FSI), leveraging the Lattice Boltzmann Method (LBM) for fluid dynamics and the Vertex Block Descent (VBD) for elastic body. The core is written in modern C++ and CUDA for parallel computation, with a user-friendly Python interface provided via pybind11. We also provide Python scripts for experiments of training and evaluating the learned controllers of soft swimmers.

![](./docs/assets/clownfish_lbs.gif)

![](./docs/assets/torus_lbs.gif)

![](./docs/assets/fish2d_navigation.gif)

## Features

- **Fluid Solver**: High-Order Moment Encoded Kinematic Solver (HOME-LBM) for fluid dynamics.
- **Solid Solver**: Vertex Block Descent(VBD) solver for deformable bodies.
- **FSI Coupling**: Robust coupling scheme to handle the interaction between the fluid and solid domains.
- **Python Bindings**: Easy-to-use Python API for setting up, running, and analyzing simulations.
- **Configurable**: Simulations are fully configurable via easy-to-read JSON files.

## Prerequisites

Before you begin, ensure you have the following dependencies installed on your system.

### 1. Core Build Tools
- A C++17 compliant compiler (GCC 9+, Clang 10+, MSVC 2019+).
- **CMake** (version 3.18 or higher).
- **Git**.
- **Python** (version 3.8 or higher) and `pip`.
- **(For Linux)**: GNU Build System tools required by some dependencies.
  ```bash
  # On Debian/Ubuntu
  sudo apt-get update && sudo apt-get install build-essential autoconf automake libtool pkg-config
  ```

### 2. NVIDIA CUDA Toolkit
NVIDIA CUDA Toolkit (version 11.6 or higher is recommended).
A CUDA-enabled NVIDIA GPU with Compute Capability 6.0 (Pascal) or higher.
Ensure that nvcc is in your system's PATH. You can check this by running nvcc --version.

### 3. vcpkg (Dependency Manager)
This project uses vcpkg to manage all C++ third-party libraries. If you don't have it, install it in a location of your choice (e.g., in your home directory or a development folder):
  ```bash
  git clone https://github.com/microsoft/vcpkg.git
  cd vcpkg
  ./bootstrap-vcpkg.sh  # on Linux/macOS
  ./bootstrap-vcpkg.bat # on Windows
  ```

### 4. cuDSS support
If you want to use GPU to accelerate the sparse linear solver in Newton's method, please install cuDSS library following the instruction: [cuDSS](https://developer.nvidia.com/cudss).

Then turn on the flag in CMakeLists.txt:

```cmake
# ...
option(USE_CUDSS "Enable the cuDSS sparse direct solver" ON)
```

## Installation

The project uses a vcpkg.json manifest file, which means vcpkg will automatically install all required C++ libraries during the CMake configuration step.

### Step 1: Clone the Repository

```bash
git clone https://github.com/changyu-hu/FreeFlow
cd FreeFlow
```

### Step 2: Export vcpkg environment variables

~~~bash
export VCPKG_ROOTCMAKE_TOOLCHAIN_FILE=$path/to/your/vcpkg/scripts/buildsystems/vcpkg.cmake
~~~

### Step 3: Build with python bindings

```bash
conda create -n fsi python=3.10
conda activate fsi
pip install -r requirements.txt
pip install -e . -v
```

This will compile the C++ core and create the Python module in your current python environment.

## Usage

To run a simulation, you should first create a JSON configuration file. The configuration file specifies the simulation parameters and solver settings. Some examples can be found in the `assets/configs` folder. 

Below is an example of a configuration file for a free-form soft swimmer:

```json
{
    "dimension": 2,                 // 2D or 3D simulation
    "fluid_viscosity": 0.005,       // fluid viscosity
    "fluid_density": 10.0,          // fluid density
    "fluid_nx": 1200,               // number of cells in x-direction
    "fluid_ny": 400,                // number of cells in y-direction
    "fluid_dx": 0.005,              // cell size of LBM solver
    "solid_solver_type": "vbd",     // solid solver, "static" or "vbd"
    "total_time": 10.0,             // not used
    "dt": 0.0025,                   // time step size 
    "output_frequency": 200,        // not used
    "output_path": "output",        // path to save simulation results and log files
    "log_level": "info",            // log level, "trace", "debug", "info", "warn", "err", "critical"
    "log_file": "simulation_2d.log", // log file name
    "global_fem_options": {
        "optimizer_type": "newton", // only support Newton's method for now
        "iterations": 1000,         // maximum number of iterations for Newton's method
        "verbose_level": 1,
        "line_search_method": "backtracking", //only support backtracking line search for now
        "force_density_abs_tol": 0.001,      // absolute tolerance for force density
        "ls_max_iter": 50,                   // maximum number of line search iterations
        "ls_beta": 0.3,                      // line search parameter beta
        "ls_alpha": 0.0001,                  // line search parameter alpha
        "linear_solver_type": "eigen_ldlt",  // linear solver type, "eigen_ldlt", "cholmod_ldlt", "cuda_qr", "cuda_lu" (recommended)
        "grad_check": false,                 // enable gradient check
        "substeps": 3,                       // number of substeps for VBD solver
        "vbd_iterations": 30,                // number of iterations for VBD solver
        "omega": 0.8                         // acceleration parameter for VBD solver
    },
    "solids": [
        {
            "mesh_path": "fish2d.mesh", // path to the mesh file  
            "density": 10.0,            // density of the solid
            "youngs_modulus": 1000.0,   // Young's modulus of the solid
            "poisson_ratio": 0.4,       // Poisson's ratio of the solid
            "lbs_control_config": {
                "cnum": 3,              // number of LBS control points
                "omega": 0.3,           // Control the localization of the LBS weights
                "stiffness": 0.1        // Dynamic Correction stiffness
            }
        }
    ],
    "boundaries": [
        {
            "type": "OutletDown",       //set boundary conditions for the fluid, this should be generated automatically, see example codes
            "pos": [
                0,
                0
            ]
        },
        //...
    ],
    "boundary_velocities": [            // set boundary velocity for the fluid (four directions in 2D, six directions in 3D), only used for Inlet boundaries
        0.0,
        0.0,
        0.0,
        0.0
    ]
}
```
There are some exmaple Python scripts in the `scripts` folder that show how to invoke the simulator's Python interface.

## RL Training

To reproduce the experiments in the paper, you can run the RL training script using the following command. 

```shell
cd rl
# train
python train.py --gpus "0,1" --train --cfg_path ./task.json
# test
python train.py --test --cfg_path ./task.json
```

The models shown in the paper are provided in the `assets` folder. You can modify the training task configuration file to change the model or training parameters.

## License

This project is licensed under the [MIT License](LICENSE).
