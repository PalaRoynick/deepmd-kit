# Build Python, C/C++ and the LAMMPS plugin from one Python environment

These instructions target Linux and a source checkout. Start with a working
C/C++ compiler, CMake >= 3.25.2, Git and Python >= 3.10 (including its development
headers). Use an install prefix you own. Keep the Python environment: the native
libraries link to the backend libraries inside it. Moving or deleting it can
break the installation. Use a new build directory when changing Python,
TensorFlow, PyTorch, MPI, compiler or CPU/CUDA variant.

## 1. Select the existing Python environment

```bash
source /absolute/path/to/venv/bin/activate
export deepmd_source_dir=/absolute/path/to/deepmd-kit
export deepmd_prefix=/absolute/path/to/deepmd-install
python -c 'import sys; print(sys.executable)'
```

If you have not created an environment yet, use `python3 -m venv /absolute/path/to/venv`
first. Run every Python installation command with the selected interpreter.
Neither CMake nor `ccmake` activates a virtual environment for subsequent shells.

For CPU, install the backends before building DeePMD-kit:

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow-cpu
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c 'import tensorflow as tf, torch; print(tf.__version__, torch.__version__)'
```

Use the backend versions pinned in the checkout's `pyproject.toml` dependency
groups for a reproducible development build. The CPU workflow was tested with
Python 3.12, TensorFlow CPU 2.21.0 and PyTorch CPU 2.11.0. Only install/enable the
backends you need; the examples below enable both.

## 2. Install the Python interface (`dp`)

```bash
cd "$deepmd_source_dir"
export DP_VARIANT=cpu DP_ENABLE_TENSORFLOW=1 DP_ENABLE_PYTORCH=1
export CMAKE_BUILD_PARALLEL_LEVEL=3
python -m pip install -e '.[test]'
dp --version
dp -h
```

`dp` is installed into the Python environment's `bin`, not into
`$deepmd_prefix/bin`. `BUILD_PY_IF` in a standalone CMake build does not replace
installation of the Python package with pip. For ordinary use omit `[test]`.

## 3. Obtain matching LAMMPS sources

Use exactly the same LAMMPS release and MPI implementation for the plugin and
LAMMPS executable. `LAMMPS_SOURCE_ROOT` is a source tree containing `src/version.h`,
not an installation prefix or the `cmake` subdirectory.

```bash
git clone --depth 1 --branch stable_22Jul2025_update2 \
  https://github.com/lammps/lammps.git /absolute/path/to/lammps
export lammps_source_dir=/absolute/path/to/lammps
```

Alternatively pass `-DLAMMPS_VERSION=stable_22Jul2025_update2` to DeePMD CMake
and omit `LAMMPS_SOURCE_ROOT`; this downloads sources during configuration.
Prefer a local source tree when also building the executable.

## 4. Build the native libraries and plugin

```bash
cmake -S "$deepmd_source_dir/source" -B "$deepmd_source_dir/build-native-cpu" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$deepmd_prefix" \
  -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')" \
  -DBUILD_CPP_IF=ON -DBUILD_PY_IF=OFF \
  -DUSE_TF_PYTHON_LIBS=ON -DUSE_PT_PYTHON_LIBS=ON \
  -DUSE_CUDA_TOOLKIT=OFF \
  -DENABLE_LAMMPS_PLUGIN=ON -DLAMMPS_SOURCE_ROOT="$lammps_source_dir" \
  -DENABLE_IPI=ON
cmake --build "$deepmd_source_dir/build-native-cpu" --parallel 3
cmake --install "$deepmd_source_dir/build-native-cpu"
```

`USE_TF_PYTHON_LIBS` discovers TensorFlow libraries and bundled TensorFlow/Google
headers from the selected Python. Do not fill in individual `TensorFlow_*`
include/library cache entries. `USE_PT_PYTHON_LIBS` similarly discovers and enables
PyTorch. For a TensorFlow-only build set `USE_PT_PYTHON_LIBS=OFF` and
`ENABLE_PYTORCH=OFF`. For a PyTorch-only build set `USE_TF_PYTHON_LIBS=OFF` and
`ENABLE_TENSORFLOW=OFF`. If configuration fails, read the first error and fix the
selected environment rather than combining headers and libraries from different
backend installations.

To use the terminal UI after initial configuration:

```bash
ccmake "$deepmd_source_dir/build-native-cpu"
```

Press `c` to configure and `g` to generate. These are normal visible cache options:

| Setting | Purpose |
| --- | --- |
| `BUILD_CPP_IF` | C/C++ libraries required by the plugin |
| `USE_TF_PYTHON_LIBS`, `USE_PT_PYTHON_LIBS` | Discover backends in `Python_EXECUTABLE` |
| `ENABLE_LAMMPS_PLUGIN` | Build the runtime-loadable plugin |
| `LAMMPS_SOURCE_ROOT`, `LAMMPS_VERSION` | Local sources or Git tag to fetch |
| `USE_CUDA_TOOLKIT` | Build NVIDIA CUDA operators; requires CUDA toolkit/nvcc |
| `ENABLE_IPI` | Build `dp_ipi`; can be disabled independently |

If MPI detection points to missing headers, install a complete MPI development
package and select its wrappers with `MPI_C_COMPILER` and `MPI_CXX_COMPILER`.
Use the same wrappers for both builds. For a serial build pass
`-DCMAKE_DISABLE_FIND_PACKAGE_MPI=ON` to DeePMD and `-DBUILD_MPI=OFF` to LAMMPS.
Do not mix serial and MPI headers.

## 5. Build and load LAMMPS

```bash
cmake -S "$lammps_source_dir/cmake" -B "$lammps_source_dir/build-cpu" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$deepmd_prefix" \
  -DBUILD_SHARED_LIBS=ON -DPKG_PLUGIN=ON -DPKG_KSPACE=ON -DPKG_EXTRA-FIX=ON \
  -DLAMMPS_INSTALL_RPATH=ON
cmake --build "$lammps_source_dir/build-cpu" --parallel 3
cmake --install "$lammps_source_dir/build-cpu"
"$deepmd_prefix/bin/lmp" -h
```

To explicitly verify plugin loading, put this line at the beginning of a LAMMPS
input file and execute it with `lmp -in file`:

```lammps
plugin load /absolute/path/to/deepmd-install/lib/libdeepmd_lmp.so
```

Then use `pair_style deepmd /absolute/path/to/model.pb` (or a supported PyTorch
model). Loading the plugin alone checks linking/registration; run a representative
simulation to validate your model. See [LAMMPS installation](install-lammps.md)
for Kokkos and built-in mode.

Expected outputs are separate:

- Python environment: `bin/dp` and the installed Python package.
- DeePMD prefix: native libraries under `lib`, including `libdeepmd_lmp.so`,
  headers under `include/deepmd`, and `bin/dp_ipi` if enabled.
- LAMMPS installation: `bin/lmp`, produced by the separate LAMMPS build.

A DeePMD CMake install containing only `dp_ipi` in `bin` is therefore normal.
If `dp` is missing, activate the environment or install the Python package.
If `lmp` is missing, build LAMMPS. For missing shared-library errors inspect
`ldd "$deepmd_prefix/lib/libdeepmd_lmp.so"` and retain the backend environment;
use `LD_LIBRARY_PATH` for nonstandard runtime library locations if necessary.

## NVIDIA CUDA variant (not GPU-tested here)

Use a separate Python environment and separate build directories. Install
CUDA-enabled TensorFlow/PyTorch versions following their official compatibility
instructions, and install a compatible CUDA toolkit containing `nvcc`. A working
GPU driver or a Python CUDA runtime wheel alone is insufficient for compilation.
Confirm `nvcc --version` and backend GPU detection first.

For the Python package set `DP_VARIANT=cuda` before pip installation. For the
native build use the command above with `-DUSE_CUDA_TOOLKIT=ON`, optionally
`-DCUDAToolkit_ROOT=/usr/local/cuda` and
`-DCMAKE_CUDA_ARCHITECTURES=<your GPU architecture>`. Keep
`USE_TF_PYTHON_LIBS`/`USE_PT_PYTHON_LIBS` pointed at the CUDA environment.
CUDA DeePMD operators and LAMMPS Kokkos are separate features: this switch does
not enable `deepmd/kk`. See the Kokkos instructions in the LAMMPS guide for that
additional build. CUDA compilation and GPU execution were not validated by the
CPU checks.
