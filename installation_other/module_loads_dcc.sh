#!/usr/bin/env bash

# Shared module loader for Duke Compute Cluster (DCC) installation scripts.
# Usage:
#   source "installation_other/module_loads_dcc.sh"
#   load_dcc_modules "12.8"
#
# DCC differs from Perlmutter in three ways that matter here:
#   1. DCC uses Tcl Environment Modules 5.3.0, NOT Lmod. There is no 'module
#      spider' and no Lmod 'module reset'; we use 'module purge' instead.
#   2. There are no Cray compiler wrappers (cc/CC/ftn). MPI compilation goes
#      through mpicc/mpicxx from whichever OpenMPI module is loaded.
#   3. The CUDA modulefile sets PATH and LD_LIBRARY_PATH but NOT CUDA_HOME,
#      so we derive CUDA_HOME from the location of nvcc.

# Paths are declared as env-overridable variables so callers can adjust
# platform-specific locations without modifying script logic.
MODULES_SH_PATH="${MODULES_SH_PATH:-/etc/profile.d/modules.sh}"
MODULES_INIT_BASH_PATH="${MODULES_INIT_BASH_PATH:-/usr/share/Modules/init/bash}"
LMOD_INIT_BASH_PATH="${LMOD_INIT_BASH_PATH:-/usr/share/lmod/lmod/init/bash}"

# MPI selection.
#
# Infiniband/OpenMPI/5.0.2-CUDA is the CUDA-aware build and would be preferable,
# but it is BROKEN on DCC as of 2026-07-21 and cannot be linked against:
#   - Its modulefile prepends /opt/apps/staging/ucx-1.18.0/lib, which DOES NOT
#     EXIST (only ucx-1.20.0 is installed).
#   - libmpi.so and libopen-pal.so.80 carry 85 undefined UCX symbols
#     (ucp_*, uct_*, ucm_*, ucs_*). Putting ucx-1.20.0 on LIBRARY_PATH does not
#     resolve them.
#   - They also reference xpmem_*, and no libxpmem exists anywhere on the system.
# Building mpi4py against it fails at the link step. OpenMPI/4.1.6 links cleanly,
# so it is the default.
#
# Consequence: MPI is NOT CUDA-aware. HydraGNN's distributed training moves
# gradients over torch.distributed/NCCL rather than MPI, so the effect is limited
# to mpi4py data-distribution calls, which stage through host memory.
#
# If Duke repairs the UCX installation, set:
#   DCC_MPI_PRIMARY_MODULE=Infiniband/OpenMPI/5.0.2-CUDA
DCC_MPI_PRIMARY_MODULE="${DCC_MPI_PRIMARY_MODULE:-OpenMPI/4.1.6}"
DCC_MPI_FALLBACK_MODULE="${DCC_MPI_FALLBACK_MODULE:-OpenMPI/4.1.5rc2}"
DCC_CMAKE_MODULE="${DCC_CMAKE_MODULE:-cmake/3.28.3}"
DCC_CONDA_MODULE="${DCC_CONDA_MODULE:-Anaconda3/2024.02}"

# Records which MPI module actually loaded, for the caller and for logging.
DCC_MPI_MODULE_LOADED=""

load_dcc_modules() {
  local expected_cuda_mm="$1"

  if ! command -v module >/dev/null 2>&1; then
    if [[ -f "${MODULES_SH_PATH}" ]]; then
      source "${MODULES_SH_PATH}"
    elif [[ -f "${MODULES_INIT_BASH_PATH}" ]]; then
      source "${MODULES_INIT_BASH_PATH}"
    elif [[ -f "${LMOD_INIT_BASH_PATH}" ]]; then
      source "${LMOD_INIT_BASH_PATH}"
    fi
  fi

  if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: 'module' command not found. Ensure you're running on DCC login/compute nodes."
    return 1
  fi

  # Tcl Environment Modules has no Lmod-style 'module reset'; purge is the
  # equivalent way to reach a deterministic starting state.
  module purge 2>/dev/null || true

  module load "CUDA/${expected_cuda_mm}" || {
    echo "ERROR: failed to load CUDA/${expected_cuda_mm}."
    echo "       DCC provides CUDA/12.4 and CUDA/12.8; there is no CUDA/12.9."
    return 1
  }

  # The CUDA modulefile does not export CUDA_HOME, but torch/PyG source builds
  # and mpi4py expect it. Derive it from nvcc's location.
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(dirname "$(command -v nvcc)")")" && pwd)"
    export CUDA_HOME
    export CUDA_PATH="${CUDA_HOME}"
  fi

  # Prefer the CUDA-aware OpenMPI (it also pulls Infiniband/UCX). If it fails to
  # load, purge and rebuild the module state around vanilla OpenMPI instead.
  if module load "${DCC_MPI_PRIMARY_MODULE}" 2>/dev/null; then
    DCC_MPI_MODULE_LOADED="${DCC_MPI_PRIMARY_MODULE}"
  else
    echo "WARNING: ${DCC_MPI_PRIMARY_MODULE} failed to load; falling back to ${DCC_MPI_FALLBACK_MODULE}"
    module purge 2>/dev/null || true
    module load "CUDA/${expected_cuda_mm}" || return 1
    if command -v nvcc >/dev/null 2>&1; then
      CUDA_HOME="$(cd "$(dirname "$(dirname "$(command -v nvcc)")")" && pwd)"
      export CUDA_HOME
      export CUDA_PATH="${CUDA_HOME}"
    fi
    module load "${DCC_MPI_FALLBACK_MODULE}" || {
      echo "ERROR: neither ${DCC_MPI_PRIMARY_MODULE} nor ${DCC_MPI_FALLBACK_MODULE} could be loaded."
      return 1
    }
    DCC_MPI_MODULE_LOADED="${DCC_MPI_FALLBACK_MODULE}"
  fi
  export DCC_MPI_MODULE_LOADED

  module load "${DCC_CMAKE_MODULE}" || true
  module load "${DCC_CONDA_MODULE}" || true
}

print_dcc_activation_instructions() {
  local expected_cuda_mm="$1"
  local venv_path="$2"

  cat <<EOF
Module load + activation (for future sessions):
module purge
module load CUDA/${expected_cuda_mm}
module load ${DCC_MPI_MODULE_LOADED:-${DCC_MPI_PRIMARY_MODULE}}
module load ${DCC_CMAKE_MODULE}
module load ${DCC_CONDA_MODULE}
export CUDA_HOME="\$(cd "\$(dirname "\$(dirname "\$(command -v nvcc)")")" && pwd)"
source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || eval "\$("\$(conda info --base)/bin/conda" shell.bash hook)"
conda activate ${venv_path}
EOF
}
