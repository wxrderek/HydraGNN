#!/usr/bin/env bash
# hydragnn_installation_bash_script_dcc.sh
# Complete automated setup for the HydraGNN environment on the Duke Compute Cluster
# (DCC), youlab-gpu partition: AlmaLinux 9, NVIDIA RTX A5000 (compute capability 8.6).
#
# This is the DCC counterpart of hydragnn_installation_bash_script_perlmutter.sh.
# Scope is the examples/qmugs/ density matrix learning task ONLY. Dependencies not
# reachable from that task (ADIOS2, DDStore, DeepHyper, DeepSpeed, GPTL, TensorFlow)
# are deliberately omitted; see references/dcc_info/perlmutter_to_dcc_transfer.md.
#
# DIFFERENCES FROM THE PERLMUTTER SCRIPT:
#  1) PyG compiled extensions are installed from PREBUILT WHEELS. Perlmutter needed
#     source builds because its glibc predated the 2.33 the wheels require; DCC has
#     glibc 2.34, so the wheels load directly. A source build remains as fallback.
#  2) No Cray compiler wrappers (cc/CC/ftn) exist on DCC. mpi4py builds against
#     mpicc from the loaded OpenMPI module.
#  3) CUDA 12.8 is the DCC ceiling (there is no 12.9), so cu128 wheels are used.
#  4) TORCH_CUDA_ARCH_LIST is 8.6 for the A5000s, not 8.0 for Perlmutter's A100s.
#  5) conda/pip caches are forced off $HOME, which has a hard 25 GB quota on DCC.
#  6) The environment lives on /hpc/group, NOT inside the repo on /work. The VAST
#     filesystem backing /work rejects filenames containing * ? : | which psi4's
#     libint basis sets require. See the Set Base Installation Directory section.
#
# Usage:
#   bash installation_other/hydragnn_installation_bash_script_dcc.sh
#
# Optional env vars:
#   VENV_PATH=/path/to/env
#   PYTHON_VERSION=3.11
#   EXPECTED_CUDA_MM=12.8
#   TORCH_CUDA_TAG=cu128
#   TORCH_CUDA_ARCH_LIST=8.6
#   MAX_JOBS=16
#   INSTALL_PSI4=1
#   INSTALL_ROOT=/some/path/HydraGNN-Installation-Duke-Compute-Cluster

set -Eeuo pipefail

PYTHON_VERSION=3.11
EXPECTED_CUDA_MM=12.8
TORCH_CUDA_TAG=cu128
INSTALL_PSI4=1

# =========================
# Pretty printing helpers
# =========================
hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }
subbanner() { echo "-- $1"; }

banner "Starting HydraGNN environment setup on DCC ($(date))"

# ============================================================
# Module initialization
# ============================================================
banner "Configure DCC Modules"
EXPECTED_CUDA_MM="${EXPECTED_CUDA_MM:-12.8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/module_loads_dcc.sh"
load_dcc_modules "${EXPECTED_CUDA_MM}"

echo "MPI module loaded: ${DCC_MPI_MODULE_LOADED}"
echo "CUDA_HOME:         ${CUDA_HOME:-unset}"

# ============================================================
# Installation root
# ============================================================
banner "Set Base Installation Directory"

# The environment CANNOT live on /work, unlike the Perlmutter layout which keeps
# it inside the repo. The VAST filesystem backing /work rejects filenames
# containing * ? : | (SMB-reserved characters) with EINVAL. psi4's libint package
# ships basis-set files named e.g. "6-31g**.g94", so conda extraction fails there
# with InvalidArchiveError. /hpc/group is a different filesystem and accepts them.
INSTALL_ROOT="${INSTALL_ROOT:-/hpc/group/youlab/${USER}/HydraGNN-Installation-Duke-Compute-Cluster}"
mkdir -p "$INSTALL_ROOT"
echo "All installation components will be contained in: $INSTALL_ROOT"
cd "$INSTALL_ROOT"

# ============================================================
# Cache redirection (DCC $HOME is a hard 25 GB quota)
# ============================================================
banner "Redirect Package Caches Away From \$HOME"

# conda defaults its package cache to ~/.conda/pkgs and pip to ~/.cache/pip. A
# HydraGNN environment plus psi4 will exhaust the DCC home quota and fail
# mid-install with confusing errors, so the large caches are placed under
# INSTALL_ROOT on /work.
export CONDA_PKGS_DIRS="${INSTALL_ROOT}/.conda_pkgs"
export PIP_CACHE_DIR="${INSTALL_ROOT}/.pip_cache"

# TMPDIR is deliberately NOT on /work. Source builds create and delete many small
# files, and on the VAST-backed /work filesystem pip's cleanup of its build
# directory fails with "OSError: [Errno 39] Directory not empty". Node-local /tmp
# has ordinary POSIX unlink semantics. Build temporaries are small and need not
# outlive the job.
export TMPDIR="${TMPDIR_OVERRIDE:-/tmp/hydragnn_build_${USER}_$$}"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
trap 'rm -rf "${TMPDIR}" 2>/dev/null || true' EXIT
echo "CONDA_PKGS_DIRS = $CONDA_PKGS_DIRS"
echo "PIP_CACHE_DIR   = $PIP_CACHE_DIR"
echo "TMPDIR          = $TMPDIR (node-local; see comment above)"

# ============================================================
# Conda shell init + env creation
# ============================================================
banner "Initialize Conda + Create/Activate Environment"

if ! command -v conda >/dev/null 2>&1; then
  echo "❌ conda command not found (Anaconda3 module not loaded?)"
  exit 1
fi

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  # shellcheck disable=SC1090
  eval "$("${CONDA_BASE}/bin/conda" shell.bash hook)" 2>/dev/null || true
fi

VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/hydragnn_venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "Virtual environment path: $VENV_PATH"
echo "Python version: ${PYTHON_VERSION}"

if [[ -d "$VENV_PATH" ]]; then
  echo "Removing existing conda environment at: $VENV_PATH"
  conda deactivate >/dev/null 2>&1 || true
  conda env remove -p "$VENV_PATH" -y || rm -rf "$VENV_PATH"
fi

echo "Creating conda environment at $VENV_PATH with Python $PYTHON_VERSION"
conda create -y -p "$VENV_PATH" python="$PYTHON_VERSION"

conda activate "$VENV_PATH"
echo "Python in use: $(which python)"
python --version

# DCC has no Cray wrappers; the system GCC 11.5 is new enough (>= 9) for any
# torch C++ extension build.
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"

subbanner "Compiler sanity check (must be GCC >= 9)"
echo "CC=$(which ${CC})"
echo "CXX=$(which ${CXX})"
${CXX} --version | head -n 1

# CUDA build env hints. 8.6 = RTX A5000 (Ampere GA102).
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-16}"

# ============================================================
# pip helpers + NumPy pin
# ============================================================
banner "pip Helpers and NumPy Pin (1.26.4)"

PIP_FLAGS=(--upgrade-strategy only-if-needed)

pip_retry() {
  local tries=3 delay=3
  for ((i=1; i<=tries; i++)); do
    if pip install "${PIP_FLAGS[@]}" "$@"; then
      return 0
    fi
    echo "pip install failed (attempt $i/$tries). Retrying in ${delay}s..."
    sleep "$delay"; delay=$((delay*2))
  done
  return 1
}

assert_numpy_1264() {
  python - <<'PY'
import numpy as np
expected="1.26.4"
assert np.__version__==expected, f"NumPy is {np.__version__}, expected {expected}"
PY
}

subbanner "Upgrade pip/setuptools/wheel"
pip_retry --disable-pip-version-check -U pip setuptools wheel

subbanner "Install and pin numpy==1.26.4"
pip_retry "numpy==1.26.4"
assert_numpy_1264

# ============================================================
# Core scientific Python deps
# ============================================================
banner "Install Core Python Packages"

# Versions mirror the working Perlmutter environment
# (references/perlmutter_info/hydragnn_venv/pip-freeze.txt).
# TensorFlow and tensorflow_datasets are intentionally NOT installed: they appear
# in no requirements file, are unreachable from examples/qmugs/, and contend with
# the numpy 1.26.4 pin.
pip_retry click==8.0.0
pip_retry ninja
pip_retry cmake
pip_retry astunparse
pip_retry expecttest
pip_retry hypothesis
pip_retry numpy==1.26.4
pip_retry psutil==7.1.0
pip_retry pyyaml
pip_retry requests
pip_retry setuptools
pip_retry typing-extensions
pip_retry sympy==1.14.0
pip_retry filelock
pip_retry networkx
pip_retry jinja2
pip_retry tqdm==4.67.3
pip_retry types-dataclasses
pip_retry scipy==1.14.1
pip_retry matscipy
pip_retry matplotlib==3.10.6
pip_retry pyparsing
pip_retry build
pip_retry Cython
pip_retry tensorboard==2.20.0
pip_retry scikit-learn==1.5.1
pip_retry pytest
pip_retry ase==3.26.0
pip_retry rdkit==2026.3.2
pip_retry jarvis-tools
pip_retry pymatgen
pip_retry igraph
pip_retry mendeleev==0.16.0
pip_retry lmdb
pip_retry h5py==3.14.0
pip_retry vesin==0.4.2

# Reached directly by examples/qmugs/ (scikit-image for marching cubes in the
# density isosurface figures; pandas for EDA and metrics tables).
pip_retry scikit-image==0.26.0
pip_retry pandas==3.0.3
pip_retry orjson==3.11.9
assert_numpy_1264

# ============================================================
# CUDA-aware PyTorch (pip wheels)
# ============================================================
banner "Install CUDA PyTorch (Before PyG)"

# Match CUDA/12.8 => cu128 wheels. The A5000 driver (595.x, CUDA 13.2 capable)
# is forward compatible with a 12.8 runtime.
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu128}"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

subbanner "Install PyTorch from ${PYTORCH_INDEX_URL}"
pip_retry --index-url "${PYTORCH_INDEX_URL}" torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
assert_numpy_1264

# NOTE: on a DCC login node there is no GPU and no driver, so cuda available will
# report False here. That is expected and is NOT an installation failure; the real
# check must run inside an srun/sbatch job on the youlab-gpu partition.
python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu name =", torch.cuda.get_device_name(0))
else:
    print("(no GPU visible - expected on a login node)")
PY

subbanner "Install extra torch-adjacent packages without touching torch"
pip_retry torch-ema==0.3 torchmetrics==1.4.0 --no-deps

# ============================================================
# PyTorch-Geometric stack (PREBUILT WHEELS on DCC)
# ============================================================
banner "PyTorch-Geometric Stack (Prebuilt Wheels; glibc 2.34 on DCC)"

# Perlmutter forced source builds to dodge a GLIBC_2.33 requirement. DCC runs
# AlmaLinux 9 with glibc 2.34, so the official wheels load without issue. If the
# wheel index is unavailable or a wheel is missing, fall back to a source build.
PYG_WHEEL_INDEX="https://data.pyg.org/whl/torch-2.8.0+${TORCH_CUDA_TAG}.html"

subbanner "Uninstall any existing PyG components to avoid mixing wheels/source"
pip uninstall -y pyg-lib torch-sparse torch-scatter torch-cluster torch-spline-conv torch-geometric >/dev/null 2>&1 || true

install_pyg_component() {
  local pkg_spec="$1"
  subbanner "Installing ${pkg_spec} (wheel first, source fallback)"
  if pip_retry --no-cache-dir -f "${PYG_WHEEL_INDEX}" "${pkg_spec}"; then
    return 0
  fi
  echo "WARNING: wheel install of ${pkg_spec} failed; falling back to source build."
  pip_retry --no-cache-dir --no-binary=:all: --no-build-isolation "${pkg_spec}"
}

install_pyg_component "torch-scatter==2.1.2"
install_pyg_component "torch-sparse==0.6.18"
install_pyg_component "torch-cluster==1.6.3"
install_pyg_component "torch-spline-conv==1.2.2"

# pyg-lib is not installed: it is absent from the working Perlmutter environment
# and HydraGNN runs without it.

subbanner "Install torch-geometric (pure python wrapper package)"
pip_retry torch-geometric==2.6.1
assert_numpy_1264

# Versions pinned to what the Perlmutter environment actually resolved to.
# NOTE: requirements-torch.txt specifies e3nn==0.5.1, but the Perlmutter script
# installed e3nn unpinned and resolved to 0.6.0. We match the environment that
# actually ran, not the stale pin.
subbanner "Install e3nn and openequivariance"
pip_retry e3nn==0.6.0
pip_retry openequivariance==0.6.8
assert_numpy_1264

subbanner "PyG import sanity check"
python - <<'PY'
import torch
import torch_geometric
print("torch and torch_geometric simultaneous import success", torch.__version__, torch_geometric.__version__)
PY

python - <<'PY'
mods = ["torch_sparse", "torch_scatter", "torch_cluster", "torch_spline_conv"]
for m in mods:
    try:
        __import__(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAIL ({e})")
PY

# ============================================================
# mpi4py
# ============================================================
banner "mpi4py (v4.1.1)"

# Built from sdist against the loaded OpenMPI module. Perlmutter used the Cray
# wrapper (CC=cc MPICC=cc); DCC has no such wrapper, so mpicc is used directly.
# --no-binary is essential: a prebuilt mpi4py wheel bundles its own MPI and will
# not interoperate with the cluster's OpenMPI/InfiniBand stack.
#
# NOTE: this links against the non-CUDA-aware OpenMPI/4.1.6. The CUDA-aware
# Infiniband/OpenMPI/5.0.2-CUDA cannot be linked on DCC (its UCX dependency is
# missing); see the comment block in module_loads_dcc.sh.
subbanner "Building mpi4py against ${DCC_MPI_MODULE_LOADED} ($(command -v mpicc))"
MPICC="$(command -v mpicc)" pip_retry --no-cache-dir --no-binary=mpi4py "mpi4py==4.1.1"

python - <<'PY'
import mpi4py
from mpi4py import MPI
print("mpi4py:", mpi4py.__version__, "at", mpi4py.__file__)
print("MPI library version:", MPI.Get_library_version().strip().splitlines()[0])
PY
assert_numpy_1264

# ============================================================
# Psi4
# ============================================================
# Psi4 is conda-only and is the single most disruptive step in this script: its
# conda-forge solve can displace pip-installed packages (on Perlmutter it pulled
# networkx, PyYAML and typing_extensions over to conda builds). It is therefore
# installed LAST, after everything pip-managed is in place.
INSTALL_PSI4="${INSTALL_PSI4:-1}"
if [[ "$INSTALL_PSI4" -eq 1 ]]; then
  banner "Psi4 (conda-forge)"

  # First attempt: ask the solver to leave the existing environment alone. If
  # conda can satisfy psi4 without disturbing numpy, the pin survives untouched.
  subbanner "Attempt 1: solve with --freeze-installed (preserves numpy 1.26.4)"
  if conda install -y -p "$VENV_PATH" -c conda-forge --freeze-installed psi4 python=3.11; then
    echo "psi4 installed with the existing environment frozen."
  else
    echo "WARNING: frozen solve failed; retrying with an unconstrained solve."
    subbanner "Attempt 2: unconstrained solve (numpy may be displaced)"
    conda install -y -p "$VENV_PATH" -c conda-forge psi4 python=3.11
  fi

  # Restore the pinned scientific stack over psi4's conda solve.
  #
  # psi4 does not only displace numpy. On DCC its solve pulled conda scipy 1.17.1
  # over the pinned pip scipy 1.14.1, which broke scikit-learn and torch_cluster
  # with "cannot import name '_promote' from scipy.spatial.transform._rotation":
  # both were compiled against the 1.14 ABI.
  #
  # The working Perlmutter environment shows numpy, scipy AND scikit-learn all
  # marked pypi_0 in conda-list.txt, i.e. all three were pip-reinstalled after
  # psi4. That is the undocumented manual fix-up, reproduced here.
  #
  # This is done unconditionally rather than conditionally: the three are coupled
  # by compiled ABI, so restoring them as a set is both simpler and safer than
  # repairing whichever one happens to trip an assertion first. Reinstalling
  # already-correct versions is cheap and idempotent.
  #
  # It leaves psi4 running against versions it did not solve for. That is unclean,
  # but it is the configuration verified to work for the qmugs task on Perlmutter.
  subbanner "Restore pinned scientific stack over the psi4 solve"
  pip_retry --force-reinstall --no-deps "numpy==1.26.4" "scipy==1.14.1" "scikit-learn==1.5.1"
  # --no-deps above is what keeps the psi4 solve from dragging numpy/scipy back, but it
  # also means scikit-learn's own runtime dependencies are NOT restored. threadpoolctl
  # is the one that matters: the psi4 solve removes it, sklearn imports it at module
  # scope, and so 'import hydragnn' dies. Both versions match the Perlmutter freeze.
  pip_retry "threadpoolctl==3.6.0" "joblib==1.5.3"
  assert_numpy_1264
  python - <<'PY'
import numpy, scipy, sklearn
expected = {"numpy": "1.26.4", "scipy": "1.14.1", "sklearn": "1.5.1"}
actual = {"numpy": numpy.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
for k, v in expected.items():
    assert actual[k] == v, f"{k} is {actual[k]}, expected {v}"
print("pinned stack restored:", actual)
PY

  subbanner "psi4 import check"
  python - <<'PY'
try:
    import psi4
    print("psi4:", psi4.__version__)
except Exception as e:
    print("psi4 import FAILED:", e)
PY
fi

# ============================================================
# Final verification
# ============================================================
banner "Final Verification"

subbanner "Recheck PyTorch"
python - <<'PY'
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu name =", torch.cuda.get_device_name(0))
else:
    print("(no GPU visible - expected on a login node)")
PY

# A failure here must fail the script. Exiting 0 with a broken import would hand
# back an environment that only breaks later, inside a training run.
#
# PYTHONNOUSERSITE=1 is essential and was learned the hard way. Every SLURM script in
# examples/qmugs/scripts/ exports it, so the jobs cannot see ~/.local/lib/pythonX.Y/
# site-packages. Verifying WITHOUT it lets a stray user-site copy of a package satisfy
# the import here while the real job still fails -- exactly how a missing
# threadpoolctl passed this check and then killed a preprocessing run. Verify the
# environment under the same conditions the jobs run under.
subbanner "Import closure reached by examples/qmugs/"
PYTHONNOUSERSITE=1 python - <<'PY'
import sys
mods = [
    "numpy", "scipy", "matplotlib", "pandas", "torch", "torch_geometric",
    "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
    "rdkit", "skimage", "mpi4py", "tqdm", "ase", "e3nn", "sklearn",
    "mendeleev", "vesin", "psi4", "threadpoolctl", "joblib",
]
bad = []
for m in mods:
    try:
        __import__(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAIL ({e})")
        bad.append(m)
print()
if bad:
    print("FAILED IMPORTS:", bad)
    sys.exit(1)
print("FAILED IMPORTS: none")
PY

subbanner "Final numpy check"
assert_numpy_1264
python -c "import numpy; print('numpy', numpy.__version__)"

# ============================================================
# Final Summary
# ============================================================
banner "Final Summary"
cat <<EOF
Base install:        $INSTALL_ROOT
Virtual environment: $VENV_PATH

Modules baseline:
  CUDA/${EXPECTED_CUDA_MM}
  ${DCC_MPI_MODULE_LOADED}
  cmake/3.28.3
  Anaconda3/2024.02

PyTorch:
  CUDA wheel tag:    ${TORCH_CUDA_TAG}
  Index URL:         ${PYTORCH_INDEX_URL}
  Arch list:         ${TORCH_CUDA_ARCH_LIST} (RTX A5000, Ampere GA102)

PyTorch-Geometric:
  Installed from prebuilt wheels: ${PYG_WHEEL_INDEX}
  pyg-lib:           not installed (absent from the Perlmutter environment too)

mpi4py:              built from sdist against ${DCC_MPI_MODULE_LOADED}
Psi4:                $( [[ "${INSTALL_PSI4}" -eq 1 ]] && echo "installed via conda-forge" || echo "skipped" )

NOT installed (out of scope for examples/qmugs/):
  ADIOS2, DDStore, DeepHyper, DeepSpeed, GPTL/gptl4py, TensorFlow

Remaining manual step:
  conda activate ${VENV_PATH}
  python -m pip install -e . --no-deps
EOF

echo "✅ HydraGNN-Installation-Duke-Compute-Cluster environment setup complete!"
echo ""
print_dcc_activation_instructions "${EXPECTED_CUDA_MM}" "${VENV_PATH}"
