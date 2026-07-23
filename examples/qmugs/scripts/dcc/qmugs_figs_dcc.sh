#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-Figs
#SBATCH -o figs-job-%j.out
#SBATCH -e figs-job-%j.out
#SBATCH -t 00:20:00
#SBATCH -p youlab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --mem=128G

# DCC counterpart of scripts/perlmutter/qmugs_figs_perlmutter.sh.
#
# Header differences from Perlmutter, none of which change what the Python sees:
#   - '-C cpu -q debug' become '-p youlab' (the 48-core CPU node). If it is busy,
#     '-p common' or '-p youlab-gpu' also work; plotting needs no GPU.
#   - '-c 32' is kept: this is a single-task job and 32 <= 48 cores.
#   - '--mem' is explicit; DCC defaults to 2 GB/core.

function cmd() {
    echo "$@"
    time "$@"
}

# --- Paths (override with environment variables if needed) ---
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/work/$USER/HydraGNN}
echo "HydraGNN root: $HYDRAGNN_ROOT"
# The environment lives OUTSIDE the repository. The VAST filesystem backing /work
# rejects filenames containing * ? : |, and psi4's libint ships basis sets such as
# 6-31g**.g94, so it cannot be installed under /work.
VENV_PATH=${VENV_PATH:-/hpc/group/youlab/$USER/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv}
EXAMPLE_DIR="$HYDRAGNN_ROOT/examples/qmugs"
LOG_DIR=${LOG_DIR:-$HYDRAGNN_ROOT/logs/QMugs_inference}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-}

# --- Figure configs ---
# Metric figures always run; these flags enable sampled visualizations
RUN_METRICS_FIGS=${RUN_METRICS_FIGS:-1}
RUN_SAMPLE_FIGS=${RUN_SAMPLE_FIGS:-0}
RUN_ELECTRON_DENSITY_FIGS=${RUN_ELECTRON_DENSITY_FIGS:-0}
RUN_DIPOLE_FIGS=${RUN_DIPOLE_FIGS:-0}
FIGS_SAMPLE_MODE=${FIGS_SAMPLE_MODE:-random_molecules}
FIGS_NUM_RANDOM_MOLECULES=${FIGS_NUM_RANDOM_MOLECULES:-2}
FIGS_NUM_BEST_CONFORMERS=${FIGS_NUM_BEST_CONFORMERS:-2}
FIGS_NUM_SMALLEST_MOLECULES=${FIGS_NUM_SMALLEST_MOLECULES:-2}
FIGS_CHEMBL_IDS=${FIGS_CHEMBL_IDS:-}
FIGS_SEED=${FIGS_SEED:-0}
FIGS_DENSITY_POINT_FILTER=${FIGS_DENSITY_POINT_FILTER:-top_percentile}
FIGS_DENSITY_TOP_PERCENTILE=${FIGS_DENSITY_TOP_PERCENTILE:-99.0}
FIGS_DENSITY_ABSOLUTE_CUTOFF=${FIGS_DENSITY_ABSOLUTE_CUTOFF:-1.0e-4}
FIGS_DENSITY_MAX_POINTS=${FIGS_DENSITY_MAX_POINTS:-5000}
FIGS_DENSITY_ISOSURFACE_PERCENTILE=${FIGS_DENSITY_ISOSURFACE_PERCENTILE:-99.0}
FIGS_DENSITY_ISOSURFACE_ABSOLUTE_LEVEL=${FIGS_DENSITY_ISOSURFACE_ABSOLUTE_LEVEL:-}
FIGS_SHOW_3D_AXES=${FIGS_SHOW_3D_AXES:-0}
FIGS_HIDE_LEGEND=${FIGS_HIDE_LEGEND:-0}
FIGS_MATRIX_POOL_SIZE=${FIGS_MATRIX_POOL_SIZE:-1}
FIGS_METRICS_MAX_NATOMS=${FIGS_METRICS_MAX_NATOMS:-}
FIGS_METRICS_MAX_DENSITY_MATRIX_DIM=${FIGS_METRICS_MAX_DENSITY_MATRIX_DIM:-}
FIGS_METRICS_NATOMS_BIN_SIZE=${FIGS_METRICS_NATOMS_BIN_SIZE:-}
FIGS_METRICS_DENSITY_MATRIX_DIM_BIN_SIZE=${FIGS_METRICS_DENSITY_MATRIX_DIM_BIN_SIZE:-}

# --- DCC module + conda setup ---
# load_dcc_modules begins with 'module purge' and then loads CUDA, OpenMPI, cmake and
# Anaconda3. It MUST run before any conda activation: the purge unloads Anaconda3 and
# tears down the conda shell function, which would silently leave us on system Python.
source "$HYDRAGNN_ROOT/installation_other/module_loads_dcc.sh"
load_dcc_modules 12.8

# Clear any conda state inherited from the calling shell. If the caller already had
# an environment active, CONDA_PREFIX/CONDA_SHLVL are inherited here, but the
# 'module purge' above tore down the conda that set them. 'conda activate' would then
# see CONDA_PREFIX already equal to the target, treat it as a no-op reactivation, and
# leave the module Anaconda3 python first on PATH -- every import fails while the
# script still reports success.
unset CONDA_SHLVL CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found."
    exit 1
fi

CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    eval "$($CONDA_BASE/bin/conda shell.bash hook)"
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: VENV_PATH does not exist: $VENV_PATH"
    echo "Set VENV_PATH to your DCC HydraGNN conda env path."
    exit 1
fi

echo "Virtual environment path: $VENV_PATH"
conda activate "$VENV_PATH"

# Assert the activation actually took effect (see the CONDA_* note above).
if [[ "$(command -v python)" != "$VENV_PATH"/* ]]; then
    echo "ERROR: conda activate did not take effect."
    echo "       python resolves to: $(command -v python)"
    echo "       expected it under:  $VENV_PATH"
    exit 1
fi

cd "$HYDRAGNN_ROOT" || exit 1
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== Module List ====="
module list

echo "===== Check ====="
which python
python -c "import torch; print(torch.__version__, torch.__file__)"
python -c "import matplotlib; print(matplotlib.__version__, matplotlib.__file__)"
python -c "import rdkit; print(rdkit.__version__)"
python -c "import psi4; print(psi4.__version__)"
python -c "import hydragnn"

echo "===== LD_LIBRARY_PATH ====="
echo "$LD_LIBRARY_PATH" | tr ':' '\n'

# --- MPI/runtime envs ---
# The MPICH_* variables are Cray MPICH settings and are inert under DCC's OpenMPI.
# They are kept so the two scripts stay line-for-line comparable.
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MPICH_GPU_SUPPORT_ENABLED=0
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=8
export HYDRAGNN_NUM_WORKERS=1
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1
export HYDRAGNN_TRACE_LEVEL=0
export HYDRAGNN_MAX_NUM_BATCH=1000
export HYDRAGNN_CUSTOM_DATALOADER=1

# ----------------------------------------------------------------------------------------------------
# SET THESE
#
# LOG_DIR still names a Perlmutter inference run. Point it at a DCC inference run
# before using this script. Perlmutter /pscratch/sd/w/wxrderek/artifacts maps to
# /work/$USER/artifacts on DCC.

LOG_DIR=$HYDRAGNN_ROOT/logs/qmugs-inference-50469262-NN1-DCC
ARTIFACTS_DIR=/work/$USER/artifacts/artifacts_model-QMugs0001_max800_delta_uptr_data-QMugs0001_max800_delta_uptr-DCC

RUN_METRICS_FIGS=1

RUN_SAMPLE_FIGS=1
RUN_ELECTRON_DENSITY_FIGS=1
RUN_DIPOLE_FIGS=1
FIGS_SAMPLE_MODE="random_molecules"
# FIGS_NUM_BEST_CONFORMERS=3
# FIGS_NUM_SMALLEST_MOLECULES=5
FIGS_NUM_RANDOM_MOLECULES=2
# FIGS_CHEMBL_IDS=CHEMBL180570

FIGS_SHOW_3D_AXES=0
FIGS_HIDE_LEGEND=0

FIGS_MATRIX_POOL_SIZE=5
FIGS_METRICS_NATOMS_BIN_SIZE=4
FIGS_METRICS_MAX_NATOMS=100


# ----------------------------------------------------------------------------------------------------

FIGS_ARGS=()
if [ -n "$ARTIFACTS_DIR" ]; then
    # cubeprops are read only when both true and predicted files are present
    FIGS_ARGS+=(--artifacts_dir="$ARTIFACTS_DIR")
fi
if [ "$RUN_METRICS_FIGS" = "1" ]; then
    FIGS_ARGS+=(--run_metrics_visuals)
fi
if [ "$RUN_SAMPLE_FIGS" = "1" ]; then
    FIGS_ARGS+=(--run_sample_visuals)
fi
if [ "$RUN_ELECTRON_DENSITY_FIGS" = "1" ]; then
    FIGS_ARGS+=(--run_electron_density_visuals)
fi
if [ "$RUN_DIPOLE_FIGS" = "1" ]; then
    FIGS_ARGS+=(--run_dipole_visuals)
fi

if [ -n "$FIGS_CHEMBL_IDS" ]; then
    FIGS_ARGS+=(--chembl_ids)
    for chembl_id in ${FIGS_CHEMBL_IDS//,/ }; do
        FIGS_ARGS+=("$chembl_id")
    done
elif [ "$FIGS_SAMPLE_MODE" = "best_conformers" ]; then
    FIGS_ARGS+=(--best_conformers)
elif [ "$FIGS_SAMPLE_MODE" = "smallest_molecules" ]; then
    FIGS_ARGS+=(--smallest_molecules)
else
    # random molecule sampling includes all test conformers for selected molecules
    FIGS_ARGS+=(--random_molecules)
fi

# point-cloud settings affect only electron density figures
FIGS_ARGS+=(--num_random_molecules="$FIGS_NUM_RANDOM_MOLECULES")
FIGS_ARGS+=(--num_best_conformers="$FIGS_NUM_BEST_CONFORMERS")
FIGS_ARGS+=(--num_smallest_molecules="$FIGS_NUM_SMALLEST_MOLECULES")
FIGS_ARGS+=(--seed="$FIGS_SEED")
FIGS_ARGS+=(--density_point_filter="$FIGS_DENSITY_POINT_FILTER")
FIGS_ARGS+=(--density_top_percentile="$FIGS_DENSITY_TOP_PERCENTILE")
FIGS_ARGS+=(--density_absolute_cutoff="$FIGS_DENSITY_ABSOLUTE_CUTOFF")
FIGS_ARGS+=(--density_max_points="$FIGS_DENSITY_MAX_POINTS")
FIGS_ARGS+=(--density_isosurface_percentile="$FIGS_DENSITY_ISOSURFACE_PERCENTILE")
if [ -n "$FIGS_DENSITY_ISOSURFACE_ABSOLUTE_LEVEL" ]; then
    FIGS_ARGS+=(--density_isosurface_absolute_level="$FIGS_DENSITY_ISOSURFACE_ABSOLUTE_LEVEL")
fi
if [ "$FIGS_SHOW_3D_AXES" = "1" ]; then
    FIGS_ARGS+=(--show_3d_axes)
fi
if [ "$FIGS_HIDE_LEGEND" = "1" ]; then
    FIGS_ARGS+=(--hide_legend)
fi
FIGS_ARGS+=(--matrix_pool_size="$FIGS_MATRIX_POOL_SIZE")
if [ -n "$FIGS_METRICS_MAX_NATOMS" ]; then
    FIGS_ARGS+=(--metrics_max_natoms="$FIGS_METRICS_MAX_NATOMS")
fi
if [ -n "$FIGS_METRICS_MAX_DENSITY_MATRIX_DIM" ]; then
    FIGS_ARGS+=(--metrics_max_density_matrix_dim="$FIGS_METRICS_MAX_DENSITY_MATRIX_DIM")
fi
if [ -n "$FIGS_METRICS_NATOMS_BIN_SIZE" ]; then
    FIGS_ARGS+=(--metrics_natoms_bin_size="$FIGS_METRICS_NATOMS_BIN_SIZE")
fi
if [ -n "$FIGS_METRICS_DENSITY_MATRIX_DIM_BIN_SIZE" ]; then
    FIGS_ARGS+=(--metrics_density_matrix_dim_bin_size="$FIGS_METRICS_DENSITY_MATRIX_DIM_BIN_SIZE")
fi

# Launched with mpirun, NOT srun: OpenMPI/4.1.6 on DCC cannot be direct-launched by
# srun (no Slurm PMI support), and figs.py initialises MPI even at one rank.
cmd mpirun -np 1 --bind-to none \
    "$EXAMPLE_DIR/scripts/dcc/mpi_rank_env.sh" \
    python -u "$EXAMPLE_DIR/figs.py" \
    --log_dir="$LOG_DIR" \
    "${FIGS_ARGS[@]}"
