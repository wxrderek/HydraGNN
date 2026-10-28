#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-AddOverlap
#SBATCH -o slurm_output/dataload/add-overlap-job-%j.out
#SBATCH -e slurm_output/dataload/add-overlap-job-%j.out
#SBATCH -t 04:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -c 2
#SBATCH --mem=256G

# Adds the overlap matrix and the number of electrons to an existing pickle dataset, with
# utils/add_overlap.py. The source dataset is only read. The new dataset gets the same
# samples in the same order plus overlap_upper_triangle and n_electrons, and the meta files
# are copied unchanged. With ADD_OVERLAP_CHECK_ONLY=1 only the ground-truth checks run and
# nothing is written apart from the log directory.
#
# Usage, from examples/qmugs:
#   sbatch scripts/dcc/qmugs_add_overlap_dcc.sh
#   ADD_OVERLAP_CHECK_ONLY=1 sbatch scripts/dcc/qmugs_add_overlap_dcc.sh
#
# No GPU is used. 24 ranks with 2 cores each fill the 48-core node.

function cmd() {
    echo "$@"
    time "$@"
}

# --- Paths (override with environment variables if needed) ---
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/work/$USER/HydraGNN}
echo "HydraGNN root: $HYDRAGNN_ROOT"
VENV_PATH=${VENV_PATH:-/hpc/group/youlab/$USER/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv}
EXAMPLE_DIR="$HYDRAGNN_ROOT/examples/qmugs"

# --- DCC module + conda setup, as in scripts/dcc/qmugs_schur_pre_dcc.sh ---
source "$HYDRAGNN_ROOT/installation_other/module_loads_dcc.sh"
load_dcc_modules 12.8

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
    exit 1
fi

conda activate "$VENV_PATH"

if [[ "$(command -v python)" != "$VENV_PATH"/* ]]; then
    echo "ERROR: conda activate did not take effect."
    echo "       python resolves to: $(command -v python)"
    echo "       expected it under:  $VENV_PATH"
    exit 1
fi

export PYTHONPATH=$HYDRAGNN_ROOT:$PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

which python
python -c "import psi4; print(psi4.__version__)"

# ----------------------------------------------------------------------------------------------------
# SET THESE
#
# ADD_OVERLAP_ELECTRON_COUNT_TOLERANCE asserts the electron count of the wavefunction density
# matrix per conformer. Leave it empty for a run that only records; set it from the
# check-only run before the run that writes the dataset.

ADD_OVERLAP_SOURCE=${ADD_OVERLAP_SOURCE:-/work/$USER/qmugs/QMugs001_max800_delta_uptr_decodercmp.pickle}
ADD_OVERLAP_DESTINATION=${ADD_OVERLAP_DESTINATION:-/work/$USER/qmugs/QMugs001_max800_delta_uptr_decodercmp_overlap.pickle}
ADD_OVERLAP_CHECK_ONLY=${ADD_OVERLAP_CHECK_ONLY:-0}
ADD_OVERLAP_MAX_SAMPLES=${ADD_OVERLAP_MAX_SAMPLES:-}
ADD_OVERLAP_ELECTRON_COUNT_TOLERANCE=${ADD_OVERLAP_ELECTRON_COUNT_TOLERANCE:-}
ADD_OVERLAP_LOG_DIR=${ADD_OVERLAP_LOG_DIR:-$HYDRAGNN_ROOT/logs/qmugs-add-overlap-$SLURM_JOB_ID}

ADD_OVERLAP_ARGS=()
if [ "$ADD_OVERLAP_CHECK_ONLY" = "1" ]; then
    ADD_OVERLAP_ARGS+=(--check_only)
else
    ADD_OVERLAP_ARGS+=(--destination="$ADD_OVERLAP_DESTINATION")
fi
if [ -n "$ADD_OVERLAP_MAX_SAMPLES" ]; then
    ADD_OVERLAP_ARGS+=(--max_samples="$ADD_OVERLAP_MAX_SAMPLES")
fi
if [ -n "$ADD_OVERLAP_ELECTRON_COUNT_TOLERANCE" ]; then
    ADD_OVERLAP_ARGS+=(--electron_count_tolerance="$ADD_OVERLAP_ELECTRON_COUNT_TOLERANCE")
fi

# run from examples/qmugs so that the utils package resolves
cd "$EXAMPLE_DIR" || exit 1

cmd mpirun -np $((SLURM_JOB_NUM_NODES*24)) --map-by ppr:24:node --bind-to none \
    "$EXAMPLE_DIR/scripts/dcc/mpi_rank_env.sh" \
    python -u -m utils.add_overlap \
    --source="$ADD_OVERLAP_SOURCE" \
    --log_dir="$ADD_OVERLAP_LOG_DIR" \
    "${ADD_OVERLAP_ARGS[@]}"
