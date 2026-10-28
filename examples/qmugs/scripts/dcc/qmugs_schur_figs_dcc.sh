#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-SchurFigs
#SBATCH -o slurm_output/figs/schur-figs-job-%j.out
#SBATCH -e slurm_output/figs/schur-figs-job-%j.out
#SBATCH -t 01:00:00
#SBATCH -p youlab-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
# aa25-19 has a faulty GPU 3, which makes torch report ZERO usable devices on that node.
# REMOVE this line once Duke OIT repairs the node.
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19

# Figures for a finished inference run (figs.py, standalone), for Schur decoder and dense
# models alike. This is the counterpart of scripts/dcc/qmugs_figs_dcc.sh, which keeps a
# hard-coded "SET THESE" block that overrides the environment; here every value comes from the
# environment and only LOG_DIR is required.
#
# What differs from qmugs_figs_dcc.sh, and why:
#
#   - One GPU. The sampled conformers are re-run through the model, and a free-running Schur
#     decoder is a sequential recursion per conformer, run twice (its own routing and teacher
#     forced, for the block-error figure).
#
#   - HYDRAGNN_ROOT defaults to /work/$USER/HydraGNN, and the job stops if hydragnn is
#     imported from anywhere else.
#
# Submit from examples/qmugs/, e.g.
#
#   LOG_DIR=/work/$USER/HydraGNN/logs/<inference run> \
#       RUN_SAMPLE_FIGS=1 RUN_ELECTRON_DENSITY_FIGS=1 RUN_DIPOLE_FIGS=1 \
#       sbatch scripts/dcc/qmugs_schur_figs_dcc.sh

HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/work/$USER/HydraGNN}
echo "HydraGNN root: $HYDRAGNN_ROOT"
VENV_PATH=${VENV_PATH:-/hpc/group/youlab/$USER/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv}
EXAMPLE_DIR="$HYDRAGNN_ROOT/examples/qmugs"

if [ -z "${LOG_DIR:-}" ]; then
    echo "ERROR: set LOG_DIR to the inference run directory"
    exit 1
fi
ARTIFACTS_DIR=${ARTIFACTS_DIR:-}
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
FIGS_MATRIX_POOL_SIZE=${FIGS_MATRIX_POOL_SIZE:-1}
FIGS_METRICS_NATOMS_BIN_SIZE=${FIGS_METRICS_NATOMS_BIN_SIZE:-}
FIGS_METRICS_MAX_NATOMS=${FIGS_METRICS_MAX_NATOMS:-}

# --- DCC module + conda setup, as in qmugs_schur_inference_dcc.sh ---
source "$HYDRAGNN_ROOT/installation_other/module_loads_dcc.sh"
load_dcc_modules 12.8
unset CONDA_SHLVL CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
CONDA_BASE=$(conda info --base 2>/dev/null)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$VENV_PATH"
if [[ "$(command -v python)" != "$VENV_PATH"/* ]]; then
    echo "ERROR: conda activate did not take effect: $(command -v python)"
    exit 1
fi

cd "$HYDRAGNN_ROOT" || exit 1
export PYTHONPATH=$PWD:$PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=16
export HYDRAGNN_USE_FSDP=0

# The venv's editable install points at /work/$USER/HydraGNN and PYTHONPATH takes precedence over
# it, so with another HYDRAGNN_ROOT the run must not silently use the main tree; check where
# hydragnn comes from.
HYDRAGNN_IMPORTED=$(python -c "import hydragnn, os; print(os.path.realpath(hydragnn.__file__))")
echo "hydragnn imported from: $HYDRAGNN_IMPORTED"
if [[ "$HYDRAGNN_IMPORTED" != "$(realpath "$HYDRAGNN_ROOT")"/* ]]; then
    echo "ERROR: hydragnn is not imported from HYDRAGNN_ROOT=$HYDRAGNN_ROOT"
    exit 1
fi
nvidia-smi -L

FIGS_ARGS=(--log_dir="$LOG_DIR" --seed="$FIGS_SEED" --matrix_pool_size="$FIGS_MATRIX_POOL_SIZE"
           --num_random_molecules="$FIGS_NUM_RANDOM_MOLECULES"
           --num_best_conformers="$FIGS_NUM_BEST_CONFORMERS"
           --num_smallest_molecules="$FIGS_NUM_SMALLEST_MOLECULES")
[ -n "$ARTIFACTS_DIR" ] && FIGS_ARGS+=(--artifacts_dir="$ARTIFACTS_DIR")
[ "$RUN_METRICS_FIGS" = "1" ] && FIGS_ARGS+=(--run_metrics_visuals)
[ "$RUN_SAMPLE_FIGS" = "1" ] && FIGS_ARGS+=(--run_sample_visuals)
[ "$RUN_ELECTRON_DENSITY_FIGS" = "1" ] && FIGS_ARGS+=(--run_electron_density_visuals)
[ "$RUN_DIPOLE_FIGS" = "1" ] && FIGS_ARGS+=(--run_dipole_visuals)
[ "$FIGS_SAMPLE_MODE" = "best_conformers" ] && FIGS_ARGS+=(--best_conformers)
[ "$FIGS_SAMPLE_MODE" = "smallest_molecules" ] && FIGS_ARGS+=(--smallest_molecules)
[ -n "$FIGS_METRICS_NATOMS_BIN_SIZE" ] && FIGS_ARGS+=(--metrics_natoms_bin_size="$FIGS_METRICS_NATOMS_BIN_SIZE")
[ -n "$FIGS_METRICS_MAX_NATOMS" ] && FIGS_ARGS+=(--metrics_max_natoms="$FIGS_METRICS_MAX_NATOMS")
if [ -n "$FIGS_CHEMBL_IDS" ]; then
    FIGS_ARGS+=(--chembl_ids)
    for chembl_id in ${FIGS_CHEMBL_IDS//,/ }; do
        FIGS_ARGS+=("$chembl_id")
    done
fi

echo "python -u $EXAMPLE_DIR/figs.py ${FIGS_ARGS[*]}"
time python -u "$EXAMPLE_DIR/figs.py" "${FIGS_ARGS[@]}"
