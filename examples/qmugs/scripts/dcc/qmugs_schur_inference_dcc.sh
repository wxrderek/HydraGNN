#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-SchurInference
#SBATCH -o slurm_output/schur_inference/schur-inference-job-%j.out
#SBATCH -e slurm_output/schur_inference/schur-inference-job-%j.out
#SBATCH -t 04:00:00
#SBATCH -p youlab-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -c 12
#SBATCH --mem=256G
# aa25-19 has a faulty GPU 3, which makes torch report ZERO usable devices on that node.
# REMOVE this line once Duke OIT repairs the node.
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19

# Inference with a trained Schur decoder model. This is the Schur counterpart of
# scripts/dcc/qmugs_inference_dcc.sh and runs the same qmugs_densmat_inference.py, which
# recognises the decoder from the SchurDecoder block of the model's saved config.json.
#
# What differs from qmugs_inference_dcc.sh, and why:
#
#   - SCHUR_ROUTING (default free_running) is passed as --schur_routing. Free running feeds
#     D_hat_{k-1} back into the recursion, so no ground truth enters the prediction;
#     teacher_forcing feeds D_{k-1} as in training and is kept for checking the inference path
#     against the logged training loss.
#
#   - SCHUR_TRACE_NORMALIZATION (default config) is passed as --schur_trace_normalization.
#     config keeps the setting the model was trained with; on or off replaces it, and on adds
#     -tn to the run name.
#
#   - Every value comes from the environment. There is no hard-coded "SET THESE" block.
#     Figures are off unless RUN_FIGS=1; the FIGS_* variables below map onto the --figs_*
#     options of qmugs_densmat_inference.py.
#
#   - MODEL_DIR and CHECKPOINT_PATH are required. The default checkpoint search sorts file
#     names, which puts _epoch_10.pk before _epoch_9.pk, so the checkpoint is always named.
#
#   - One node. The recursion is sequential per conformer; 4 ranks share the test set.
#
#   - HYDRAGNN_ROOT defaults to /work/$USER/HydraGNN, and the job stops if hydragnn is
#     imported from anywhere else.
#
# Submit from examples/qmugs/, e.g.
#
#   MODEL_DIR=/work/$USER/HydraGNN/logs/<run> CHECKPOINT_PATH=<run>_epoch_<n>.pk \
#       sbatch scripts/dcc/qmugs_schur_inference_dcc.sh

function cmd() {
    echo "$@"
    time "$@"
}

# --- Paths (override with environment variables if needed) ---
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/work/$USER/HydraGNN}
echo "HydraGNN root: $HYDRAGNN_ROOT"
VENV_PATH=${VENV_PATH:-/hpc/group/youlab/$USER/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv}
EXAMPLE_DIR="$HYDRAGNN_ROOT/examples/qmugs"

if [ -z "${MODEL_DIR:-}" ] || [ -z "${CHECKPOINT_PATH:-}" ]; then
    echo "ERROR: set MODEL_DIR and CHECKPOINT_PATH"
    exit 1
fi
DATASET_BASEDIR=${DATASET_BASEDIR:-/work/$USER/qmugs/QMugs001_max800_delta_uptr_decodercmp_overlap.pickle}
QMUGS_DATA_DIR=${QMUGS_DATA_DIR:-/work/$USER/qmugs}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-/work/$USER/artifacts/DCC_MODEL-QMugs001_max800_delta_uptr_decodercmp_DATA-QMugs001_max800_delta_uptr_decodercmp}
SCHUR_ROUTING=${SCHUR_ROUTING:-free_running}
SCHUR_TRACE_NORMALIZATION=${SCHUR_TRACE_NORMALIZATION:-config}
SCHUR_RUN_SUFFIX=""
if [ "$SCHUR_TRACE_NORMALIZATION" = "on" ]; then
    SCHUR_RUN_SUFFIX="-tn"
fi
PERC_DOWNSAMPLE_TEST=${PERC_DOWNSAMPLE_TEST:-1.0}
BATCH_SIZE=${BATCH_SIZE:-1}
RUN_DOWNSTREAM_CALCULATION=${RUN_DOWNSTREAM_CALCULATION:-0}
DOWNSTREAM_CALCULATIONS=${DOWNSTREAM_CALCULATIONS:-}
RUN_FIGS=${RUN_FIGS:-0}
RUN_SAMPLE_FIGS=${RUN_SAMPLE_FIGS:-0}
RUN_ELECTRON_DENSITY_FIGS=${RUN_ELECTRON_DENSITY_FIGS:-0}
RUN_DIPOLE_FIGS=${RUN_DIPOLE_FIGS:-0}
FIGS_NUM_RANDOM_MOLECULES=${FIGS_NUM_RANDOM_MOLECULES:-}
FIGS_MATRIX_POOL_SIZE=${FIGS_MATRIX_POOL_SIZE:-}
FIGS_METRICS_NATOMS_BIN_SIZE=${FIGS_METRICS_NATOMS_BIN_SIZE:-}
FIGS_METRICS_MAX_NATOMS=${FIGS_METRICS_MAX_NATOMS:-}

# --- DCC module + conda setup, as in qmugs_inference_dcc.sh ---
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

echo "Virtual environment path: $VENV_PATH"
conda activate "$VENV_PATH"

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
python -c "import psi4; print(psi4.__version__)"

# The venv's editable install points at /work/$USER/HydraGNN and PYTHONPATH takes precedence over
# it, so with another HYDRAGNN_ROOT the run must not silently use the main tree; check where
# hydragnn comes from.
HYDRAGNN_IMPORTED=$(python -c "import hydragnn, os; print(os.path.realpath(hydragnn.__file__))")
echo "hydragnn imported from: $HYDRAGNN_IMPORTED"
if [[ "$HYDRAGNN_IMPORTED" != "$(realpath "$HYDRAGNN_ROOT")"/* ]]; then
    echo "ERROR: hydragnn is not imported from HYDRAGNN_ROOT=$HYDRAGNN_ROOT"
    exit 1
fi
python -c "from hydragnn.decoders import SchurDecoderModel; print('schur decoder importable')"

echo "===== GPUs ====="
nvidia-smi -L

# --- MPI/runtime envs, as in qmugs_inference_dcc.sh ---
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=8
export HYDRAGNN_NUM_WORKERS=1
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

export HYDRAGNN_TRACE_LEVEL=0
export HYDRAGNN_MAX_NUM_BATCH=1000
export HYDRAGNN_TASK_PARALLEL_PROPORTIONAL_SPLIT=0

export HYDRAGNN_DDSTORE_METHOD=1
export HYDRAGNN_CUSTOM_DATALOADER=1

MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_IP=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
if [ -z "$MASTER_IP" ]; then
    MASTER_IP="$MASTER_HOST"
fi
export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT=${MASTER_PORT:-29501}
export HYDRAGNN_MASTER_ADDR=$MASTER_ADDR
export HYDRAGNN_MASTER_PORT=$MASTER_PORT

# the decoder has only ever run under DDP; inference wraps nothing
export HYDRAGNN_USE_FSDP=0

DOWNSTREAM_CALCULATION_ARGS=()
if [ "$RUN_DOWNSTREAM_CALCULATION" = "1" ]; then
    DOWNSTREAM_CALCULATION_ARGS+=(--run_downstream_calculation)
    if [ -n "$DOWNSTREAM_CALCULATIONS" ]; then
        DOWNSTREAM_CALCULATION_ARGS+=(--downstream_calculations)
        for calculation in $DOWNSTREAM_CALCULATIONS; do
            DOWNSTREAM_CALCULATION_ARGS+=("$calculation")
        done
    fi
fi

FIGS_ARGS=()
if [ "$RUN_FIGS" = "1" ]; then
    FIGS_ARGS+=(--run_figs)
    [ "$RUN_SAMPLE_FIGS" = "1" ] && FIGS_ARGS+=(--run_sample_figs)
    [ "$RUN_ELECTRON_DENSITY_FIGS" = "1" ] && FIGS_ARGS+=(--run_electron_density_figs)
    [ "$RUN_DIPOLE_FIGS" = "1" ] && FIGS_ARGS+=(--run_dipole_figs)
    [ -n "$FIGS_NUM_RANDOM_MOLECULES" ] && FIGS_ARGS+=(--figs_num_random_molecules="$FIGS_NUM_RANDOM_MOLECULES")
    [ -n "$FIGS_MATRIX_POOL_SIZE" ] && FIGS_ARGS+=(--figs_matrix_pool_size="$FIGS_MATRIX_POOL_SIZE")
    [ -n "$FIGS_METRICS_NATOMS_BIN_SIZE" ] && FIGS_ARGS+=(--figs_metrics_natoms_bin_size="$FIGS_METRICS_NATOMS_BIN_SIZE")
    [ -n "$FIGS_METRICS_MAX_NATOMS" ] && FIGS_ARGS+=(--figs_metrics_max_natoms="$FIGS_METRICS_MAX_NATOMS")
fi

# Launched with mpirun, NOT srun: OpenMPI/4.1.6 on DCC cannot be direct-launched by srun.
# See scripts/dcc/mpi_rank_env.sh for why the per-rank shim is needed.
cmd mpirun -np $((SLURM_JOB_NUM_NODES*4)) --map-by ppr:4:node --bind-to none \
    "$EXAMPLE_DIR/scripts/dcc/mpi_rank_env.sh" \
    python -u "$EXAMPLE_DIR/qmugs_densmat_inference.py" \
    --model_dir="$MODEL_DIR" \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --basedir="$DATASET_BASEDIR" \
    --qmugs_data_dir="$QMUGS_DATA_DIR" \
    --artifacts_dir="$ARTIFACTS_DIR" \
    --log=qmugs-schur-inference-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-DCC-$SCHUR_ROUTING$SCHUR_RUN_SUFFIX \
    --batch_size=$BATCH_SIZE \
    --schur_routing="$SCHUR_ROUTING" \
    --schur_trace_normalization="$SCHUR_TRACE_NORMALIZATION" \
    --perc_downsample_test="$PERC_DOWNSAMPLE_TEST" \
    "${DOWNSTREAM_CALCULATION_ARGS[@]}" \
    "${FIGS_ARGS[@]}" \
    --electron_density_method="cubeprop" \
    --manual_density_num_grid_points=100
