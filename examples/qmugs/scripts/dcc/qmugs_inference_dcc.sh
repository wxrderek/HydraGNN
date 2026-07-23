#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-Inference
#SBATCH -o inference-job-%j.out
#SBATCH -e inference-job-%j.out
#SBATCH -t 04:00:00
#SBATCH -p youlab-gpu
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -c 12
#SBATCH --mem=256G
# aa25-19 has a faulty GPU 3 ("Unable to determine the device handle for gpu
# 0000:B1:00.0"), which makes torch report ZERO usable devices on that node.
# Slurm still shows it as healthy and idle, so jobs land there first. REMOVE this
# line once Duke OIT repairs the node.
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19

# DCC counterpart of scripts/perlmutter/qmugs_inference_perlmutter.sh.
#
# Header differences from Perlmutter, none of which change what the Python sees:
#   - '-C gpu -q debug' become '-p youlab-gpu'; DCC has no constraints or debug QOS.
#   - '-c 32' becomes '-c 12'. youlab-gpu nodes have 48 cores, and unlike Perlmutter
#     a DCC allocation is not exclusive: 4 tasks x 12 = 48 is the whole node.
#   - '--gpus-per-task=1' becomes '--gres=gpu:4' with NO gpu binding. This reproduces
#     Perlmutter's '--gpu-bind=none': every rank must see all 4 GPUs, because
#     hydragnn/utils/distributed/distributed.py picks its device by local rank and
#     only does so when torch.cuda.device_count() > 1. Under mpirun that local rank
#     comes from OMPI_COMM_WORLD_LOCAL_RANK; see scripts/dcc/mpi_rank_env.sh.
#   - '--mem' is explicit; DCC defaults to 2 GB/core, which is not enough here.
#
# N=8 is 8 of the 15 youlab-gpu nodes and will queue behind other lab jobs. Reduce it
# if you only need a quick pass; nothing below depends on the node count.

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

MODEL_DIR=${MODEL_DIR:-$HYDRAGNN_ROOT/logs/qmugs-55245318-NN2-PM-FSDP0-V2-TP0}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-qmugs-55245318-NN2-PM-FSDP0-V2-TP0_epoch_65.pk}
DATASET_BASEDIR=${DATASET_BASEDIR:-/work/$USER/qmugs/QMugs001_new.pickle}
QMUGS_DATA_DIR=${QMUGS_DATA_DIR:-/work/$USER/qmugs}
LOG_NAME=${LOG_NAME:-QMugs_inference}

# --- Inference data configs ---
PERC_DOWNSAMPLE_TEST=${PERC_DOWNSAMPLE_TEST:-1.0}

# --- Downstream task configs ---
ARTIFACTS_DIR=${ARTIFACTS_DIR:-$HYDRAGNN_ROOT/logs/$LOG_NAME/artifacts}
RUN_DOWNSTREAM_CALCULATION=${RUN_DOWNSTREAM_CALCULATION:-0}
DOWNSTREAM_CALCULATIONS=${DOWNSTREAM_CALCULATIONS:-}
RUN_DOWNSTREAM_PREDICTION=${RUN_DOWNSTREAM_PREDICTION:-0}
DOWNSTREAM_PREDICTIONS=${DOWNSTREAM_PREDICTIONS:-}

# --- Figure configs ---
# RUN_FIGS controls only post-inference plotting on rank 0
RUN_FIGS=${RUN_FIGS:-0}

# Sampled figures rerun inference on a small selected subset
RUN_SAMPLE_FIGS=${RUN_SAMPLE_FIGS:-0}
RUN_ELECTRON_DENSITY_FIGS=${RUN_ELECTRON_DENSITY_FIGS:-0}
RUN_DIPOLE_FIGS=${RUN_DIPOLE_FIGS:-0}
FIGS_SAMPLE_MODE=${FIGS_SAMPLE_MODE:-random_molecules}
FIGS_NUM_RANDOM_MOLECULES=${FIGS_NUM_RANDOM_MOLECULES:-2}
FIGS_NUM_BEST_CONFORMERS=${FIGS_NUM_BEST_CONFORMERS:-2}
FIGS_NUM_SMALLEST_MOLECULES=${FIGS_NUM_SMALLEST_MOLECULES:-2}
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
# ADIOS2 is deliberately NOT installed on DCC; qmugs always uses pickle datasets and
# hydragnn guards the adios2 import. Report its absence instead of failing.
python -c "import adios2; print(adios2.__version__, adios2.__file__)" 2>/dev/null \
    || echo "adios2: NOT INSTALLED (expected on DCC; qmugs uses pickle datasets)"
python -c "import torch; print(torch.__version__, torch.__file__)"
python -c "import psi4; print(psi4.__version__)"
python -c "import hydragnn"

echo "===== GPUs ====="
nvidia-smi -L

echo "===== LD_LIBRARY_PATH ====="
echo "$LD_LIBRARY_PATH" | tr ':' '\n'

# --- MPI/runtime envs ---
# The MPICH_* variables are Cray MPICH settings and are inert under DCC's OpenMPI.
# They are kept so the two scripts stay line-for-line comparable. Note in particular
# that MPI on DCC is NOT CUDA-aware: the CUDA-aware OpenMPI module is broken (missing
# UCX libraries), so mpi4py transfers stage through host memory. HydraGNN moves
# gradients over torch.distributed/NCCL, so this affects throughput, not correctness.
export MPICH_ENV_DISPLAY=0
export MPICH_VERSION_DISPLAY=0
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONNOUSERSITE=1

# A5000s have no NVLink on these nodes. If NCCL stalls during rendezvous, uncomment:
# export NCCL_P2P_DISABLE=1

export OMP_NUM_THREADS=8
export HYDRAGNN_NUM_WORKERS=1
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1

export HYDRAGNN_TRACE_LEVEL=0
export HYDRAGNN_MAX_NUM_BATCH=1000
export TASK_PARALLEL=0
export HYDRAGNN_TASK_PARALLEL_PROPORTIONAL_SPLIT=0
export BATCH_SIZE=1

export HYDRAGNN_DDSTORE_METHOD=1
export HYDRAGNN_CUSTOM_DATALOADER=1

# Distributed rendezvous for torch/c10d
MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_IP=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
if [ -z "$MASTER_IP" ]; then
    MASTER_IP="$MASTER_HOST"
fi
export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT=${MASTER_PORT:-29501}
export HYDRAGNN_MASTER_ADDR=$MASTER_ADDR
export HYDRAGNN_MASTER_PORT=$MASTER_PORT

# FSDP knobs (for multi-dataset, set HYDRAGNN_USE_FSDP=0)
export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_FSDP_VERSION=2
export HYDRAGNN_FSDP_STRATEGY=FULL_SHARD

TASK_PARALLEL_ARG=""
if [ "$TASK_PARALLEL" = "1" ]; then
    TASK_PARALLEL_ARG="--task_parallel"
fi

# ----------------------------------------------------------------------------------------------------
# SET THESE
#
# MODEL_DIR and CHECKPOINT_PATH still name a Perlmutter run. Point them at a DCC
# training run before using this script; nothing else here depends on them.
# Perlmutter /pscratch/sd/w/wxrderek/{qmugs,artifacts} map to /work/$USER/{qmugs,artifacts}.

MODEL_DIR=$HYDRAGNN_ROOT/logs/qmugs-train-50461062-NN1-DCC-FSDP0-V2
CHECKPOINT_PATH=qmugs-train-50461062-NN1-DCC-FSDP0-V2_epoch_4.pk

DATASET_BASEDIR=/work/$USER/qmugs/QMugs0001_max800_delta_uptr-DCC.pickle
ARTIFACTS_DIR=/work/$USER/artifacts/artifacts_model-QMugs0001_max800_delta_uptr_data-QMugs0001_max800_delta_uptr-DCC
QMUGS_DATA_DIR=/work/$USER/qmugs

PERC_DOWNSAMPLE_TEST=0.5

RUN_DOWNSTREAM_CALCULATION=1
DOWNSTREAM_CALCULATIONS="electron_density dipole_moment"

RUN_DOWNSTREAM_PREDICTION=0
DOWNSTREAM_PREDICTIONS=""

RUN_FIGS=0
RUN_SAMPLE_FIGS=1
RUN_DIPOLE_FIGS=1
RUN_ELECTRON_DENSITY_FIGS=1

FIGS_SAMPLE_MODE="best_conformers"
FIGS_NUM_BEST_CONFORMERS=2
FIGS_NUM_SMALLEST_MOLECULES=2

FIGS_SHOW_3D_AXES=0
FIGS_HIDE_LEGEND=1

FIGS_MATRIX_POOL_SIZE=5
FIGS_METRICS_NATOMS_BIN_SIZE=4
FIGS_METRICS_MAX_NATOMS=100

# ----------------------------------------------------------------------------------------------------

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

DOWNSTREAM_PREDICTION_ARGS=()
if [ "$RUN_DOWNSTREAM_PREDICTION" = "1" ]; then
    DOWNSTREAM_PREDICTION_ARGS+=(--run_downstream_prediction)
    if [ -n "$DOWNSTREAM_PREDICTIONS" ]; then
        DOWNSTREAM_PREDICTION_ARGS+=(--downstream_predictions)
        for prediction in $DOWNSTREAM_PREDICTIONS; do
            DOWNSTREAM_PREDICTION_ARGS+=("$prediction")
        done
    fi
fi

FIGS_ARGS=()
if [ "$RUN_FIGS" = "1" ]; then
    FIGS_ARGS+=(--run_figs)
    # density matrix sampled visuals are enabled independently of metric plots
    if [ "$RUN_SAMPLE_FIGS" = "1" ]; then
        FIGS_ARGS+=(--run_sample_figs)
    fi
    if [ "$RUN_ELECTRON_DENSITY_FIGS" = "1" ]; then
        FIGS_ARGS+=(--run_electron_density_figs)
    fi
    if [ "$RUN_DIPOLE_FIGS" = "1" ]; then
        FIGS_ARGS+=(--run_dipole_figs)
    fi

    if [ "$FIGS_SAMPLE_MODE" = "best_conformers" ]; then
        FIGS_ARGS+=(--figs_best_conformers)
    elif [ "$FIGS_SAMPLE_MODE" = "smallest_molecules" ]; then
        FIGS_ARGS+=(--figs_smallest_molecules)
    else
        # random molecule sampling includes all test conformers for selected molecules
        FIGS_ARGS+=(--figs_random_molecules)
    fi

    # point-cloud settings affect only electron density figures
    FIGS_ARGS+=(--figs_num_random_molecules="$FIGS_NUM_RANDOM_MOLECULES")
    FIGS_ARGS+=(--figs_num_best_conformers="$FIGS_NUM_BEST_CONFORMERS")
    FIGS_ARGS+=(--figs_num_smallest_molecules="$FIGS_NUM_SMALLEST_MOLECULES")
    FIGS_ARGS+=(--figs_seed="$FIGS_SEED")
    FIGS_ARGS+=(--figs_density_point_filter="$FIGS_DENSITY_POINT_FILTER")
    FIGS_ARGS+=(--figs_density_top_percentile="$FIGS_DENSITY_TOP_PERCENTILE")
    FIGS_ARGS+=(--figs_density_absolute_cutoff="$FIGS_DENSITY_ABSOLUTE_CUTOFF")
    FIGS_ARGS+=(--figs_density_max_points="$FIGS_DENSITY_MAX_POINTS")
    FIGS_ARGS+=(--figs_density_isosurface_percentile="$FIGS_DENSITY_ISOSURFACE_PERCENTILE")
    if [ -n "$FIGS_DENSITY_ISOSURFACE_ABSOLUTE_LEVEL" ]; then
        FIGS_ARGS+=(--figs_density_isosurface_absolute_level="$FIGS_DENSITY_ISOSURFACE_ABSOLUTE_LEVEL")
    fi
    if [ "$FIGS_SHOW_3D_AXES" = "1" ]; then
        FIGS_ARGS+=(--figs_show_3d_axes)
    fi
    if [ "$FIGS_HIDE_LEGEND" = "1" ]; then
        FIGS_ARGS+=(--figs_hide_legend)
    fi
    FIGS_ARGS+=(--figs_matrix_pool_size="$FIGS_MATRIX_POOL_SIZE")
    if [ -n "$FIGS_METRICS_MAX_NATOMS" ]; then
        FIGS_ARGS+=(--figs_metrics_max_natoms="$FIGS_METRICS_MAX_NATOMS")
    fi
    if [ -n "$FIGS_METRICS_MAX_DENSITY_MATRIX_DIM" ]; then
        FIGS_ARGS+=(--figs_metrics_max_density_matrix_dim="$FIGS_METRICS_MAX_DENSITY_MATRIX_DIM")
    fi
    if [ -n "$FIGS_METRICS_NATOMS_BIN_SIZE" ]; then
        FIGS_ARGS+=(--figs_metrics_natoms_bin_size="$FIGS_METRICS_NATOMS_BIN_SIZE")
    fi
    if [ -n "$FIGS_METRICS_DENSITY_MATRIX_DIM_BIN_SIZE" ]; then
        FIGS_ARGS+=(--figs_metrics_density_matrix_dim_bin_size="$FIGS_METRICS_DENSITY_MATRIX_DIM_BIN_SIZE")
    fi
fi

# Launched with mpirun, NOT srun: OpenMPI/4.1.6 on DCC cannot be direct-launched by
# srun (no Slurm PMI support). "--bind-to none" is the analogue of Perlmutter's
# unbound layout -- the Slurm cgroup already confines the job to its 48 cores, and
# OMP_NUM_THREADS controls threading. See scripts/dcc/mpi_rank_env.sh for why the
# per-rank shim is needed.
cmd mpirun -np $((SLURM_JOB_NUM_NODES*4)) --map-by ppr:4:node --bind-to none \
    "$EXAMPLE_DIR/scripts/dcc/mpi_rank_env.sh" \
    python -u "$EXAMPLE_DIR/qmugs_densmat_inference.py" \
    --model_dir="$MODEL_DIR" \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --basedir="$DATASET_BASEDIR" \
    --qmugs_data_dir="$QMUGS_DATA_DIR" \
    --artifacts_dir="$ARTIFACTS_DIR" \
    --log=qmugs-inference-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-DCC \
    --batch_size=$BATCH_SIZE \
    --perc_downsample_test="$PERC_DOWNSAMPLE_TEST" \
    "${DOWNSTREAM_CALCULATION_ARGS[@]}" \
    "${DOWNSTREAM_PREDICTION_ARGS[@]}" \
    "${FIGS_ARGS[@]}" \
    --electron_density_method="cubeprop" \
    --manual_density_num_grid_points=100
