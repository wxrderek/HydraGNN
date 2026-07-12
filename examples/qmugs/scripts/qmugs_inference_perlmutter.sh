#!/bin/bash
#SBATCH -A m5216
#SBATCH -J QMugs-Inference
#SBATCH -o inference-job-%j.out
#SBATCH -e inference-job-%j.out
#SBATCH -t 00:30:00
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH -c 32

function cmd() {
    echo "$@"
    time "$@"
}

# --- Paths (override with environment variables if needed) ---
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/global/homes/w/wxrderek/HydraGNN}
VENV_PATH=${VENV_PATH:-/global/homes/w/wxrderek/HydraGNN/HydraGNN-Installation-Perlmutter/hydragnn_venv}
EXAMPLE_DIR=$HYDRAGNN_ROOT/examples/qmugs
MODEL_DIR=${MODEL_DIR:-$HYDRAGNN_ROOT/logs/qmugs-55245318-NN2-PM-FSDP0-V2-TP0}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-qmugs-55245318-NN2-PM-FSDP0-V2-TP0_epoch_65.pk}
DATASET_BASEDIR=${DATASET_BASEDIR:-/pscratch/sd/w/wxrderek/qmugs/QMugs001_new.pickle}
QMUGS_DATA_DIR=${QMUGS_DATA_DIR:-/pscratch/sd/w/wxrderek/qmugs}
LOG_NAME=${LOG_NAME:-QMugs_inference}

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

# --- Perlmutter module + conda setup ---
# module reset
ml nersc-default/1.0 || true
ml conda/Miniforge3-24.11.3-0 || ml conda/Miniforge3-24.7.1-0

source "$HYDRAGNN_ROOT/installation_DOE_supercomputers/module_loads_perlmutter.sh"
load_perlmutter_modules 12.9

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
    echo "Set VENV_PATH to your Perlmutter HydraGNN conda env path."
    exit 1
fi

echo "Virtual environment path: $VENV_PATH"
conda activate "$VENV_PATH"

cd "$HYDRAGNN_ROOT" || exit 1
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== Module List ====="
module list

echo "===== Check ====="
which python
python -c "import adios2; print(adios2.__version__, adios2.__file__)"
python -c "import torch; print(torch.__version__, torch.__file__)"
python -c "import psi4; print(psi4.__version__)"
python -c "import hydragnn"

echo "===== LD_LIBRARY_PATH ====="
echo "$LD_LIBRARY_PATH" | tr ':' '\n'

# --- MPI/runtime envs ---
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

MODEL_DIR=$HYDRAGNN_ROOT/logs/qmugs-55245318-NN2-PM-FSDP0-V2-TP0
CHECKPOINT_PATH=qmugs-55245318-NN2-PM-FSDP0-V2-TP0_epoch_65.pk

DATASET_BASEDIR=/pscratch/sd/w/wxrderek/qmugs/QMugs001_new.pickle
ARTIFACTS_DIR=/pscratch/sd/w/wxrderek/artifacts_model-QMugs001_data-QMugs001
QMUGS_DATA_DIR=/pscratch/sd/w/wxrderek/qmugs

RUN_DOWNSTREAM_CALCULATION=1
DOWNSTREAM_CALCULATIONS="electron_density dipole_moment"

RUN_DOWNSTREAM_PREDICTION=0
DOWNSTREAM_PREDICTIONS=""

RUN_FIGS=1
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

cmd srun -N$SLURM_JOB_NUM_NODES -n$((SLURM_JOB_NUM_NODES*4)) -c32 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none -l --kill-on-bad-exit=1 \
    --export=ALL \
    python -u "$EXAMPLE_DIR/qmugs_densmat_inference.py" \
    --model_dir="$MODEL_DIR" \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --basedir="$DATASET_BASEDIR" \
    --qmugs_data_dir="$QMUGS_DATA_DIR" \
    --artifacts_dir="$ARTIFACTS_DIR" \
    --log=qmugs-inference-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES \
    --batch_size=$BATCH_SIZE \
    "${DOWNSTREAM_CALCULATION_ARGS[@]}" \
    "${DOWNSTREAM_PREDICTION_ARGS[@]}" \
    "${FIGS_ARGS[@]}" \
    --electron_density_method="cubeprop" \
    --manual_density_num_grid_points=100
