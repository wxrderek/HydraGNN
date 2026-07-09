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
export BATCH_SIZE=32

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

cmd srun -N$SLURM_JOB_NUM_NODES -n4 -c32 --ntasks-per-node=4 --gpus-per-task=1 --gpu-bind=none -l --kill-on-bad-exit=1 \
    --export=ALL \
    python -u "$EXAMPLE_DIR/qmugs_densmat_inference.py" \