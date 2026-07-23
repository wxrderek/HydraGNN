#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-Train
#SBATCH -o train-job-%j.out
#SBATCH -e train-job-%j.out
#SBATCH -t 04:00:00
#SBATCH -p youlab-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH -c 12
#SBATCH --mem=256G

# aa25-19 has a faulty GPU 3 ("Unable to determine the device handle for gpu
# 0000:B1:00.0"), which makes torch report ZERO usable devices on that node.
# Slurm still shows it as healthy and idle, so jobs land there first. REMOVE this
# line once Duke OIT repairs the node.

#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19

# DCC counterpart of scripts/perlmutter/qmugs_densmat_perlmutter.sh.
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
export BATCH_SIZE=16
export NUM_EPOCH=5
export MAX_DENSITY_MATRIX_SIZE=800
export UPDATE_MAX_PADDED_DIMENSION=1

export HYDRAGNN_DDSTORE_METHOD=1
export HYDRAGNN_CUSTOM_DATALOADER=0

# Single-dataset default (same as Frontier script). For multi-dataset, use the full list.
MULTI_MODEL_LIST=$datadir0
# MULTI_MODEL_LIST=$datadir0,$datadir1,$datadir2,$datadir3,$datadir4,$datadir5,$datadir6,$datadir7,$datadir8,$datadir9,$datadir10,$datadir11,$datadir12,$datadir13,$datadir14,$datadir15

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

SIZE_AWARE_ARGS=()
if [ -n "${MAX_DENSITY_MATRIX_SIZE:-}" ]; then
    SIZE_AWARE_ARGS+=(--max_density_matrix_size="$MAX_DENSITY_MATRIX_SIZE")
fi
if [ "$UPDATE_MAX_PADDED_DIMENSION" = "1" ]; then
    SIZE_AWARE_ARGS+=(--update_max_padded_dimension)
fi

# Launched with mpirun, NOT srun: OpenMPI/4.1.6 on DCC cannot be direct-launched by
# srun (no Slurm PMI support). "--bind-to none" is the analogue of Perlmutter's
# unbound layout -- the Slurm cgroup already confines the job to its 48 cores, and
# OMP_NUM_THREADS controls threading. See scripts/dcc/mpi_rank_env.sh for why the
# per-rank shim is needed.
cmd mpirun -np $((SLURM_JOB_NUM_NODES*4)) --map-by ppr:4:node --bind-to none \
    "$EXAMPLE_DIR/scripts/dcc/mpi_rank_env.sh" \
    python -u "$EXAMPLE_DIR/qmugs_densmat_train.py" \
    --log=qmugs-train-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-DCC-FSDP$HYDRAGNN_USE_FSDP-V$HYDRAGNN_FSDP_VERSION --everyone \
    --inputfile="$EXAMPLE_DIR/qmugs_densmat.json" \
    --batch_size=$BATCH_SIZE --num_epoch=$NUM_EPOCH \
    --precision=fp32 \
    --pickle \
    --perc_load=0.01 \
    --perc_train=0.8 \
    "${SIZE_AWARE_ARGS[@]}" \
    --modelname="QMugs0001_max800_delta_uptr"
