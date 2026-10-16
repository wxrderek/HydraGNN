#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-SchurDataLoad
#SBATCH -o slurm_output/dataload/schur-dataload-job-%j.out
#SBATCH -e slurm_output/dataload/schur-dataload-job-%j.out
#SBATCH -t 24:00:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -c 12
#SBATCH --mem=256G

# Preprocessing for the sequential Schur decoder. This is the Schur counterpart of
# scripts/dcc/qmugs_pre_dcc.sh, kept as a separate script so the legacy dense-head
# datasets are never overwritten by a Schur run or the reverse.
#
# What differs from qmugs_pre_dcc.sh, and why:
#   - --inputfile points at qmugs_schur.json, so the task variant, the padding width and
#     the target layout all come from the Schur config rather than the dense one.
#   - SCHUR_MODELNAME is distinct, so the two datasets sit side by side on disk.
#   - The script-local shell variables are prefixed SCHUR_ so a shell that has both
#     scripts' variables exported cannot silently mix them. The HYDRAGNN_* variables keep
#     their names, since those are read by the Python.
#
# The dataset this writes carries four extra per-orbital fields (orbital_to_atom,
# orbital_angular_momentum, orbital_within_shell, orbital_shell) plus the derived
# one-hot widths and species list in the pickle metadata. The decoder needs all of them,
# and datasets preprocessed before they existed will make training stop with an explicit
# error. See the Schur Decoder section of examples/qmugs/README.md.
#
# Header differences from Perlmutter are as in qmugs_pre_dcc.sh; preprocessing needs no
# GPU, '-c 12' x 4 tasks is the whole 48-core node, and '--mem' is explicit because DCC
# defaults to 2 GB/core.

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

echo "===== LD_LIBRARY_PATH ====="
echo "$LD_LIBRARY_PATH" | tr ':' '\n'

# --- MPI/runtime envs ---
# The MPICH_* variables are Cray MPICH settings and are inert under DCC's OpenMPI.
# They are kept so the two scripts stay line-for-line comparable.
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
export HYDRAGNN_CUSTOM_DATALOADER=0

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

# Nothing is trained here, so FSDP is irrelevant; it is set only because the log name
# below records it and the two preprocessing scripts stay comparable.
export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_FSDP_VERSION=2
export HYDRAGNN_FSDP_STRATEGY=SHARD_GRAD_OP

# ----------------------------------------------------------------------------------------------------
# SET THESE
#
# SCHUR_MAX_DENSITY_MATRIX_SIZE is the size cap in basis functions. It must be at least
# as large as "max_basis_dim" in qmugs_schur.json, or the decoder will skip conformers
# the loader bothered to prepare. With --update_max_padded_dimension it also becomes the
# padding width N_pad, so keeping it tight is what keeps the stored targets small:
# T(800) is 1.3 MB per conformer against T(2002) at 8 MB.
#
# SCHUR_MODELNAME must match the training script's, since it is the on-disk dataset name.
#
# This dataset is SHARED with the dense-head route. scripts/dcc/qmugs_pre_dcc.sh writes the
# same MODELNAME and the same artifacts directory, so run ONE of the two, not both. They are
# kept as separate entry points only because each route has its own config file, and they are
# interchangeable because preprocessing reads just four things from the config, on which
# qmugs_densmat.json and qmugs_schur.json agree: Architecture.radius,
# Architecture.max_neighbours, Variables_of_interest.output_names and
# Variables_of_interest.mask_output_loss. Change any of those in one config and the two
# routes stop sharing a dataset.
#
# Sharing works because _mol_to_graph attaches the per-orbital metadata to every conformer
# regardless of task, and the density_matrix_delta_uptr output name additionally stores the
# promolecular upper triangle. The dense head reads y (= Delta D); the Schur decoder reads y
# plus the promolecular triangle and rebuilds the total before its recursion.

SCHUR_MODELNAME=${SCHUR_MODELNAME:-QMugs001_max800_delta_uptr_decodercmp}
SCHUR_MAX_DENSITY_MATRIX_SIZE=${SCHUR_MAX_DENSITY_MATRIX_SIZE:-800}
SCHUR_UPDATE_MAX_PADDED_DIMENSION=${SCHUR_UPDATE_MAX_PADDED_DIMENSION:-1}
SCHUR_PERC_LOAD=${SCHUR_PERC_LOAD:-0.01}
SCHUR_PERC_TRAIN=${SCHUR_PERC_TRAIN:-0.8}
# only used to size the dataloaders during preprocessing; training reads its own
SCHUR_BATCH_SIZE=${SCHUR_BATCH_SIZE:-8}
SCHUR_NUM_EPOCH=${SCHUR_NUM_EPOCH:-5}

SCHUR_SIZE_AWARE_ARGS=()
if [ -n "${SCHUR_MAX_DENSITY_MATRIX_SIZE:-}" ]; then
    SCHUR_SIZE_AWARE_ARGS+=(--max_density_matrix_size="$SCHUR_MAX_DENSITY_MATRIX_SIZE")
fi
if [ "$SCHUR_UPDATE_MAX_PADDED_DIMENSION" = "1" ]; then
    SCHUR_SIZE_AWARE_ARGS+=(--update_max_padded_dimension)
fi

# --- More paths ---
# Perlmutter /pscratch/sd/w/wxrderek/artifacts maps to /work/$USER/artifacts on DCC.
# Datasets and artifacts may live on /work; only psi4's own installation cannot.
#
# qmugs_schur.json uses "density_matrix_delta_uptr", a delta target, so --artifacts_dir is
# mandatory here: it is where the promolecular baselines are written, both as the .npy
# artifacts that inference and figures read and as the upper triangle stored on each
# conformer, which is what the decoder needs to rebuild the total inside its recursion.
SCHUR_ARTIFACTS_DIR=${SCHUR_ARTIFACTS_DIR:-/work/$USER/artifacts/DCC_MODEL-${SCHUR_MODELNAME}_DATA-${SCHUR_MODELNAME}}

# Launched with mpirun, NOT srun: OpenMPI/4.1.6 on DCC cannot be direct-launched by
# srun (no Slurm PMI support). "--bind-to none" is the analogue of Perlmutter's
# unbound layout -- the Slurm cgroup already confines the job to its 48 cores, and
# OMP_NUM_THREADS controls threading. See scripts/dcc/mpi_rank_env.sh for why the
# per-rank shim is needed.
cmd mpirun -np $((SLURM_JOB_NUM_NODES*4)) --map-by ppr:4:node --bind-to none \
    "$EXAMPLE_DIR/scripts/dcc/mpi_rank_env.sh" \
    python -u "$EXAMPLE_DIR/qmugs_densmat_train.py" \
    --log=qmugs-schur-pre-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-DCC --everyone \
    --inputfile="$EXAMPLE_DIR/qmugs_schur.json" \
    --artifacts_dir="$SCHUR_ARTIFACTS_DIR" \
    --batch_size=$SCHUR_BATCH_SIZE --num_epoch=$SCHUR_NUM_EPOCH \
    --precision=fp32 \
    --pickle \
    --perc_load=$SCHUR_PERC_LOAD \
    --perc_train=$SCHUR_PERC_TRAIN \
    "${SCHUR_SIZE_AWARE_ARGS[@]}" \
    --preonly \
    --streaming_preonly \
    --modelname="$SCHUR_MODELNAME"
