#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-SchurTrain
#SBATCH -o schur-train-job-%j.out
#SBATCH -e schur-train-job-%j.out
#SBATCH -t 48:00:00
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

# Training with the sequential Schur decoder. This is the Schur counterpart of
# scripts/dcc/qmugs_densmat_dcc.sh, kept separate so the two model families never share
# a config, a dataset, a checkpoint directory or a set of shell variables.
#
# What differs from qmugs_densmat_dcc.sh, and why:
#
#   - --inputfile points at qmugs_schur.json, whose SchurDecoder block turns the decoder
#     on and shrinks the Base output head to a stub. The decoder builds the matrix by the
#     bordering recursion instead of emitting it from a dense head.
#
#   - HYDRAGNN_USE_FSDP is 0, against 1 in the dense script. The decoder is attached by
#     composition around the backbone (hydragnn/decoders/schur/model.py), and the unused
#     Base heads are frozen so that DDP with find_unused_parameters=False accepts them.
#     Sharding that wrapper under FSDP is untested. Turn it on only after a DDP run works.
#
#   - '-N 1' rather than '-N 2'. Start on one node; the recursion is latency bound rather
#     than bandwidth bound, so scaling out buys less here than it does for the dense head.
#
#   - Batch size 8. The recursion grows the matrix out of place, so the retained autograd
#     graph is about K^3/3 floats per molecule: 288 MB at K = 600, 683 MB at K = 800 in
#     fp32. Batch size is the lever if it runs out of memory. See the "What to expect"
#     part of the Schur Decoder section in examples/qmugs/README.md.
#
#   - Script-local shell variables are prefixed SCHUR_ so a shell holding both scripts'
#     variables cannot silently mix them. HYDRAGNN_* keeps its names, being read by Python.
#
# Header differences from Perlmutter are as in qmugs_densmat_dcc.sh: no constraints or
# debug QOS, '-c 12' x 4 tasks fills the 48-core node, '--gres=gpu:4' with no GPU binding
# so every rank sees all four devices, and '--mem' explicit against DCC's 2 GB/core.

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
python -c "from hydragnn.decoders import SchurDecoderModel; print('schur decoder importable')"

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

# The decoder is a wrapper composed around the backbone and has only been run under DDP.
# Its unused Base heads are frozen so DDP with find_unused_parameters=False accepts them;
# FSDP sharding of the wrapper is untested. Leave this at 0 until a DDP run is healthy.
export HYDRAGNN_USE_FSDP=0
export HYDRAGNN_FSDP_VERSION=2
export HYDRAGNN_FSDP_STRATEGY=FULL_SHARD

# NaN guard, read at hydragnn/train/train_validate_test.py:689-690. Both default to off, so
# leaving them alone reproduces the unguarded runs exactly. Set them on the sbatch command
# line, e.g.
#
#   HYDRAGNN_SKIP_NONFINITE_STEP=1 sbatch scripts/dcc/qmugs_schur_dcc.sh
#
# HYDRAGNN_SKIP_NONFINITE_STEP=1 takes the gradient norm before every optimizer step, and if
# it is not finite it logs the epoch, batch, rank, grad_norm, loss and the batch's ChEMBL ids,
# then skips the step instead of writing NaN into the weights. All ranks agree on the verdict
# through an MPI allreduce, so no rank steps while another skips.
#
# HYDRAGNN_CLIP_GRAD_NORM=<x> additionally rescales the gradient to norm x. Leave it at 0
# while measuring: clipping changes what the run learns, and the first question is only
# whether grad_norm reads inf (a magnitude overflowed) or nan (a 0/0 in some backward).
#
# HYDRAGNN_GRAD_NORM_STRIDE=<n> logs the gradient norm every n batches even when it is finite.
# A norm that creeps up over epochs means a slow blow-up; one that is flat and then jumps in a
# single step means one bad batch. The loss shows neither.
export HYDRAGNN_SKIP_NONFINITE_STEP=${HYDRAGNN_SKIP_NONFINITE_STEP:-0}
export HYDRAGNN_CLIP_GRAD_NORM=${HYDRAGNN_CLIP_GRAD_NORM:-0}
export HYDRAGNN_GRAD_NORM_STRIDE=${HYDRAGNN_GRAD_NORM_STRIDE:-0}

# ----------------------------------------------------------------------------------------------------
# SET THESE
#
# SCHUR_MODELNAME must match the dataset written by scripts/dcc/qmugs_schur_pre_dcc.sh,
# which is the SAME dataset scripts/dcc/qmugs_densmat_dcc.sh trains the dense head on. That
# is deliberate: the two routes are meant to differ only in the decoder, so they share the
# conformers, the splits, the graphs, the delta target and the GNN backbone. If you point
# one route at a different dataset the comparison stops being controlled.
#
# SCHUR_MAX_DENSITY_MATRIX_SIZE caps which conformers are loaded. Keep it at or below
# "max_basis_dim" in qmugs_schur.json: conformers above max_basis_dim are skipped inside
# the decoder, and it is clearer to filter them at load time than to pay for loading them.
#
# The three SCHUR_ROUTING / SCHUR_SUPERVISION / SCHUR_TARGET variables override the
# corresponding keys in qmugs_schur.json, so a sweep needs no config edits and the saved
# config still records what actually ran. Leave one empty to take the config's value.
#
#   routing      teacher_forcing | free_running
#   supervision  joint | semi_sequential | per_step
#   target       total | delta      ("delta" needs a dataset preprocessed with a
#                                    density_matrix_delta_* output name)

SCHUR_MODELNAME=${SCHUR_MODELNAME:-QMugs001_max800_delta_uptr_decodercmp}
SCHUR_MAX_DENSITY_MATRIX_SIZE=${SCHUR_MAX_DENSITY_MATRIX_SIZE:-800}
SCHUR_UPDATE_MAX_PADDED_DIMENSION=${SCHUR_UPDATE_MAX_PADDED_DIMENSION:-1}
SCHUR_PERC_LOAD=${SCHUR_PERC_LOAD:-0.01}
SCHUR_PERC_TRAIN=${SCHUR_PERC_TRAIN:-0.8}
SCHUR_BATCH_SIZE=${SCHUR_BATCH_SIZE:-128}
SCHUR_NUM_EPOCH=${SCHUR_NUM_EPOCH:-100}
SCHUR_PRECISION=${SCHUR_PRECISION:-fp32}

SCHUR_ROUTING=${SCHUR_ROUTING:-teacher_forcing}
SCHUR_SUPERVISION=${SCHUR_SUPERVISION:-joint}
SCHUR_TARGET=${SCHUR_TARGET:-total}

SCHUR_SIZE_AWARE_ARGS=()
if [ -n "${SCHUR_MAX_DENSITY_MATRIX_SIZE:-}" ]; then
    SCHUR_SIZE_AWARE_ARGS+=(--max_density_matrix_size="$SCHUR_MAX_DENSITY_MATRIX_SIZE")
fi
if [ "$SCHUR_UPDATE_MAX_PADDED_DIMENSION" = "1" ]; then
    SCHUR_SIZE_AWARE_ARGS+=(--update_max_padded_dimension)
fi

SCHUR_DECODER_ARGS=()
if [ -n "${SCHUR_ROUTING:-}" ]; then
    SCHUR_DECODER_ARGS+=(--schur_routing="$SCHUR_ROUTING")
fi
if [ -n "${SCHUR_SUPERVISION:-}" ]; then
    SCHUR_DECODER_ARGS+=(--schur_supervision="$SCHUR_SUPERVISION")
fi
if [ -n "${SCHUR_TARGET:-}" ]; then
    SCHUR_DECODER_ARGS+=(--schur_target="$SCHUR_TARGET")
fi

# The run name records the decoder settings, so a sweep does not overwrite its own logs
# or checkpoints. Checkpoints from a dense-head run are NOT loadable here: attaching the
# decoder prefixes every state dict key with "backbone.". Keep "continue" at 0 in the
# config and give each sweep point its own name.
SCHUR_RUN_NAME=qmugs-schur-train-$SLURM_JOB_ID-NN$SLURM_JOB_NUM_NODES-DCC-$SCHUR_ROUTING-$SCHUR_SUPERVISION-$SCHUR_TARGET

# Launched with mpirun, NOT srun: OpenMPI/4.1.6 on DCC cannot be direct-launched by
# srun (no Slurm PMI support). "--bind-to none" is the analogue of Perlmutter's
# unbound layout -- the Slurm cgroup already confines the job to its 48 cores, and
# OMP_NUM_THREADS controls threading. See scripts/dcc/mpi_rank_env.sh for why the
# per-rank shim is needed.
cmd mpirun -np $((SLURM_JOB_NUM_NODES*4)) --map-by ppr:4:node --bind-to none \
    "$EXAMPLE_DIR/scripts/dcc/mpi_rank_env.sh" \
    python -u "$EXAMPLE_DIR/qmugs_densmat_train.py" \
    --log="$SCHUR_RUN_NAME" --everyone \
    --inputfile="${SCHUR_INPUTFILE:-$EXAMPLE_DIR/qmugs_schur.json}" \
    --batch_size=$SCHUR_BATCH_SIZE --num_epoch=$SCHUR_NUM_EPOCH \
    --precision=$SCHUR_PRECISION \
    --pickle \
    --perc_load=$SCHUR_PERC_LOAD \
    --perc_train=$SCHUR_PERC_TRAIN \
    "${SCHUR_SIZE_AWARE_ARGS[@]}" \
    "${SCHUR_DECODER_ARGS[@]}" \
    --modelname="$SCHUR_MODELNAME"
