#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-Ckpt
#SBATCH -o inspect-ckpt-job-%j.out
#SBATCH -e inspect-ckpt-job-%j.out
#SBATCH -t 01:00:00
#SBATCH -p youlab-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=48G

# Runs examples/qmugs/inspect_checkpoints.py, the read-only report of where a NaN sits in a
# saved checkpoint and how large the weights had grown before it appeared.
#
# No GPU is used, and no MPI: the script is single process and imports only torch. It lands
# on youlab-gpu only because the youlab CPU nodes are saturated, so no --gres is requested
# and the GPUs on the node stay free.
#
# --mem=48G is for the dense checkpoint, which is 6.3 GB on disk because its output head is
# 1600 x 320400 and AdamW keeps two moments of it. inspect_checkpoints.py memory maps the file
# and scans in 16M-element chunks, so the resident set stays far below that, but a 2 GB tensor
# is paged in whole and the margin is cheap.
#
# CKPT_PATHS defaults to the three checkpoints the NaN investigation needs:
#   1. the last healthy Schur checkpoint of 53692784, written at the end of epoch 24, roughly
#      fifteen optimizer steps before that run's weights went non-finite;
#   2. a checkpoint of 53692804, the sibling Schur run at batch 128 that was still healthy at
#      epoch 48, which is the control for "were 53692784's weights already anomalous";
#   3. the fully poisoned dense checkpoint of 53692783, which is the one that says WHICH
#      parameters hold the NaN.

function cmd() {
    echo "$@"
    time "$@"
}

# --- Paths (override with environment variables if needed) ---
HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/work/$USER/HydraGNN}
echo "HydraGNN root: $HYDRAGNN_ROOT"
# The environment lives OUTSIDE the repository; see qmugs_schur_decomp_test_dcc.sh for why.
VENV_PATH=${VENV_PATH:-/hpc/group/youlab/$USER/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv}
EXAMPLE_DIR="$HYDRAGNN_ROOT/examples/qmugs"
LOG_DIR=${LOG_DIR:-$HYDRAGNN_ROOT/logs}

SCHUR_DIED=qmugs-schur-train-53692784-NN1-DCC-teacher_forcing-joint-total
SCHUR_ALIVE=qmugs-schur-train-53692804-NN1-DCC-teacher_forcing-joint-total
DENSE_DEAD=qmugs-train-53692783-NN2-DCC-FSDP0-V2

# 53692804 keeps only its newest checkpoint and deletes the rest, so the second path follows
# the symlink rather than naming an epoch. If that run is still going, snapshot it first --
#   cp -L $LOG_DIR/$SCHUR_ALIVE/$SCHUR_ALIVE.pk /tmp/schur_alive.pk
# -- and pass the copy, or the file can be deleted out from under the read.
CKPT_PATHS=${CKPT_PATHS:-"\
$LOG_DIR/$SCHUR_DIED/${SCHUR_DIED}_epoch_24.pk \
$LOG_DIR/$SCHUR_ALIVE/$SCHUR_ALIVE.pk \
$LOG_DIR/$DENSE_DEAD/${DENSE_DEAD}_epoch_37.pk"}

CKPT_TOP=${CKPT_TOP:-25}
CKPT_NUM_THREADS=${CKPT_NUM_THREADS:-8}

# --- DCC module + conda setup ---
module purge 2>/dev/null || true
module load Anaconda3/2024.02

# Clear any conda state inherited from the calling shell; 'module purge' above tore down
# the conda that set it, and 'conda activate' would otherwise treat the activation as a
# no-op reactivation and leave the module python first on PATH.
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

echo "===== Checkpoints ====="
for path in $CKPT_PATHS; do
    if [ -f "$path" ]; then
        ls -l "$path"
    else
        echo "MISSING: $path"
    fi
done

# --- Runtime envs ---
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$CKPT_NUM_THREADS
export MKL_NUM_THREADS=$CKPT_NUM_THREADS

cmd python -u "$EXAMPLE_DIR/inspect_checkpoints.py" \
    $CKPT_PATHS \
    --top="$CKPT_TOP"
