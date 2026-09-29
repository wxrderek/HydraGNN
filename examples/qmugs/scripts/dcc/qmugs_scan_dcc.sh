#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-Scan
#SBATCH -o scan-job-%j.out
#SBATCH -e scan-job-%j.out
#SBATCH -t 12:00:00
#SBATCH -p youlab-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=32G

# Runs examples/qmugs/scan_dataset_health.py, the read-only screen of the preprocessed
# pickle datasets for the inputs that can produce a NaN in PAINN: near-coincident atoms,
# atoms with no neighbour inside the cutoff, self loops, non-finite positions or targets,
# and outsized target entries.
#
# No GPU is used. It lands on youlab-gpu only because the youlab CPU nodes are saturated,
# so no --gres is requested and the GPUs on the node stay free for other jobs. '-c 8' is
# for the threaded BLAS behind torch.cdist, not for multiple ranks: the script is single
# process and imports no MPI.
#
# It does need torch_geometric, since the pickles hold torch_geometric Data objects and
# unpickling them imports the class. It does not import hydragnn.
#
# The cost is dominated by reading the pickles, roughly 2.5 MB per conformer, so the 0.1
# draw is about 180 GB of reads while the 0.01 draw is about 18 GB. Set SCAN_LIMIT to cap
# the conformers read per label if the full pass is too slow. Datasets are scanned in the
# order given and each prints its summary as it finishes, so listing the small draws first
# means a walltime overrun costs only the last one.

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

# --- Scan configs ---
# The three draws that matter for the NaN investigation, oldest first: the 636-conformer
# draw is the only one whose training runs never went NaN, the 0.01 draw is where both the
# dense head and the Schur decoder failed after ~1000 steps, and the 0.1 draw failed inside
# epoch 0. qmugs.py draws molecules with random.sample, so these are independent draws and
# a pathological conformer can be in one and not another.
SCAN_DATASETS=${SCAN_DATASETS:-"QMugs0001_max500_delta_uptr QMugs001_max800_delta_uptr_decodercmp QMugs01_max800_delta_uptr"}
SCAN_DATASET_DIR=${SCAN_DATASET_DIR:-/work/$USER/qmugs}
SCAN_LABELS=${SCAN_LABELS:-"trainset valset testset"}
# must match the radius the graphs were built with, or the isolated-atom count is meaningless
SCAN_RADIUS=${SCAN_RADIUS:-7.0}
SCAN_LIMIT=${SCAN_LIMIT:-0}
SCAN_N_REPORT=${SCAN_N_REPORT:-10}

SCAN_NUM_THREADS=${SCAN_NUM_THREADS:-8}

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
python -c "import torch_geometric; print(torch_geometric.__version__)"

# --- Runtime envs ---
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$SCAN_NUM_THREADS
export MKL_NUM_THREADS=$SCAN_NUM_THREADS

# ----------------------------------------------------------------------------------------------------
# SET THESE

SCAN_ARGS=()
SCAN_ARGS+=(--datasets $SCAN_DATASETS)
SCAN_ARGS+=(--dataset_dir="$SCAN_DATASET_DIR")
SCAN_ARGS+=(--labels $SCAN_LABELS)
SCAN_ARGS+=(--radius="$SCAN_RADIUS")
SCAN_ARGS+=(--limit="$SCAN_LIMIT")
SCAN_ARGS+=(--n_report="$SCAN_N_REPORT")

cmd python -u "$EXAMPLE_DIR/scan_dataset_health.py" \
    "${SCAN_ARGS[@]}"
