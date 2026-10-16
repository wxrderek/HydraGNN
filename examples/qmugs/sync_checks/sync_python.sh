#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-SyncCheck
#SBATCH -o sync-check-job-%j.out
#SBATCH -e sync-check-job-%j.out
#SBATCH -t 04:00:00
#SBATCH -p youlab-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --exclude=dcc-youlab-gpu-ferc-s-aa25-19

# Generic launcher for the sync check Python helpers (eval_checkpoint.py, compare_pickles.py,
# ground_truth_check.py), with the module, conda and MPI setup of scripts/dcc/qmugs_densmat_dcc.sh
# copied verbatim so the helpers run in exactly the environment the pipeline runs in.
#
#   sbatch [sbatch overrides] examples/qmugs/sync_checks/sync_python.sh <helper.py> [helper args]
#
# The header requests one CPU task and no GPU, which is what compare_pickles.py and
# ground_truth_check.py need. eval_checkpoint.py is submitted with GPUs and one rank per GPU,
# e.g. '-N 1 --ntasks-per-node=4 --gres=gpu:4 -c 12 --mem=256G'. NRANKS sets the number of MPI
# ranks and defaults to every task of the allocation.

function cmd() {
    echo "$@"
    time "$@"
}

HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/work/$USER/HydraGNN}
echo "HydraGNN root: $HYDRAGNN_ROOT"
VENV_PATH=${VENV_PATH:-/hpc/group/youlab/$USER/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv}
EXAMPLE_DIR="$HYDRAGNN_ROOT/examples/qmugs"

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
conda activate "$VENV_PATH"
if [[ "$(command -v python)" != "$VENV_PATH"/* ]]; then
    echo "ERROR: conda activate did not take effect; python resolves to $(command -v python)"
    exit 1
fi

cd "$HYDRAGNN_ROOT" || exit 1
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== Check ====="
which python
python -c "import torch; print(torch.__version__, torch.__file__)"
nvidia-smi -L 2>/dev/null || echo "no GPUs in this allocation"
git -C "$HYDRAGNN_ROOT" log --oneline -1

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export HYDRAGNN_NUM_WORKERS=1
export HYDRAGNN_USE_VARIABLE_GRAPH_SIZE=1
export HYDRAGNN_AGGR_BACKEND=mpi
export HYDRAGNN_VALTEST=1
export HYDRAGNN_USE_FSDP=0

MASTER_HOST=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_IP=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1 {print $1}')
if [ -z "$MASTER_IP" ]; then
    MASTER_IP="$MASTER_HOST"
fi
export MASTER_ADDR="$MASTER_IP"
export MASTER_PORT=${MASTER_PORT:-29503}
export HYDRAGNN_MASTER_ADDR=$MASTER_ADDR
export HYDRAGNN_MASTER_PORT=$MASTER_PORT

NRANKS=${NRANKS:-$SLURM_NTASKS}
HELPER="$1"
shift
cmd mpirun -np "$NRANKS" --map-by ppr:$((NRANKS < 4 ? NRANKS : 4)):node --bind-to none \
    "$EXAMPLE_DIR/scripts/dcc/mpi_rank_env.sh" \
    python -u "$EXAMPLE_DIR/sync_checks/$HELPER" "$@"
