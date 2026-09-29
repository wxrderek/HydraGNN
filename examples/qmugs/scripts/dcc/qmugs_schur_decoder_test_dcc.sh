#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-SchurDec
#SBATCH -o schur-decoder-job-%j.out
#SBATCH -e schur-decoder-job-%j.out
#SBATCH -t 00:30:00
#SBATCH -p youlab-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=32G

# Runs examples/qmugs/schur_decoder_test.py, the verification suite for the sequential
# Schur decoder in hydragnn/decoders/schur/. DCC-only, like its sibling
# qmugs_schur_decomp_test_dcc.sh; there is no Perlmutter counterpart.
#
# Everything runs in float64 on CPU on small synthetic matrices, so the resource shape is
# much lighter than the diagnostic script: minutes, not hours, and no GPU. '-c 8' is for
# the threaded BLAS, not for multiple ranks.
#
# Unlike the diagnostic, this one does import hydragnn (it is testing it), so it needs
# torch_geometric as well as torch. It still imports no MPI and runs single process.
# Psi4 is only touched when --wfn_path is given.

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

# --- Test configs ---
DECODER_SEED=${DECODER_SEED:-0}
# Optional: a Psi4 wavefunction file, which adds the oracle round trip and the teacher
# forcing check on a real QMugs conformer instead of only on synthetic matrices. Leave
# empty to run the synthetic checks alone.
DECODER_WFN_PATH=${DECODER_WFN_PATH:-}

DECODER_NUM_THREADS=${DECODER_NUM_THREADS:-8}

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
python -c "import hydragnn; print(hydragnn.__file__)"

# --- Runtime envs ---
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=$DECODER_NUM_THREADS
export MKL_NUM_THREADS=$DECODER_NUM_THREADS

# ----------------------------------------------------------------------------------------------------
# SET THESE

DECODER_ARGS=()
DECODER_ARGS+=(--seed="$DECODER_SEED")
if [ -n "$DECODER_WFN_PATH" ]; then
    DECODER_ARGS+=(--wfn_path="$DECODER_WFN_PATH")
fi

# The script exits nonzero if any check fails, so SLURM reports the job as FAILED rather
# than leaving a silent failure in the log.
cmd python -u "$EXAMPLE_DIR/schur_decoder_test.py" \
    "${DECODER_ARGS[@]}"
