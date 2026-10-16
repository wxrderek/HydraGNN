#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-Schur
#SBATCH -o slurm_output/schur_decomp/schur-job-%j.out
#SBATCH -e slurm_output/schur_decomp/schur-job-%j.out
#SBATCH -t 04:00:00
#SBATCH -p youlab-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --mem=128G

# Runs examples/qmugs/schur_decomp_test.py, the LDL^T diagnostic on ground-truth QMugs
# density matrices. Unlike the other scripts in this directory there is no Perlmutter
# counterpart; this is DCC-only.
#
# Resource shape follows scripts/dcc/qmugs_figs_dcc.sh: one task, CPU only, no GPU.
# The work is dominated by dense O(K^3) LAPACK calls on matrices up to 2002 x 2002, so
# '-c 32' is there to give the threaded BLAS something to use rather than to run
# several ranks. '--mem' is explicit; DCC defaults to 2 GB/core.
#
# Like scripts/dcc/qmugs_eda_dcc.sh, this script does NOT source module_loads_dcc.sh.
# schur_decomp_test.py imports only numpy, pandas, torch, matplotlib and psi4 -- no
# hydragnn, no mpi4py, no CUDA -- so it needs conda and nothing else. The Anaconda3
# module is loaded directly.

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
LOG_DIR=${LOG_DIR:-$HYDRAGNN_ROOT/logs/qmugs-schur-diagnostics}

# --- Diagnostic configs ---
# Sampling is stratified over density matrix size K, so the total conformer count is
# NUM_SIZE_BINS x NUM_PER_BIN. The large-K tail is where the rank deficiency is most
# severe, which is why the sample is stratified rather than uniform.
SCHUR_NUM_SIZE_BINS=${SCHUR_NUM_SIZE_BINS:-10}
SCHUR_NUM_PER_BIN=${SCHUR_NUM_PER_BIN:-20}
SCHUR_SEED=${SCHUR_SEED:-0}
# Cholesky jitter as a fraction of the mean diagonal entry. Pivots below
# PIVOT_FLOOR_FACTOR x jitter count as numerically zero.
SCHUR_JITTER_RELATIVE=${SCHUR_JITTER_RELATIVE:-1e-5}
SCHUR_PIVOT_FLOOR_FACTOR=${SCHUR_PIVOT_FLOOR_FACTOR:-10.0}
# 1e-8 sits inside the float64 noise floor of eigh (lambda_max ~ 2.7, so the cutoff lands
# at ~2.7e-8) and admitted spurious eigenvalues for 140 of 200 conformers in the first run,
# which is what produced the meaningless 1e8 range condition numbers there.
SCHUR_RANK_TOLERANCE=${SCHUR_RANK_TOLERANCE:-1e-6}
SCHUR_NUM_ENTRY_SAMPLES=${SCHUR_NUM_ENTRY_SAMPLES:-20000}
SCHUR_NUM_PROMOLECULAR=${SCHUR_NUM_PROMOLECULAR:-10}
SCHUR_NUM_JITTER_SWEEP=${SCHUR_NUM_JITTER_SWEEP:-5}
SCHUR_EXPLICIT_CHECK_MAX_DIM=${SCHUR_EXPLICIT_CHECK_MAX_DIM:-400}

# M1 - M5. The Picard analysis costs O(k^3) per sampled step, so NUM_K_SAMPLES steps
# geometrically spaced cost roughly twice the single largest step. The M2 rollout needs
# exact truncated latents at every step, which is O(K^4), hence the separate size cap.
SCHUR_NUM_SPECTRAL=${SCHUR_NUM_SPECTRAL:-20}
SCHUR_NUM_K_SAMPLES=${SCHUR_NUM_K_SAMPLES:-24}
SCHUR_NUM_ROLLOUT=${SCHUR_NUM_ROLLOUT:-6}
SCHUR_ROLLOUT_MAX_DIM=${SCHUR_ROLLOUT_MAX_DIM:-260}
SCHUR_ROLLOUT_RESIDUAL_TARGET=${SCHUR_ROLLOUT_RESIDUAL_TARGET:-1e-3}
SCHUR_NUM_LOWDIN=${SCHUR_NUM_LOWDIN:-8}

SCHUR_NUM_THREADS=${SCHUR_NUM_THREADS:-32}

# --- DCC module + conda setup ---
module purge 2>/dev/null || true
module load Anaconda3/2024.02

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
python -c "import torch; print(torch.__version__, torch.__file__)"
python -c "import matplotlib; print(matplotlib.__version__, matplotlib.__file__)"
python -c "import psi4; print(psi4.__version__)"

echo "===== LD_LIBRARY_PATH ====="
echo "$LD_LIBRARY_PATH" | tr ':' '\n'

# --- Runtime envs ---
export PYTHONNOUSERSITE=1

# The dense factorizations are the whole cost of this job, so the BLAS is given the
# full core allocation. schur_decomp_test.py also calls torch.set_num_threads and
# psi4.set_num_threads with --num_threads, which should match this.
export OMP_NUM_THREADS=$SCHUR_NUM_THREADS
export MKL_NUM_THREADS=$SCHUR_NUM_THREADS

# ----------------------------------------------------------------------------------------------------
# SET THESE
#
# Defaults above give 10 x 20 = 200 conformers stratified over K. Reduce
# SCHUR_NUM_PER_BIN for a quick smoke run.

SCHUR_ARGS=()
SCHUR_ARGS+=(--log_dir="$LOG_DIR")
SCHUR_ARGS+=(--num_size_bins="$SCHUR_NUM_SIZE_BINS")
SCHUR_ARGS+=(--num_per_bin="$SCHUR_NUM_PER_BIN")
SCHUR_ARGS+=(--seed="$SCHUR_SEED")
SCHUR_ARGS+=(--jitter_relative="$SCHUR_JITTER_RELATIVE")
SCHUR_ARGS+=(--pivot_floor_factor="$SCHUR_PIVOT_FLOOR_FACTOR")
SCHUR_ARGS+=(--rank_tolerance="$SCHUR_RANK_TOLERANCE")
SCHUR_ARGS+=(--num_entry_samples="$SCHUR_NUM_ENTRY_SAMPLES")
SCHUR_ARGS+=(--num_promolecular="$SCHUR_NUM_PROMOLECULAR")
SCHUR_ARGS+=(--num_jitter_sweep="$SCHUR_NUM_JITTER_SWEEP")
SCHUR_ARGS+=(--explicit_check_max_dim="$SCHUR_EXPLICIT_CHECK_MAX_DIM")
SCHUR_ARGS+=(--num_spectral="$SCHUR_NUM_SPECTRAL")
SCHUR_ARGS+=(--num_k_samples="$SCHUR_NUM_K_SAMPLES")
SCHUR_ARGS+=(--num_rollout="$SCHUR_NUM_ROLLOUT")
SCHUR_ARGS+=(--rollout_max_dim="$SCHUR_ROLLOUT_MAX_DIM")
SCHUR_ARGS+=(--rollout_residual_target="$SCHUR_ROLLOUT_RESIDUAL_TARGET")
SCHUR_ARGS+=(--num_lowdin="$SCHUR_NUM_LOWDIN")
SCHUR_ARGS+=(--num_threads="$SCHUR_NUM_THREADS")

# Launched as plain python, NOT through mpirun and scripts/dcc/mpi_rank_env.sh. The
# other scripts here need both because they run MPI programs that read SLURM_PROCID;
# schur_decomp_test.py imports no MPI and runs in a single process, so the rank shim
# has nothing to remap.
cmd python -u "$EXAMPLE_DIR/schur_decomp_test.py" \
    "${SCHUR_ARGS[@]}"
