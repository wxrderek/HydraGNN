#!/bin/bash
#SBATCH -A youlab
#SBATCH -J QMugs-CompositeLossTest
#SBATCH -o slurm_output/schur_train/composite-loss-test-job-%j.out
#SBATCH -e slurm_output/schur_train/composite-loss-test-job-%j.out
#SBATCH -t 00:20:00
#SBATCH -p youlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem=16G

# Runs composite_loss_test.py, the CPU float64 checks of the composite loss terms.
# Submit from examples/qmugs: sbatch scripts/dcc/qmugs_composite_loss_test_dcc.sh

HYDRAGNN_ROOT=${HYDRAGNN_ROOT:-/work/$USER/HydraGNN}
VENV_PATH=${VENV_PATH:-/hpc/group/youlab/$USER/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv}

source "$HYDRAGNN_ROOT/installation_other/module_loads_dcc.sh"
load_dcc_modules 12.8
unset CONDA_SHLVL CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VENV_PATH"
if [[ "$(command -v python)" != "$VENV_PATH"/* ]]; then
    echo "ERROR: conda activate did not take effect"
    exit 1
fi

export PYTHONPATH=$HYDRAGNN_ROOT:$PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4

cd "$HYDRAGNN_ROOT/examples/qmugs" || exit 1
python -u composite_loss_test.py
