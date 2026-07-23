#!/usr/bin/env bash
set -e

function lbk() {
    echo ""
    printf '=%.0s' {1..100}
    echo ""
}

lbk
echo "Exporting information for environment transfer across systems"
echo "NOTE: the resultant directory is automatically gitignored if it lands in references/"
lbk

# --- Configure current system ---
CURRENT_SYSTEM="dcc"

# SET THESE according to directories on respective systems
if [[ "$CURRENT_SYSTEM" == "perlmutter" ]]; then
    HYDRAGNN_ROOT="/pscratch/sd/w/wxrderek/HydraGNN"
    VENV_PATH="$HYDRAGNN_ROOT/HydraGNN-Installation-Perlmutter/hydragnn_venv"
elif [[ "$CURRENT_SYSTEM" == "dcc" ]]; then
    HYDRAGNN_ROOT="/work/${USER}/HydraGNN"
    VENV_PATH="/hpc/group/youlab/${USER}/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv"
fi

echo "HydraGNN root: $HYDRAGNN_ROOT"
echo "HydraGNN venv path: $VENV_PATH"

# --- Directories to output information ---
OUT_DIR="$HYDRAGNN_ROOT/references/${CURRENT_SYSTEM}_info"
MODULE_DIR="$OUT_DIR/modules"
VENV_DIR="$OUT_DIR/hydragnn_venv"
mkdir -p "$OUT_DIR" "$MODULE_DIR" "$VENV_DIR"

lbk

# --- Module loading and environment activation ---

# Perlmutter
if [[ "$CURRENT_SYSTEM" == "perlmutter" ]]; then
    echo "Working for Perlmutter system"

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

    conda activate "$VENV_PATH"

    cd "$HYDRAGNN_ROOT" || exit 1
    export PYTHONPATH=$PWD:$PYTHONPATH

# Duke Compute Cluster (DCC)
elif [[ "$CURRENT_SYSTEM" == "dcc" ]]; then
    echo "Working for Duke Compute Cluster system"

    # No 'ml' shortcut and no conda module to preload: load_dcc_modules purges first
    # and loads Anaconda3 itself. DCC provides CUDA/12.4 and CUDA/12.8
    source "$HYDRAGNN_ROOT/installation_other/module_loads_dcc.sh"
    load_dcc_modules 12.8

    # 'module purge' inside load_dcc_modules tears down whatever conda configured the
    # calling shell, but CONDA_PREFIX/CONDA_SHLVL stay exported. 'conda activate' would
    # then see CONDA_PREFIX already equal to the target, treat it as a no-op
    # reactivation, and leave the module Anaconda3 python first on PATH -- every export
    # below would describe the WRONG environment while the script reports success.
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

# ... no support for other systems
else
    echo "Environment transfer files only supports Perlmutter and DCC"
    exit 1
fi

# --- Get module information ---
lbk
echo "Getting module information..."
module list &> "$MODULE_DIR/module-list-postload.txt" || true
module avail &> "$MODULE_DIR/module-avail.txt" || true

if [[ "$CURRENT_SYSTEM" == "perlmutter" ]]; then
    module spider &> "$MODULE_DIR/module-spider.txt" || true

elif [[ "$CURRENT_SYSTEM" == "dcc" ]]; then
    # DCC runs Tcl Environment Modules 5.3.0, not Lmod: there is no 'module spider'.
    # Record the module system itself and the full definition of every module we load,
    # which is what 'module spider' was providing on Perlmutter.
    module --version &> "$MODULE_DIR/module-system.txt" || true
    echo "MODULEPATH=$MODULEPATH" >> "$MODULE_DIR/module-system.txt"

    : > "$MODULE_DIR/module-show-hydragnn-relevant.txt"
    for MOD in "CUDA/12.8" "$DCC_MPI_MODULE_LOADED" "$DCC_CMAKE_MODULE" "$DCC_CONDA_MODULE"; do
        echo "===== module show $MOD =====" >> "$MODULE_DIR/module-show-hydragnn-relevant.txt"
        module show "$MOD" &>> "$MODULE_DIR/module-show-hydragnn-relevant.txt" || true
        echo "" >> "$MODULE_DIR/module-show-hydragnn-relevant.txt"
    done

else
    echo "Environment transfer files only supports Perlmutter and DCC"
    exit 1
fi

echo "Got all module information"

# --- Get environment information ---
lbk
echo "Getting environment information..."
conda env export --from-history > "$VENV_DIR/environment.yml"
conda list > "$VENV_DIR/conda-list.txt"
conda list --json > "$VENV_DIR/conda-list.json"
conda list --explicit > "$VENV_DIR/conda-list-explicit.txt"
conda info > "$VENV_DIR/conda-info.txt"
conda info --json > "$VENV_DIR/conda-info.json"
conda config --show-sources > "$VENV_DIR/conda-config.txt"
python -m pip freeze > "$VENV_DIR/pip-freeze.txt"
python -m pip list > "$VENV_DIR/pip-list.txt"
python -m pip list --format=json > "$VENV_DIR/pip-list.json"
python --version > "$VENV_DIR/python-version.txt"
which python >> "$VENV_DIR/python-version.txt"

# 'pip freeze' also reports packages found in ~/.local/lib/pythonX.Y/site-packages,
# which the SLURM scripts hide via PYTHONNOUSERSITE=1. Recording sys.path makes it
# possible to tell which entries above actually live inside the environment.
python -c "import sys; print('\n'.join(sys.path))" > "$VENV_DIR/python-sys-path.txt"
echo "Got all environment information"

lbk
echo "DONE"
lbk