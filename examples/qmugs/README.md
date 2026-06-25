# QMugs

## Dev Reminders for Perlmutter
Before doing anything, make sure the correct modules are loaded on the machine. From the HydraGNN root directory, run
```
source installation_DOE_supercomputers/module_loads_perlmutter.sh
load_perlmutter_modules "12.9"
module -t list | sort
```

Set up the `conda` environment by running
```
bash installation_DOE_supercomputers/hydragnn_installation_bash_script_perlmutter.sh
conda activate .../hydragnn_venv/
python -m pip install -e . --no-deps
```
Note the `--no-deps` flag avoids reinstalling possibly incompatible versions of torch family modules after compatible ones have been installed by the installation script. 

The `psi4` package only supports downloading via `conda`. With the `conda` environment activated and all other HydraGNN dependencies installed, run:
```
conda install psi4 python=3.11 -c conda-forge
```

Navigate to the `qmugs/` directory to run the code:
```
cd examples/qmugs
```