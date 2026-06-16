# QMugs

## Dev Reminders for Perlmutter
Before doing anything, make sure the correct modules are loaded on the machine. From the HydraGNN root directory, run
```
cd installation_DOE_supercomputers
source module_loads_perlmutter.sh
load_perlmutter_modules "12.4"
```

Then install `rdkit` with `conda install rdkit==2026.3.2`.

The `psi4` package only supports downloading via `conda`. With a `conda` environment activated and all other HydraGNN dependencies installed, run:
```
conda install psi4 python=3.11 -c conda-forge
```

Navigate to the `qmugs/` directory to run the code:
```
cd examples/qmugs
```