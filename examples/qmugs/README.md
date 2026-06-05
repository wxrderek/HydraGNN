# QMugs

## Dev Reminders
Before doing anything, make sure the following modules are loaded on the machine:
```
module load conda
module load mpich/4.3.0 # replace with compatible version mpi version
```

The `psi4` package only supports downloading via `conda`. With a `conda` environment activated and all other HydraGNN dependencies installed, run:
```
conda install psi4 python=3.12 -c conda-forge
```

Navigate to the `qmugs/` directory to run the code:
```
cd examples/qmugs
```