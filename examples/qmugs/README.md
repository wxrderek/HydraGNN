# QMugs

## Directory Layout
```
qmugs
|
├── figs                             # functions to plot things
|   ├── density_visualizers.py       # NOT IMPLEMENTED
|   └── more to come...
| 
├── scripts                          # SLURM scripts
|   ├── qmugs_densmat_perlmutter.sh  # runs training on Perlmutter
|   ├── qmugs_eda_perlmutter.sh      # runs EDA script on Perlmutter
|   ├── qmugs_pre_perlmutter.sh      # runs data preprocessing to prep for training on Perlmutter
|   └── test.sh                      # tests environment and package setup on Perlmutter
|
├── utils
|   ├── download_data.json           # data download config, UPDATE THIS to point to your own directories
|   ├── download_data.py             # Python wrapper for data download functions
|   ├── download_data.sh             # shell script to run the data download
|   └── utils.py                     # utilities, importantly including density matrix pre/post-processing
| 
├── eda.py                           # script with EDA functionalities for the QMugs dataset
├── figs.py                          # Python script to run figure plotting
├── qmugs_densmat_inference.py       # Python script to run inference on a trained model 
├── qmugs_densmat_train.py           # Python script to run training
├── qmugs.py                         # Python file containing the definition of QMugsDataset class
└── README.md

```

## Dev Reminders for Perlmutter
Before doing anything, make sure the correct modules are loaded on the machine. From the HydraGNN root directory, run
```
source installation_DOE_supercomputers/module_loads_perlmutter.sh
load_perlmutter_modules "12.9"
module -t list | sort
```

Activate the `conda` environment by running
```
conda activate HydraGNN-Installation-Perlmutter/hydragnn_venv/
```

The `psi4` package only supports downloading via `conda`. With the `conda` environment activated and all other HydraGNN dependencies installed, run:
```
conda install psi4 python=3.11 -c conda-forge
```

Navigate to the `qmugs/` directory to run the code:
```
cd examples/qmugs
```

### Environment Setup
Set up the `conda` environment by running
```
bash installation_DOE_supercomputers/hydragnn_installation_bash_script_perlmutter.sh
conda activate HydraGNN-Installation-Perlmutter/hydragnn_venv/
python -m pip install -e . --no-deps
```
Note the `--no-deps` flag avoids reinstalling possibly incompatible versions of torch family modules after compatible ones have been installed by the installation script. 

### Learning total density (alpha + beta)
For learning the total density matrix, set `config.json` as follows:
```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0],
    "output_dim": [4008004],
    "output_names": [
        "density_matrix"
    ],
    "type": ["graph"],
    ...
}
```

### Delta learning total density
For delta learning the total density with the promolecular density matrix as a baseline, set `config.json` as follows:

```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0],
    "output_dim": [4008004],
    "output_names": [
        "density_matrix_delta"
    ],
    "type": ["graph"],
    ...
}
```

The preprocessing path requires `--artifacts_dir` and saves unpadded promolecular density matrices as float32 NumPy files under `artifacts_dir/promolecular_density_matrices/<chembl_id>/conf_<conformer_id>.npy`. Training targets are padded total-density deltas, while inference and figures add the stored promolecular baseline back before computing metrics or downstream physical quantities.

### Learning only the upper triangle
To learn only the upper triangle of the total density, set `config.json` as follows:

```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0],
    "output_dim": [2005003],
    "output_names": [
        "density_matrix_uptr"
    ],
    "type": ["graph"],
    ...
}
```

To delta learn only the upper triangle of the total density, set `config.json` as follows:

```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0],
    "output_dim": [2005003],
    "output_names": [
        "density_matrix_delta_uptr"
    ],
    "type": ["graph"],
    ...
}
```


### Learning alpha and beta separately

For learning alpha and beta density matrices separately, set `config.json` as follows:
```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0, 1],
    "output_dim": [4008004, 4008004],
    "output_names": [
        "alpha_density_matrix",
        "beta_density_matrix"
    ],
    "type": ["graph", "graph"],
    ...
}
```

### Masking
NOTE: the padded region is by default masked out from the train, validation, and test losses. The `mask_output_loss` configuration only determines how the train and validation losses are further masked, and does NOT affect the test loss. 

The `data.y_mask` attribute stores this further train and validation mask. The `data.y_test_mask` attribute stores the default mask for the train and validation losses and the only mask for the test loss. 
