# QMugs

## Directory Layout
```
qmugs
|
├── eda                                    # results from eda.py
|   ├── densmat_eda.json                   # EDA configuration
|   ├── entries.png                        # dataset entry counts
|   └── sizes.png                          # density matrix size distribution
|
├── figs                                   # functions to plot things
|   ├── density_matrix_visualizers.py      # density matrix heatmaps
|   ├── density_visualizers.py             # electron density point clouds and isosurfaces
|   ├── dipole_moment_visualizers.py       # dipole moment vectors and parity plots
|   └── metrics_visualizers.py             # error metrics vs. molecule size
|
├── scripts                                # SLURM scripts, one subdirectory per machine
|   ├── dcc                                # Duke Compute Cluster
|   |   ├── qmugs_densmat_dcc.sh           # runs training
|   |   ├── qmugs_eda_dcc.sh               # runs EDA script (NO LONGER USED)
|   |   ├── qmugs_figs_dcc.sh              # runs figures given a previous inference run
|   |   ├── qmugs_inference_dcc.sh         # runs inference
|   |   ├── qmugs_pre_dcc.sh               # runs data preprocessing to prep for training
|   |   └── test.sh                        # tests environment and package setup
|   └── perlmutter                         # Perlmutter (NERSC), same six scripts
|
├── utils
|   ├── download_data.json                 # data download config, UPDATE THIS to point to your own directories
|   ├── download_data.py                   # Python wrapper for data download functions
|   ├── download_data.sh                   # shell script to run the data download
|   ├── inference_utils.py                 # utilities used only by the inference path
|   └── utils.py                           # utilities, importantly including density matrix pre/post-processing
|
├── eda.py                                 # script with EDA functionalities for the QMugs dataset
├── figs.py                                # Python script to run figure plotting
├── qmugs_densmat.json                     # HydraGNN configuration for the learning task
├── qmugs_densmat_inference.py             # Python script to run inference on a trained model
├── qmugs_densmat_train.py                 # Python script to run preprocessing or training
├── qmugs.py                               # Python file containing QMugsDataset and QMugsInference
├── requirements.txt                       # QMugs-specific dependencies
└── README.md

```
All other files in this directory are log files from previous runs.
---

## Dev Reminders for Perlmutter

### Module Loading and Environment Activation
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

---

## Dev for Duke Compute Cluster (DCC)

### Module Loading and Environment Activation
Before doing anything, make sure the correct modules are loaded on the machine. From the HydraGNN root directory, run
```
source installation_other/module_loads_dcc.sh
load_dcc_modules "12.8"
module -t list | sort
```

Activate the `conda` environment by running
```
conda activate /hpc/group/youlab/$USER/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv
```

Always load the modules BEFORE activating the environment. `load_dcc_modules` begins with `module purge`, which unloads `Anaconda3` and tears down the `conda` shell function; running it after activation silently leaves you on the system Python. Unlike on Perlmutter, `psi4` is installed by the installation script and needs no separate `conda install` step.

### Environment Setup

Set up the `conda` environment by running
```
bash installation_other/hydragnn_installation_bash_script_dcc.sh
conda activate /hpc/group/youlab/$USER/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv
python -m pip install -e . --no-deps
```
Note the `--no-deps` flag avoids reinstalling possibly incompatible versions of torch family modules after compatible ones have been installed by the installation script. 

The environment is built under `/hpc/group/youlab/$USER/`, NOT inside the repository as on Perlmutter. This is not a preference: the VAST filesystem backing `/work` rejects filenames containing `*`, `?`, `:`, or `|`, and `psi4` ships basis sets named e.g. `6-31g**.g94`, so it cannot be installed there. Override with `INSTALL_ROOT` if needed. See `references/dcc_info/perlmutter_to_dcc_transfer.md`. 

Prefer running the installation script inside a `youlab-gpu` batch job rather than on a login node, so that the CUDA checks are meaningful. Login nodes have no GPU. 

---

## Learning Task Configs

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


### Specifying output constraints

The density matrix learning pipeline supports 2 forms of output constraints. By "output constaints",
we mean constraints applied on the output of the last output head of the network before loss computation, such that the network architecture remains identical. 

For learning the total density matrix (full or upper triangle), the `trace_norm` constraint is implemented, which normalizes the trace of the output to equal the number of electrons in the molecule. 

```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0],
    "output_dim": [4008004],
    "output_names": [
        "density_matrix"
    ],
    "output_constraints": [
        ["trace_norm"]
    ],
    "type": ["graph"],
}
```

For delta learning the total density matrix (full or upper triangle), the `trace_zero` constraint is
implemented, which zeros out the trace of the delta matrix, as the promolecular and total density
matrices are expected to have equal trace.

```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0],
    "output_dim": [4008004],
    "output_names": [
        "density_matrix_delta"
    ],
    "output_constraints": [
        ["trace_zero"]
    ],
    "type": ["graph"],
    ...
}
```

## Other Architecture Configs (added for this task)

### Concat-style graph pooling

The default pooling modes (`"sum"`, `"mean"`, `"max"`) are permutation invariant; they collapse the
node embeddings into a single `hidden_dim`-wide vector in which the atom ordering is lost. But the
target density matrix is indexed by atomic-orbital blocks ordered by the atom ordering in the
SDF/wfn files, which is the same ordering as the graph nodes (specifically, the ordering used by Psi4 
for the def2-SVP basis). Concatenation pooling zero-pads the node embeddings to a fixed number 
of atoms and flattens them, so the head receives a per-atom resolved representation in which row/column 
position is still recoverable.

To use it, set the `Architecture` block of `config.json` as follows:

```
"Architecture": {
    "mpnn_type": "PAINN",
    "hidden_dim": 200,
    "graph_pooling": "concat",
    "output_heads": {
        "graph": {
            "num_sharedlayers": 0,
            "dim_sharedlayers": 400,
            "num_headlayers": 2,
            "dim_headlayers": [1000, 4000]
        }
    },
    ...
}
```

The pooled embedding has width `pooled_dim = max_nodes * hidden_dim`, where `max_nodes` is the
padding width in atoms.

**Specifying `max_nodes`.** You may set `"max_nodes"` explicitly in the `Architecture` block to pin
the architecture. If you leave it out, `qmugs_densmat_train.py` fills it from the `max_natoms` value
recorded in the pickle metadata during preprocessing, which is the largest atom count over the
train, validation, and test splits. Datasets preprocessed before this feature was added do not
carry `max_natoms`; training will stop with an error asking you to either set `max_nodes` by hand or
re-run preprocessing.

**Automatic widening of the first layer.** A single `Linear(pooled_dim, dim_sharedlayers)` would
contract the concatenated embedding by two orders of magnitude in one step. The training script
therefore widens whichever layer consumes the pooled embedding to `pooled_dim`, and logs a warning
naming the old value, the new value, and the contraction ratio it avoided:

- with `"num_sharedlayers": 0` the shared trunk is skipped entirely and the head consumes the
  pooled embedding directly, so `dim_headlayers[0]` is widened;
- with `"num_sharedlayers"` at least 1, `dim_sharedlayers` is widened instead.

Setting `"num_sharedlayers": 0` is the recommended configuration, since it leaves exactly one
`pooled_dim`-wide layer in the network and lets `dim_headlayers` control everything downstream of
it. Note that `dim_sharedlayers` must still be present in the config even when it is ignored.

Both the resolved `max_nodes` and the widened layer dimension are written into the saved
`logs/<run>/config.json`, so inference reconstructs the identical model with no extra flags.

**What to expect.**

- *Parameter count.* At `hidden_dim: 200` and roughly 160 atoms, `pooled_dim` is about 32,000 and
  the widened layer alone holds about 1.0e9 parameters, on top of whatever the final
  `Linear(dim_headlayers[-1], output_dim)` costs. Check the warning in the log for the exact figures
  for your dataset, and size `hidden_dim` and `dim_headlayers` accordingly.
- *Loss of permutation invariance.* This is the intended trade, but now the
  model is tied to the atom ordering in the source files. That ordering must be identical between
  preprocessing and inference, and the model will not generalize to a reordering of the same
  molecule. The QMugs SDF/wfn files provide a fixed ordering, which is the same ordering that
  defines the basis ordering of the target, so this holds for the pipeline as written.
- *Molecules larger than `max_nodes`.* Training or inference stops with an error naming the
  offending atom count. It does not silently drop atoms.
- *Supported models.* Only `"PAINN"` and `"EGNN"` accept `"concat"`. Other `mpnn_type` values 
  raise an error at model construction.

Existing checkpoints trained with `"sum"` or `"mean"` pooling are unaffected and still run under
`qmugs_densmat_inference.py` with no config edits, since each checkpoint is rebuilt from its own
saved `config.json`.
