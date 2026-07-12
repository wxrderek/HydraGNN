# `examples/qmugs/AGENTS.md`

**To confirm you have read these files at the beginning of each task, output: "Tetrads and octads permute sometimes; I have read AGENTS.md in examples/qmugs/" at the very beginning of the output, after the specified output for AGENTS.md at root**

This directory contains the task specific code for density matrix learning from the
QMugs dataset. Please read `AGENTS.md` at project root for high-level information.

---

# The current state

Currently, the following are implemented:

- raw data downloading and preprocessing (WILL NOT MODIFY FURTHER)
- exploratory data analysis (WILL NOT MODIFY FURTHER)
- QMugs dataset wrapper objects
- A training script for HydraGNN on the QMugs dataset
- A basic inference script for a model trained with HydraGNN on the QMugs dataset (very hard-coded)

---

# High-level goals

The following are short-term high level goals for the QMugs-specific code. 
Refer to the task specified in each prompt for details; this is just big picture.

- dramatically improve upon memory usage in loading and storing data as pickle files, specifically masks
- add code to compute scalar densities on a grid with `psi4` from true and predicted density matrices
- add code to compute the dipole moment from true and predicted density matrices as a downstream task
- add plotting functionalities to visualize molecules, densities, inference results, etc.

The following are goals for the HydraGNN module itself, which are necessary in order to implement functionalities
that would benefit the density matrix learning task.

- support for far more memory-efficient mask storage and usage
- support for custom loss functions
- support for postprocessing for supervision on downstream tasks

---

# Directory structure

The setup of this directory is as follows:

- `eda/`: results from `eda.py` (WILL NOT MODIFY FURTHER)
- `figs/`: code and results from plotting/visualizations
    - `density_visualizers.py`: wrappers/functions to visualize densities
- `scripts/`: shell scripts to run Perlmutter jobs to run Python scripts
    - `qmugs_densmat_perlmutter.sh`: runs training
    - `qmugs_eda_perlmutter.sh`: runs EDA (NO LONGER USED)
    - `qmugs_inference_perlmutter.sh`: runs inference 
    - `qmugs_pre_perlmutter.sh`: runs data loading and preprocessing prior to training
    - `test.sh`: test for environment and package setup
- `utils/`: utility functions, including data loading, pre/postprocessing
    - `densmat.png`: plot of matrix heatmap with masking (ARTIFACT)
    - `download_data.json`: configuration for data download (WILL NOT MODIFY FURTHER)
    - `download_data.py`: python script for data download and parsing (WILL NOT MODIFY FURTHER)
    - `download_data.sh`: shell script for data download (WILL NOT MODIFY FURTHER)
    - `utils.py`: utility functions, including
        - density matrix pre-processing
        - density matrix masking
        - grid building (maybe unnecessary)
- `qmugs.py`: classes for QMugs dataset integration with HydraGNN
    - `QMugsDataset`: class representing loaded subset of QMugs dataset, with functions for loading
    - `QMugsInference`: class for inference with a pretrained HydraGNN model
    -  various helper functions for internal use in the two classes
- `qmugs_densmat_train.py`: script to either preprocess QMugs data for training, ot run HydraGNN training on preprocessed data
- `qmugs_densmat_inference.py`: script to run inference, currently very hardcoded
- `qmugs_densmat.json`: configuration file for HydraGNN functionalities

All other files are log files from previous runs. 

---

# Remarks

## Dataset loading

I will ALWAYS assume that datasets are loaded in pickle format instead of ADIOS. 
Prioritize compatibility with pickle format. If this forces incompatibility with ADIOS
unless heavy, complex changes must be made, do not worry about ADIOS compatibility
and through an error if a user tries to use an ADIOS dataset. 

---

## Density matrix padding

A key challenge in the task is that density matrices have non-uniform sizes. 
Currently, I naively pad all matrices with 0's up to the maximum size in the dataset.
I mask the loss from supervising on the padded entries.

Among other issues, this stresses memory resources tremendously. 
The current time frame does not allow us to try a different strategy. 
We will just seek to make this type of padding and masking maximally memory efficient. 


