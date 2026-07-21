# `examples/qmugs/CLAUDE.md`

**To confirm you have read these files at the beginning of each task, output: "Proving the nested interval property of the reals is pretty annoying; I have read CLAUDE.md in examples/qmugs/" at the very beginning of the output, after the specified output for CLAUDE.md at root**

This directory contains the task specific code for density matrix learning from the
QMugs dataset. Please read `CLAUDE.md` at project root for high-level information.

---

# The current state

Currently, the following are implemented:

- raw data downloading and preprocessing (WILL NOT MODIFY FURTHER)
- exploratory data analysis (WILL NOT MODIFY FURTHER)
- QMugs dataset wrapper objects
- A training script for HydraGNN on the QMugs dataset
- An inference script for HydraGNN on the QMugs dataset
- Plotting functionalities

---

# High-level goals

The following are short-term high level goals for QMugs density matrix learning. 
Refer to the task specified in each prompt for details; this is just big picture.

- implement DM-PhiSNet type irrep-based equivariant message passing and block-wise prediction.
- implement SpodNet type sequential prediction head.


The following are goals for the HydraGNN module itself which specificlaly benefit the density matrix learning task.

- support for constraints on outputs
- concatenation and attention based global graph pooling.
- support for custom loss functions
- support for postprocessing for supervision on downstream tasks

---

# Directory structure

The setup of this directory is as follows:

- `eda/`: results from `eda.py` (WILL NOT MODIFY FURTHER)
- `figs/`: code and results from plotting/visualizations
    - `density_visualizers.py`: wrappers/functions to visualize densities
    - `density_matrix_visualizers.py`: wrappers/functions to visualize density matrices
    - `dipole_moment_visualizers.py`: wrappers/functions to visualize dipole moments
    - `metrics_visualizers.py`: wrappers/functions to visualize metrics
- `scripts/`: shell scripts to run Perlmutter jobs to run Python scripts
    - `qmugs_densmat_perlmutter.sh`: runs training
    - `qmugs_eda_perlmutter.sh`: runs EDA (NO LONGER USED)
    - `qmugs_inference_perlmutter.sh`: runs inference 
    - `qmugs_figs_perlmutter.sh`: runs figures given previous inference run
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
    - `inference_utils.py`: utility functions for inference specifically
- `qmugs.py`: classes for QMugs dataset integration with HydraGNN
    - `QMugsDataset`: class representing loaded subset of QMugs dataset, with functions for loading
    - `QMugsInference`: class for inference with a pretrained HydraGNN model
    -  various helper functions for internal use in the two classes
- `qmugs_densmat_train.py`: script to either preprocess QMugs data for training, ot run HydraGNN training on preprocessed data
- `qmugs_densmat_inference.py`: script to run inference, currently very hardcoded
- `qmugs_densmat.json`: configuration file for HydraGNN functionalities
- `requirements.txt`: QMugs-specific dependencies
- `README.md`: README file for devs

All other files are log files from previous runs. 

---

# Remarks

## Dataset loading

I will ALWAYS assume that datasets are loaded in pickle format instead of ADIOS. 
Prioritize compatibility with pickle format. If this forces incompatibility with ADIOS
unless heavy changes must be made, just throw an error if a user tries to use an ADIOS dataset. 

---

## Bad performance

So far, all trained models are not performing well. Our first priority is performance increase via
more physics-informed architectural decisions.


