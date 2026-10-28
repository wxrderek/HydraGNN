# `examples/qmugs/CLAUDE.md`

This directory contains the task specific code for density matrix learning from the
QMugs dataset. Please read `CLAUDE.md` at project root for high-level information.

---

# The current state

Currently, the following are implemented:

- raw data downloading and preprocessing (WILL NOT MODIFY FURTHER)
- exploratory data analysis (WILL NOT MODIFY FURTHER; archived)
- QMugs dataset wrapper objects
- A training script for HydraGNN on the QMugs dataset
- An inference script for HydraGNN on the QMugs dataset
- Plotting functionalities

---

# High-level goals

The one goal right now is to push the Schur decoder to be as good as possible while keeping dataset size and training time relatively fixed.
By "good", we mean (1) beating the baseline promolecular densities, and (2) beating the dense head models. 
Thus, all modifications should prioritize the Schur decoder; compatibility with the dense head takes second wheel, and can be back-implemented later.

The following are goals for the HydraGNN module itself which specificlaly benefit the density matrix learning task.

- support for constraints on outputs
- concatenation and attention based global graph pooling.
- support for custom loss functions
- support for postprocessing for supervision on downstream tasks

---

# Directory structure

The setup of this directory is as follows:

- `figs/`: plotting functions used by `figs.py`
    - `density_visualizers.py`: electron density point clouds and isosurfaces
    - `density_matrix_visualizers.py`: density matrix heatmaps and eigenvalue spectra
    - `dipole_moment_visualizers.py`: dipole moment vectors and parity plots
    - `metrics_visualizers.py`: error metrics against molecule size
    - `schur_decoder_visualizers.py`: Schur decoder figures, i.e. block error against step, pivots and latent norms
- `scripts/dcc/`: SLURM scripts for the Duke Compute Cluster, submitted from this directory
    - `qmugs_pre_dcc.sh`: preprocessing for the dense head
    - `qmugs_schur_pre_dcc.sh`: preprocessing for the Schur decoder, writes the same dataset as `qmugs_pre_dcc.sh`
    - `qmugs_densmat_dcc.sh`: training with the dense head
    - `qmugs_schur_dcc.sh`: training with the Schur decoder
    - `qmugs_add_overlap_dcc.sh`: adds the overlap matrix and the number of electrons to an existing pickle dataset
    - `qmugs_inference_dcc.sh`: inference with a dense head model
    - `qmugs_schur_inference_dcc.sh`: inference with a Schur decoder model
    - `qmugs_figs_dcc.sh`: figures from a finished dense head inference run
    - `qmugs_schur_figs_dcc.sh`: figures from a finished inference run, Schur or dense, configured only through environment variables
    - `qmugs_schur_trace_test_dcc.sh`: unit tests of the Schur decoder trace normalization
    - `qmugs_composite_loss_test_dcc.sh`: unit tests of the composite loss terms
    - `mpi_rank_env.sh`: per-rank environment shim used by `mpirun`
    - `test.sh`: test for environment and package setup
- `schur_long/`: the 400-epoch Schur decoder run and its evaluation (not tracked by git; `examples/qmugs/.gitignore`)
    - `README.md`: restart card and history of the run
    - `REPORT-2026-10-21.md`: evaluation of the finished run, comparison with the dense head, epoch cutoff, next steps
    - `run_segment.sh`: SLURM script that runs one training segment and resumes from the newest checkpoint
    - `plan_segment.py`: decides how the next segment starts, including rollback and learning rate halving after a crash
    - `make_segment_config.py`: writes the config of a segment that resumes an earlier one
    - `monitor.py`: runs `qmugs_densmat_train.py` unchanged with read-only probes
    - `summarize.py`: summary tables and figures of all segments, from the ledger `logs/schur-long/`
    - `qmugs_schur_400.json`: configuration of the run
- `loss_trials/`: launchers for loss and constraint trials on the Schur decoder; the 2026-10-22 round is archived (not tracked by git; `examples/qmugs/.gitignore`)
    - `README.md`: how to run a trial round, what was archived
    - `REPORT-2026-10-23.md`: report of the 2026-10-22 round (nine configurations, 50 epochs), copied from its archived ledger
    - `make_trial_configs.py`: writes the trial configs `dense_cN.json` and `schur_cN.json` from the two base configs
    - `run_trial_segment.sh`: SLURM script for one Schur trial segment, runs `schur_long/run_segment.sh`, then evaluates a finished campaign
    - `eval_best.sh`: submits the evaluation of the best checkpoint of a Schur trial campaign, its figures and its downstream evaluation
    - `figs_after.sh`: submits the figures of one evaluation job, to start once it has succeeded
    - `downstream.sh`: submits the downstream evaluation (electron density, dipole moment) of one Schur checkpoint on the 186-conformer test subset
- `slurm_output/`: SLURM logs, one subdirectory per job type, see its `README.md`
- `utils/`: utility functions, including data loading, pre/postprocessing
    - `download_data.json`: configuration for data download (WILL NOT MODIFY FURTHER)
    - `download_data.py`: python script for data download and parsing (WILL NOT MODIFY FURTHER)
    - `download_data.sh`: shell script for data download (WILL NOT MODIFY FURTHER)
    - `utils.py`: utility functions, including
        - density matrix pre-processing
        - density matrix masking
    - `inference_utils.py`: utility functions for inference specifically
    - `add_overlap.py`: adds the overlap matrix and the number of electrons to an existing pickle dataset, with ground-truth checks
- `eda/densmat_eda.json`: density matrix sizes per molecule, read by preprocessing with a size cap (copied back from the archive 2026-10-21)
- `qmugs.py`: classes for QMugs dataset integration with HydraGNN
    - `QMugsDataset`: class representing loaded subset of QMugs dataset, with functions for loading
    - `QMugsInference`: class for inference with a pretrained HydraGNN model, dense head or Schur decoder
    -  various helper functions for internal use in the two classes
- `qmugs_densmat_train.py`: script to either preprocess QMugs data for training, or run HydraGNN training on preprocessed data
- `qmugs_densmat_inference.py`: script to run inference, dense head or Schur decoder
- `schur_trace_test.py`: unit tests of the Schur decoder trace normalization, CPU float64
- `composite_loss_test.py`: unit tests of the composite loss terms in `hydragnn/domain_losses/`, CPU float64
- `figs.py`: script to run figure plotting given a previous inference run
- `qmugs_densmat.json`: configuration file for the dense head
- `qmugs_schur.json`: configuration file for the Schur decoder
- `requirements.txt`: QMugs-specific dependencies
- `README.md`: README file for devs

Run outputs go to `logs/` at the repository root. `logs/baselines/README.md` lists the kept models (Schur epoch 391, dense epoch 13), their reference evaluations and the recorded train, validation and test split. Code and outputs that are no longer used were moved to `/work/xw262/HydraGNN_archive/`: EDA, the Perlmutter scripts, the sync checks and the Schur decoder unit tests on 2026-10-21; the loss trials, the overlap dataset checks, the old baseline evaluations and the October 2026 presentation on 2026-10-28. See its `README.md` and `MANIFEST.tsv`.

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


