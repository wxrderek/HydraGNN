# Task: Delta Learning for Total Density Matrices

Status: Done
Estimated scope: medium / large

---

# Objective

Implement a path which performs delta learning on total density matrices, with promolecular density matrices as a baseline.

From now, '1-RDM' and 'density matrix' are used interchangeably. the underlying object is, most precisely, the one-electron reduced density matrix.

More specifically, in ONLY the `examples/qmugs/` directory for now:

- in `utils/utils.py`: 
    - add a new function `promolecular_density_matrix()` which takes in a Psi4 molecule object (Wavefunction or Molecule) and outputs its promolecular total density matrix (promolecular 1-RDM), i.e. the approximation of the 1-RDM by overlapping densities of non-interacting atoms. the output should be `numpy` formatted. the computation should try to use existing dependencies, namely Psi4 and RDKit, if possible; if not, suggest a new dependency.

- in `qmugs.py`:
    - modify `QMugsDataset` to optionally load data for delta learning for only the total density matrix (raise NotImplemented for separate alpha/beta predictions), i.e. for each sample, get the promolecular 1-RDM given by the `promolecular_density_matrix()` newly written above and load the delta matrix (full 1-RDM minus promolecular 1-RDM) as the training target in each Data object. this should be triggered by a new argument to the initialization of the dataset object, passed in via the `qmugs_densmat.json` config file by setting `var_config["output_names"]` to `density_matrix_delta`.
    - modify `QMugsInference` to, depending on the saved config of the loaded pretrained model, optionally steer towards the delta learning path. the only difference in processing should be that both the ground truth and predicted matrices in the dataset itself will be the delta matrix, meaning for all metrics computations and downstream calculation/prediction, the true and predicted matrices should be the true and predicted delta matrix, plus the promolecular baseline, such that we are still evaluating the final reconstruction, not the delta itself. the promolecular baselines should be read from the specified `artifacts_dir`, which will be assumed to be the same as that specified in the training of the model.

- add an `artifacts_dir` argument to `qmugs_pre_perlmutter.sh`, in a very similar stype to that in `qmugs_inference_perlmutter.sh`, such that if `density_matrix_delta` is specified, the computed promolecular density matrices are saved to the artifacts directory by `qmugs_densmat_train.py`'s data loading logic (under a separate subdirectory as `cubeprop` used by `qmugs_densmat_inference.py`, evidently).

- update `figs.py` such that if the provided inference directory was for a delta learning model, then all plots are plotted for the total density matrix, not the delta matrix.

- store the `var_config["output_names"]` read from the `config.json` in the model directory (from which the pretrained model is loaded) in the inference config JSON file saved by `qmugs_densmat_inference.py`.

- the inference metrics computed should stay the exact same. the only difference, again, is that prior to computing them, the assumption is that the stored data corresponds to the deltas, so the promolecular baseline must be added prior to metrics computation and downstream calculations/predictions. 

Ensure you leave inline comments and docstrings as specified in the Validation section of this markdown.

---

# Motivation

Models so far perform poorly on
1. off diagonal entries; basically everything is smoothed to a small number
2. hydrogen atoms, which are ordered last in the Psi4 canonical ordering of the def2-SVP basis set, and also have lower occupation numbers, so the models have very large variance on their blocks in the matrix

To attempt a mitigation, we implement the well-established delta learning framework, with a baseline of the promolecular density matrix. This means the delta matrix the model is predicting has much less variance in entries (i.e. the diagonal will not be much more intense than the off diagonal), which hopefully encourages the model to do a better job on the off diagonal instead of treating it as noise.

---

# Files to edit

- `utils/utils.py`
- `qmugs.py`
- `qmugs_densmat_train.py`
- `qmugs_pre_perlmutter.sh`
- `qmugs_densmat_perlmutter.sh` if needed; note the training logic does not need to see the artifacts directory containing promolecular baseline matrices, only the data loading logic needs this
- `qmugs_densmat_inference.py`
- `qmugs_inference_perlmutter.sh` if needed
- `figs.py`
- `README.md` if needed

Additionally, do not change `qmugs_densmat.json` directly, but ensure compatibility given the new specification in `examples/qmugs/README.md` of the "Delta learning total density" path.

---

# Files to refer to

`references/sad.cc` and `references/sad.h` contain some source code for Psi4 SAD machinery.

Reference other Python scripts in `qmugs/` for commenting style.

---

# Constraints

Edit only the files specified above. If I seem to be missing a file, please notify me.

Do NOT:

- change random seeds
- copy any changes to the two train running directories
- refer to ANY FILES in the train-running directories, namely `qmugs_run/` and `qmugs_run_mask/`

Prefer:

- minimal edits
- readable code for scientists

---

# Validation

Do NOT run code.
I will validate myself after the code edits. 

Do the following as preliminary validation:

- add inline comments to indicate tensor sizes whenever application, on your new code only; follow the style of other scripts in the `qmugs/` directory
- add short, one sentence docstrings to all new methods written, including helpers; follow the style of other scripts in the `qmugs/` directory

---

# Deliverables

Provide: 

1. summary of changes, with brief but precise explanations
2. a description of the pipeline of the entire delta learning path
3. files modified
5. risks

---

# Out of scope

- modifications to the core training and inference logic; only the treatment of the data for pre and postprocessing needs to change.
- copying to the two training running directories, until I have verified the updates

---

# Other references

Make sure you have read these files:

- `AGENTS.md` at root
- `examples/qmugs/AGENTS.md`
