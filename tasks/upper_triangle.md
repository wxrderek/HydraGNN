# Task: Predict Upper Triangles of Density Matrices Only

Status: Done
Estimated scope: medium / large

---

# Objective

Modify the `qmugs/` scripts to add the option to predict only the upper triangle of density matrices. 
Only ensure compatibility for learning the total density matrix or delta learning the total 
density matrix, no need to change things for predicting alpha and beta separately.

Specifically:

- in `qmugs.py`:
    - for the `QMugsDataset` class
        - add a boolean `upper_triangle` flag to the initialization
        - if the flag is true, then only the upper triangle entries of density matrices are stored in the dataset object
        - update masking to support upper triangle predictions
    - for the `QMugsInference` class
        - if the object loads a model that predicts upper triangles, make sure the inference metrics are computed on the WHOLE matrix, i.e. construct the full square matrix of each sample after it is loaded for inference
- in `qmugs_densmat_train.py`:
    - make things compatible if the `output_name` read from the config JSON is `density_matrix_uptr` or `density_matrix_delta_uptr`
    - make the `--preonly` path handle the upper triangle cases
- in `qmugs_densmat_inference.py`:
    - ensure compatibility with the new `output_name` options
- in `figs.py`:
    - ensure compatibility if the loaded dataset is upper triangles, i.e. ensure all figures plot the WHOLE matrix, not just the upper triangle
    - ensure this compatibility for both paths of (1) a standalone plotting call or (2) a plotting call at the end of an inference run
- update the SLURM script `qmugs_pre_perlmutter.sh` to take an argument to flag whether to load and learn the upper triangle.
- add helpers in `utils/utils.py` to handle upper triangle pre and postprocessing as needed.

Note that this may be an incomplete list of file changes. 

Specifically, we may need to check `Base.py` and `train_validate_test.py` in `hydragnn/` to see whether masking treatment 
needs to be further changes to accomodate upper triangles, including both the masking for the padded entries
and the optional random masking of any entries.

---

# Motivation

One-electron density matrices are:
(a) Hermitian
(b) positive semidefinite
(c) have traces equal to the number of electrons

Out of these, we will implement (c) trace normalization analytically on the model output, instead of
toying with the internal algorithms.

In this task, we are implementing the Hermitianity constraint by only predicting the upper triangle,
and just mirroring it over the diagonal to get the whole matrix.

---

# Files to edit

- `utils/utils.py` if needed to add a helper for upper triangle testment
- `qmugs.py`
- `qmugs_densmat_train.py`
- `qmugs_pre_perlmutter.sh`
- `qmugs_densmat_perlmutter.sh` if needed
- `qmugs_densmat_inference.py` if needed
- `qmugs_inference_perlmutter.sh` if needed
- `figs.py`
- `Base.py` in `hydragnn/`
- `train_validate_test.py` in `hydragnn/` 

Additionally, do not change `qmugs_densmat.json` directly, but ensure compatibility given the new specification in `examples/qmugs/README.md` of the "Learning only the upper triangle" path.

---

# Files to refer to

Reference other Python scripts in `qmugs/` for commenting style.

---

# Constraints

Edit only the files specified above. If I seem to be missing a file, please notify me.

Do NOT:

- change random seeds
- refer to ANY FILES in `qmugs_run/`

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
2. a description of the pipeline of the entire upper triangle path, including data loading, training, inference and figures
3. files modified
4. risks

---

# Out of scope

- modifications to the core training and inference logic; only the treatment of the data and potentially maskingfor pre and postprocessing needs to change.
- copying to `qmugs_run.py`; I will do so manually after verifying.

---

# Other references

Make sure you have read these files:

- `CLAUDE.md` at root
- `examples/qmugs/CLAUDE.md`
