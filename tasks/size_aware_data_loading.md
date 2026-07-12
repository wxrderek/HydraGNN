# Task: Size Aware Data Loading

Status: Done
Estimated scope: small

---

# Objective

Implement the ability to sample molecules with density matrices of at most a specified size.

More specifically, in the `qmugs/` directory ONLY (not `qmugs_run/` or `qmugs_run_mask/`):

- modify `qmugs.py`'s `QMugsDataset` class, specifically `load_and_split_dataset()` and `load_and_split_dataset_pickle_streaming()`, to accept a new argument `max_density_matrix_size`; if specified, only molecules with density matrices of sizes less than or equal to this integer should be sampled in the dataset.

- assume that `eda.py` has been run such that in `qmugs/eda/`, there is the file `densmat_eda.json` structured like:

```
{
    "sizes": {
        "CHEMBL1": 720,
        "CHEMBL100017": 607,
        "CHEMBL100039": 837,
        "CHEMBL100056": 520,
        ...
        ...
    }
}
```

- use this EDA result file to inform the sampling.

- add another argument to `load_and_split_dataset()`, a boolean flag called `update_max_padded_dimension`, which if true, updates the class variable `max_padded_density_matrix_dimension` to exactly the specified `max_density_matrix_size` (obviously only update if this is specified).

- modity `qmugs_densmat_train.py` and `qmugs_densmat_perlmutter.sh` to reflect the changes in `qmugs.py` and such that `max_density_matrix_size` and `update_max_padded_dimension` can be passed in as an argument.

---

# Motivation

To combat the huge disparity of density matrix sizes in loaded datasets.

---

# Files to edit

- `qmugs.py`
- `qmugs_densmat_train.py`
- `qmugs_densmat_perlmutter.sh` if needed

---

# Files to refer to

- `qmugs/eda.py` for what gets spits out in the EDA results. Note ONLY sizes have being parsed and saved.
- other Python scripts in `qmugs/` for commenting style.

---

# Constraints

Edit only the files specified above. 

Do NOT:

- change random seeds
- copy any changes to the two train running directories
- refer to ANY FILES in the train-running directories, namely `qmugs_run` and `qmugs_run_mask`

Prefer:

- minimal edits
- readable code for scientists

---

# Validation

Do NOT run code.
I will validate myself after the code edits. 

Do the following as preliminary validation:

- add inline comments to indicate tensor sizes whenever application, on your new code only; follow the style of other scripts in the `qmugs` directory
- add short, one sentence docstrings to all new methods; ; follow the style of other scripts in the `qmugs` directory

---

# Deliverables

Provide: 

1. summary of changes, with brief but precise explanations
2. files modified
3. risks

---

# Out of scope

- modifications to the core logic of the QMugs pipeline
- the usual pipeline of copying to the two training running directories, until I have verified the updates

---

# Other references

Make sure you have read these files:

- `AGENTS.md` at root
- `examples/qmugs/AGENTS.md`
