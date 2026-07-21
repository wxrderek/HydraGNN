# Task: Downstream Density

Status: Done
Estimated scope: small

---

# Objective

Implement downstream functionality to compute the electron density as a scalar field with `psi4`.

More specifically:

- modify `qmugs.py`'s `QMugsInference` class, specifically `run_streaming_inference()`, to accept in `downstream_calculations` a new "electron_density" string, which triggers the evaluation of electron density as a scalar field on a grid by `psi4`; this is within the framework implemented for in the downstream dipole moment calculations task.

- evidently, do not save the computed electron densities to memory or in a directory yet, that will come in a later task; only save the evaluation metrics.

- implement at least the following metrics: 
    - mean absolute voxel grid error: use differences between predicted vs true densities at each grid point
    - mean squared voxel grid error: same as above
    - root mean squared voxel grid error
    - normalized root mean squared voxel grid error, normalized by max-min
    - all the evaluation criterion used by Tuckerman's group in `references/tuckerman-densnet.txt`

- regarding the voxel grid used, use whatever `psi4` deems to be default; if this is not accessible, please state this and ask.

- do NOT use the grid-building helpers in `utils/utils.py`.

You do not need to plot anything yet. 

Ensure you leave inline comments as specified in the Validation section of this markdown.

---

# Motivation

Recover the scalar field version of electron density. 

---

# Files to edit

- `qmugs.py`
- `qmugs_densmat_inference.py` if needed
- `qmugs_inference_perlmutter.sh` if needed

---

# Files to refer to

The Tuckerman DensNet paper in `references/tuckerman-densnet.txt` for physically meaningful criterion to evaluate density reconstruction.

Do reference other Python scripts in `qmugs` for commenting style.

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
- add short, one sentence docstrings to the new methods in `QMugsInference`; ; follow the style of other scripts in the `qmugs` directory

---

# Deliverables

Provide: 

1. summary of changes, with brief but precise explanations
2. a description of precisely how the electron density on a grid is calculated, both programmatically and how `psi4` does it internally; the target audience is an applied mathematician
3. a description of what the default grid size is and how it is chosen
4. files modified
5. risks

---

# Out of scope

- modifications to the newly implemented downstream task pipeline in the inference scheme
- the usual pipeline of copying to the two training running directories, until I have verified the updates

---

# Other references

Make sure you have read these files:

- `AGENTS.md` at root
- `examples/qmugs/AGENTS.md`
