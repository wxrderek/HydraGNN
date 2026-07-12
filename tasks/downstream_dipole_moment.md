# Task: Downstream Dipole Moment

Status: Done
Estimated scope: medium

---

# Objective

Implement a downstream dipole moment calculation to the inference pipeline using `psi4`. 

More specifically:

- modify `qmugs.py`'s `QMugsInference` class, specifically `run_streaming_inference()` to add the ability to run downstream physics calculations, i.e. add new methods that are run if flagged in the arguments passed in; have (1) a boolean flag `run_downstream_calculation` to determine whether they are run, and (2) a list of strings `downstream_calculations` that determines which tasks to pick and run; each task should have its own method called to run it. 

- add specifically a dipole moment downstream calculation from true and predicted density matrices; keep the assumption that only total density matrix predictions are supported; use existing `psi4` functions as much as possible; check the documentation.

- add evaluation criteria that automatically runs for each specified calculation; make sure it is compatible with potential custom criteria for specific downstream tasks; no need to define a custom criterion for the dipole moment calculation yet.

- add a framework for downstream ML predictions (note this is distinct from the physics calculations immediately available, such as for dipole moments) in `run_streaming_inference()`; have (1) a boolean flag `run_downstream_prediction` to determine whether they are run, and (2) a list of strings `downstream_predictions` that determines which tasks to pick and run; there is NO NEED to implement any actual tasks for this yet.

- update `qmugs_densmat_inference.py` and `qmugs_inference_perlmutter.sh` to reflect changes in the `QMugsInference` class; specifically avoid hardcoding which `downstream_calculations` or `downstream_predictions` are run in `qmugs_densmat_inference.py`, offload this specification to `qmugs_inference_perlmutter.sh`.

Note for the inference pipeline, you MAY CHANGE the patterns seen by users such that older workflows are defunct, since this pipeline is not near finalized.

---

# Motivation

Testing the downstream utility of learned density matrices.

---

# Files to edit

- `qmugs.py`
- `qmugs_densmat_inference.py` just to change arguments passed in
- `qmugs_inference_perlmutter.sh` just to allow for new arguments to be passed

---

# Files to refer to

Nothing specific beyond the files to edit, unless necessary. 

Do reference other Python scripts in `qmugs` for commenting style.

---

# Constraints

Edit only the files specified above. 

Do NOT:

- change random seeds
- copy any changes to the two train running directories

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
2. a description of precisely how the dipole moment is calculated, both programmatically and how `psi4` does it internally; the target audience is an applied mathematician
3. files modified
4. risks

---

# Out of scope

- modifications to the logic that treats node-level prediction targets; we are only concerned with graph-level targets
- modifications to patterns seen by users which makes previously used patterns defunct (i.e. older examples should still run without errors after these changes). Obviously, passing in additional optional arguments may be implemented
- the usual pipeline of copying to the two training running directories, until I have verified the updates

---

# Other references

Make sure you have read these files:

- `AGENTS.md` at root
- `examples/qmugs/AGENTS.md`
