# Task: Upgrade Inference

Status: DONE
Estimated scope: small

---

# Objective

Update the `qmugs_densmat_inference.py` script to remove hard-coded bits and improve reproducibility.
Update the `qmugs_inference_perlmutter.sh` bash script to reflect these changes. 

More specifically:
- remove the hardcoded paths in `qmugs_densmat_inference.py`, offload them to `qmugs_inference_perlmutter.sh` as needed
- add argument-based parsing of configurations to `qmugs_densmat_inference.py` in a similar style to that in `qmugs_densmat_train.py`
- update `qmugs_inference_perlmutter.sh` accordingly to match the new run patterns

This should be a short, simple code modification which does not change core functionality.

---

# Motivation

Reproducible inference runs with minimal hard-coding.

---

# Files to edit

- `qmugs_densmat_inference.py`
- `qmugs_inference_perlmutter.sh`
- maybe `utils/utils.py`, but likely not

---

# Files to refer to

- `qmugs_densmat_train.py`: reference code style and structure of arg-parsing
- `qmugs_densmat_perlmutter.sh`: reference code style

---

# Constraints

Edit only the files specified above. 

Do NOT:

- change random seeds
- change underlying functionality

Prefer:

- minimal edits
- readable code for scientists

---

# Validation

Do NOT run code.
I will validate myself after the code edits. 

---

# Deliverables

Provide: 

1. summary of changes, with brief but precise explanations
2. files modified
3. risks

---

# Out of scope

- modifications to underlying inference functionality
- modifications to other unspecified files

---

# Other references

Make sure these are in your memory:

- `AGENTS.md` at root
- `examples/qmugs/AGENTS.md`
