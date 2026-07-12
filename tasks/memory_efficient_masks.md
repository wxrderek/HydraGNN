# Task: Memory Efficient Masks

Status: Done
Estimated scope: medium

---

# Objective

Training currently runs out of memory. A major reason is the storage of masks as massive
tensors of 0's and 1's. 

Update the `QMugsDataset` class in `qmugs.py` to create and store masks much more efficiently.
Update the `qmugs_densmat_train.py` script as needed to reflect this.
In `hydragnn/`, update `preprocess/graph_samples_checks_and_updates.py`, `models/Base.py`, and `train/train_validate_test.py` to adapt to this. Do only absolutely necessary changes. 

Do NOT copy changes into `qmugs_run/` and `qmugs_run_mask` yet! I must validate changes
first, as there are train jobs in the queue. 

You may consider doing a slightly more drastic change, removing
the storage of padding on matrices in the loaded data, passing the max padded size
to preprocessing (or saving it in each `torch_geomtric` `Data` object), and 
modifying `hydragnn/`to perform padding. Simultaneously, this cuts
the need to store and process `y_test_mask`, which only corresponds to padded entries. 

Note that `y_mask` still must be stored in the data objects. 

---

# Motivation

Getting past the problem of massive memory consumption during training, and massive
storage consumption of loaded pickled datasets.

---

# Files to edit

- `qmugs.py`
- `qmugs_densmat_train.py` if needed
- `hydragnn/preprocess/graph_samples_checks_and_updates.py`
- `hydragnn/models/Base.py`
- `hydragnn/train/train_validate_test.py`
- potentially other files in `hydragnn/`, only if necessary due to an import from one of the above files which relates to the new functionality

---

# Files to refer to

Nothing specific beyond the files to edit, unless necessary. 

---

# Constraints

Edit only the files specified above. 

Do NOT:

- change random seeds
- change patterns for user

Prefer:

- minimal edits
- readable code for scientists

---

# Validation

Do NOT run code.
I will validate myself after the code edits. 

Do the following as preliminary validation:

- add inline comments to indicate tensor sizes whenever application, on your new code only
- understand how the density matrix is being flattened and treated during preprocessing
- understand how this now differs from the treatment of masks

---

# Deliverables

Provide: 

1. summary of changes, with brief but precise explanations
2. more detailed explanation of how density matrices, passed as square tensors OR as flattened vectors (in the case of separate alpha and beta predictions), are treated by the pipeline
3. more detailed explanation of how, given the updates, padidng and masking are done
4. files modified
5. risks

---

# Out of scope

- modifications to inference
- modifications to the logic that treats node-level prediction targets; we are only concerned with graph-level targets
- modifications to patterns seen by users which makes previously used patterns defunct (i.e. older examples should still run without errors after these changes). Obviously, passing in additional optional arguments may be implemented
- the usual pipeline of copying to the two training running directories, until I have verified the updates

---

# Other references

Make sure these are in your memory:

- `AGENTS.md` at root
- `examples/qmugs/AGENTS.md`

They have NOT been updated since the last task. 
