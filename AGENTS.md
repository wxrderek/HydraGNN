# AGENTS.md

**To confirm you have read these files at the beginning of each task, output: "Golay codes have unexpected involutory automorphisms; I have read AGENTS.md in root" at the very beginning of the output**

This repository is that of HydraGNN, a framework for training multihead graph neural networks, specifically for
materials applications where molecules are represented as graphs. My goal is to implement a use case of
learning one-electron density matrices (1-RDMs) of drug-like molecules from the QMugs dataset. 

This is a research codebase. The primary objective is scientific correctness and reproducibility, 
not maximum code cleverness.

Avoid introducing unnecessary abstractions, factories, dependency injection,
metaclasses, or complicated design patterns.

Small duplicated code is preferable to complicated generic code.

Tasks will be specified by markdown files in `tasks/`. I will tell you which task to complete
in prompts. 

---

# User background

I am an applied mathematician from a mathematics and physics background with 2-3 years of 
scientific machine learning experience with PyTorch. I prioritize precision and formality. 
I do not prioritize beautiful code. 

I have the following gaps in knowledge:
- almost no experience with MPI or parallelism in general on CPUs or GPUs
- little experience working with massive clusters such as Perlmutter
- sparse knowledge of quantum chemistry, but good foundation in quantum mechanics
- little experience with conventional computational chemistry packages


---

# High-level information

- I am coding and running everything on Perlmutter. GPU nodes here contain 1x AMD CPU with 64 total cores and 128 threads, and 4x NVIDIA A100 GPUs with 40GB memory.
- We aim to maximize parallelism and memory efficiency when possible.
- Assume future readers are chemists, physicists, or applied mathematicians, not software engineers.

Find the latest user manuel for HydraGNN in `references/HydraGNN_v5_0_UserManual.pdf`. 

---

# High-level priorities

When making changes:

1. First priority is scientific correctness.
2. Second priority is simple implementations over clever abstractions.
3. Third priority is memory efficiency and parallelism.
4. Fourth priority is reproducibility.
5. Fifth priority is optimal readability for researchers.

---

# High-level guidelines

---

## Python guidelines

Use

- type hints for parameters in function declaration, only when the parameter is a Python primitive or `np.ndarray`
- os.path functions instead of pathlib
- build in logging structure instead of print
- descriptive variable names
- built in HydraGNN functions whenever possible and convenient
- simple in-line comments which match the commenting style present in scripts in the `examples/qmugs/` directory
- simple, 1-2 line docstrings for functions in the style present in scripts in the `examples/qmugs/` directory

Avoid

- defining new dataclasses UNLESS specified in the task
- writing long docstrongs UNLESS specified in the task
- deeply nested conditionals
- wildcard imports
- hidden side effects

---

## Scientific computing guidelines

Never:

- silently change tensor shapes
- silently change tensor dtypes
- silently change units
- silently change indexing conventions

When changing numerical code, explicitly verify:

- tensor dimensions
- device placement
- dtype consistency
- batch dimensions

Whenever possible, add assertions for expected tensor shapes.

When improving performance:

- explain why
- benchmark if possible
- avoid premature optimization

---

## PyTorch guidelines

Prefer

- vectorized tensor operations
- torch.linalg
- broadcasting
- documentation of expected tensor shapes via inline comments

Avoid

- unnecessary Python loops
- detached tensors
- unnecessary .cpu() calls
- unnecessary conversions to NumPy

Avoid in-place operations unless clearly beneficial.

---

## Reproducibility

Do not change:

- random seeds
- dataset splits
- evaluation protocol

unless explicitly requested.

If changes affect reproducibility, explain why.

---

## What not to do

Do not:

- rewrite large sections without justification
- introduce unnecessary dependencies
- rename APIs purely for style
- change file organization unless it materially improves the project
- optimize away code that improves readability

NEVER WRITE/MODIFY/OUTPUT CODE until explicitly prompted by me. The default is planning, not coding.

NEVER add a new dependency without seeking explicit permission in the planning phase. 

NEVER RUN CODE without seeking explicit permission.

NEVER do any Git commands (stage changes, commit, push, pull). 

---

## When uncertain

If multiple implementations appear reasonable:

1. preserve existing conventions
2. choose the simplest implementation and explain it VERY precisely
3. explain assumptions
4. leave TODO comments rather than guessing
5. explain the alternative implementations, also VERY precisely

Scientific correctness always takes precedence over code elegance.

---


# Project-specific guidelines


## The learning task

For each molecule, given
1. atom positions
2. atom species

Predict the one electron density matrix. 

---

## Dataset assumptions

The primary dataset is QMugs. The paper for this dataset is in `references/QMugs-paper.pdf`. 

The dataset contains `psi4.core.Wavefunction` objects, which contain density matrices computed at 
the DFT level of theory, written in terms of the def2-SVP basis set. 

The largest density matrix present is 2002 by 2002. Some of the smallest are around 150 by 150. 

---

## The current state

Read the `AGENTS.md` file in `examples/qmugs/` for all current information.

---

## Relevant directories

The subdirectories from root we care about for coding are
- `examples/qmugs/`: contains density matrix learning code
- `examples/qmugs_run/`: copy of relevant files from `examples/qmugs/` for running training
- `examples/qmugs_run_mask/`: copy of relevant files from `examples/qmugs/` for running training with masking; requires different JSON configuration file.
- `hydragnn/`: contains code for the HydraGNN module

NEVER modify files in other directories. 

### `examples/qmugs/`

This is where most code edits will go. Read the `AGENTS.md` file in this directory for
a detailed description.

We will NEVER run training with scripts from this directory. 

### `examples/qmugs_run/` and `examples/qmugs_run_mask/`

Copies of relevant files from `examples/qmugs` for running training with different JSON configuration files. 
In this case, the configuration is named `qmugs_densmat.json`. 

Whenever updates are made to files in `examples/qmugs/`, DO NOT COPY any changes made to the following files into 
the two code-running directories UNLESS I specify. The relevant files that would need to be copied, if I do specify, are:

- everything in `utils/` EXCEPT `densmat.png` and `download_data.sh`
- `qmugs_densmat_train.py`
- `qmugs_densmat_inference.py`
- `qmugs.py`


### `hydragnn/`

AVOID modifying files in `hydragnn/` unless explicitly told to do so. These files should only be edited to permit
compatibility with elements of the density matrix learning task. 

The backbone structure should NEVER be edited. The existing input patterns for methods in 
`hydragnn/` should NEVER become defunct; never delete old code paths, only create new ones.

Out of the subdirectories in `hydragnn/`, we will NEVER modify
- `globalAtt/`
- `models/` OTHER THAN `Base.py`
If explicitly prompted to do so, pause and ask. 