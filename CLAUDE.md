# CLAUDE.md

This repository is that of HydraGNN, a PyTorch implementation framework for training multihead graph neural networks, specifically for materials applications. 
My goal is to implement a use case of learning one-electron density matrices (1-RDMs) of drug-like molecules from the QMugs dataset. 

This is a research codebase. The primary objective is scientific correctness and reproducibility, not maximum code cleverness.
Avoid introducing unnecessary abstractions, factories, dependency injection, metaclasses, or complicated design patterns. 
Small duplicated code is preferable to complicated generic code.

---

# User background

I am an applied mathematician from a math and physics background with 2-3 years of 
scientific machine learning experience with PyTorch. I prioritize precision and formality. 
I do not prioritize beautiful code. I like conciseness. 

I like writing that is readable without me having to strain my eyes and brain.
I do not like the use of fancy syntax in writing, or symbols like "->". Commas and periods suffice.

I have the following gaps in knowledge:
- almost no experience with MPI or parallelism in general on CPUs or GPUs
- little experience working with massive clusters such as Perlmutter
- sparse knowledge of quantum chemistry, but good foundation in quantum mechanics
- little experience with conventional computational chemistry packages

---

# High-level information

- I am coding and running everything on Duke Compute Cluster. Information on the cluster is in the codebase
- Much of the code was developed on Perlmutter. Information on the cluster is also in the codebase
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

## Writing guidelines

Whenever you are writing something user-facing, i.e. a README or a markdown to log/report something,
avoid the playful, fancy jargon and syntax LLMs are optimized for, if you can help it. 

---

## "Good Science" guidelines

I want to emphasize again: scientific correctness is PARAMOUNT. Follow these rules, always:
- No shortcuts. If you even get an urge to take a shortcut, stop immediate and report it.
- No reporting results without an associated, verifiable script. 
- No claiming a result has numerical evidence without actually running code, i.e. no guesses based on rough scans. 
- No skipping evidence or failing to report evidence which does not line up with your hypothesis or my hypothesis.
- No assumptions made without verifying with me and with the references. 

---

## Python guidelines

Use

- type hints for parameters in function declaration, only when the parameter is a Python primitive or `np.ndarray`
- os.path functions instead of pathlib
- built in logging structure instead of print
- built in HydraGNN functions whenever possible and convenient
- simple in-line comments which match the commenting style present in scripts in the `examples/qmugs/` directory
- simple, 1-2 line docstrings for functions in the style present in scripts in the `examples/qmugs/` directory

Avoid

- defining new dataclasses UNLESS specified in the task
- writing long docstrongs UNLESS specified in the task
- deeply nested conditionals
- helper function overload; fold single-use helpers into parent functions
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

## Reproducibility guidelines

Do not change:

- random seeds
- dataset splits
- evaluation protocol

unless explicitly requested.

If changes affect reproducibility, explain why.
Create user-friendly READMEs whenever you see fit.

---

## Comment guidelines

- in-line comments should ALMOST ALWAYS be one line; take efforts to avoid two-line comments
- comments SHOULD NOT use math symbols; those look pretty when rendered but are utterly unreadable and non-sensical in plain text
- comments SHOULD NOT use variables, EVEN IF you define them, e.g. "NxN"; just write out what the thing is in words
- comments NEED NOT be fanciful sentences with fancy diction; no human should need to do a double take when reading them
- in fact, comments SHOULD NOT CONTAIN PERIODS, as they are not sentences, but comments
- following up the previous point, comments are NOT sensationalit media/headlines; treat them like sentences you would see in a textbook or a research paper: precise, concise, says exactly what it needs to say without fancy syntax or diction, without pretty words like "fail silently", without fancy punctuation
- if multiple lines is really, truly necessary to clearly communicate what is going on, please adhere to these other guidelines

Remember that comments are meant to help the coder understand code, not to show off how efficiently and beautifully you can explain the code.
Buzz words like "silently", "quietly", "loudly", etc. are NOT how coders talk to each other, and should thus not be how you write comments.

---

## SLURM guidelines

- you may run jobs if I instruct you to, but ALWAYS report when you do and ask permission
- to repeat, you MUST ask permission to run a series of jobs (this does not mean you need to stop between every job; if I approve you to run a series of related jobs, monitor them, and queue new ones based on their performance, you may do so)
- follow the conventions in the existing SLURM scripts

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
NEVER do any Git commands (stage changes, commit, push, pull) without very explicit permission.

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

Read the `README.md` and `CLAUDE.md` file in `examples/qmugs/` for an overview of the current state.
Read the many markdowns previous agent sessions have produced in the codebase for a precise picture.

---

## Relevant directories

The subdirectories from root we care about for coding are
- `examples/qmugs/`: contains density matrix learning code
- `hydragnn/`: contains code for the HydraGNN module

NEVER modify code files in other directories without explicit permission.
You may modify log files and such, but be EXPLICIT when you do so.

### `examples/qmugs/`

This is where most code edits will go. Read the `CLAUDE.md` file in this directory for a description

### `hydragnn/`

AVOID modifying files in `hydragnn/` unless explicitly told to do so. These files should only be edited to permit
compatibility with elements of the density matrix learning task. 

The backbone structure should NEVER be edited. The existing input patterns for methods in 
`hydragnn/` should NEVER become defunct; never delete old code paths, only create new ones.

Out of the subdirectories in `hydragnn/`, we will NEVER modify
- `globalAtt/`
- `models/` OTHER THAN `Base.py`

If explicitly prompted to do so, pause and ask. 