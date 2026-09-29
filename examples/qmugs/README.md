# QMugs

## Directory Layout
```
qmugs
|
├── eda                                    # results from eda.py
|   ├── densmat_eda.json                   # EDA configuration
|   ├── entries.png                        # dataset entry counts
|   └── sizes.png                          # density matrix size distribution
|
├── figs                                   # functions to plot things
|   ├── density_matrix_visualizers.py      # density matrix heatmaps
|   ├── density_visualizers.py             # electron density point clouds and isosurfaces
|   ├── dipole_moment_visualizers.py       # dipole moment vectors and parity plots
|   └── metrics_visualizers.py             # error metrics vs. molecule size
|
├── scripts                                # SLURM scripts, one subdirectory per machine
|   ├── dcc                                # Duke Compute Cluster
|   |   ├── qmugs_densmat_dcc.sh           # runs training
|   |   ├── qmugs_eda_dcc.sh               # runs EDA script (NO LONGER USED)
|   |   ├── qmugs_figs_dcc.sh              # runs figures given a previous inference run
|   |   ├── qmugs_inference_dcc.sh         # runs inference
|   |   ├── qmugs_pre_dcc.sh               # runs data preprocessing to prep for training
|   |   ├── qmugs_schur_dcc.sh             # runs training with the Schur decoder
|   |   ├── qmugs_schur_pre_dcc.sh         # runs data preprocessing for the Schur decoder
|   |   ├── qmugs_schur_decomp_test_dcc.sh # runs schur_decomp_test.py (DCC only)
|   |   ├── qmugs_schur_decoder_test_dcc.sh # runs schur_decoder_test.py (DCC only)
|   |   ├── qmugs_scan_dcc.sh              # runs scan_dataset_health.py (DCC only)
|   |   ├── qmugs_inspect_ckpt_dcc.sh      # runs inspect_checkpoints.py (DCC only)
|   |   └── test.sh                        # tests environment and package setup
|   └── perlmutter                         # Perlmutter (NERSC), same six scripts
|
├── utils
|   ├── download_data.json                 # data download config, UPDATE THIS to point to your own directories
|   ├── download_data.py                   # Python wrapper for data download functions
|   ├── download_data.sh                   # shell script to run the data download
|   ├── inference_utils.py                 # utilities used only by the inference path
|   └── utils.py                           # utilities, importantly including density matrix pre/post-processing
|
├── eda.py                                 # script with EDA functionalities for the QMugs dataset
├── figs.py                                # Python script to run figure plotting
├── qmugs_densmat.json                     # HydraGNN configuration for the learning task
├── qmugs_schur.json                       # HydraGNN configuration for the Schur decoder
├── qmugs_densmat_inference.py             # Python script to run inference on a trained model
├── qmugs_densmat_train.py                 # Python script to run preprocessing or training
├── schur_decomp_test.py                   # diagnostics on ground-truth density matrices, for the Schur decoder
├── schur_decoder_test.py                  # verification suite for hydragnn/decoders/schur/
├── scan_dataset_health.py                 # read-only screen of a preprocessed dataset for NaN-producing inputs
├── inspect_checkpoints.py                 # read-only report of where a NaN sits in a saved checkpoint
├── qmugs.py                               # Python file containing QMugsDataset and QMugsInference
├── requirements.txt                       # QMugs-specific dependencies
└── README.md

```
All other files in this directory are log files from previous runs.
---

## Dev Reminders for Perlmutter

### Module Loading and Environment Activation
Before doing anything, make sure the correct modules are loaded on the machine. From the HydraGNN root directory, run
```
source installation_DOE_supercomputers/module_loads_perlmutter.sh
load_perlmutter_modules "12.9"
module -t list | sort
```

Activate the `conda` environment by running
```
conda activate HydraGNN-Installation-Perlmutter/hydragnn_venv/
```

The `psi4` package only supports downloading via `conda`. With the `conda` environment activated and all other HydraGNN dependencies installed, run:
```
conda install psi4 python=3.11 -c conda-forge
```

Navigate to the `qmugs/` directory to run the code:
```
cd examples/qmugs
```

### Environment Setup
Set up the `conda` environment by running
```
bash installation_DOE_supercomputers/hydragnn_installation_bash_script_perlmutter.sh
conda activate HydraGNN-Installation-Perlmutter/hydragnn_venv/
python -m pip install -e . --no-deps
```
Note the `--no-deps` flag avoids reinstalling possibly incompatible versions of torch family modules after compatible ones have been installed by the installation script. 

---

## Dev for Duke Compute Cluster (DCC)

### Module Loading and Environment Activation
Before doing anything, make sure the correct modules are loaded on the machine. From the HydraGNN root directory, run
```
source installation_other/module_loads_dcc.sh
load_dcc_modules "12.8"
module -t list | sort
```

Activate the `conda` environment by running
```
conda activate /hpc/group/youlab/$USER/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv
```

Always load the modules BEFORE activating the environment. `load_dcc_modules` begins with `module purge`, which unloads `Anaconda3` and tears down the `conda` shell function; running it after activation silently leaves you on the system Python. Unlike on Perlmutter, `psi4` is installed by the installation script and needs no separate `conda install` step.

### Environment Setup

Set up the `conda` environment by running
```
bash installation_other/hydragnn_installation_bash_script_dcc.sh
conda activate /hpc/group/youlab/$USER/HydraGNN-Installation-Duke-Compute-Cluster/hydragnn_venv
python -m pip install -e . --no-deps
```
Note the `--no-deps` flag avoids reinstalling possibly incompatible versions of torch family modules after compatible ones have been installed by the installation script. 

The environment is built under `/hpc/group/youlab/$USER/`, NOT inside the repository as on Perlmutter. This is not a preference: the VAST filesystem backing `/work` rejects filenames containing `*`, `?`, `:`, or `|`, and `psi4` ships basis sets named e.g. `6-31g**.g94`, so it cannot be installed there. Override with `INSTALL_ROOT` if needed. See `references/dcc_info/perlmutter_to_dcc_transfer.md`. 

Prefer running the installation script inside a `youlab-gpu` batch job rather than on a login node, so that the CUDA checks are meaningful. Login nodes have no GPU. 

---

## Learning Task Configs

### Learning total density (alpha + beta)
For learning the total density matrix, set `config.json` as follows:
```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0],
    "output_dim": [4008004],
    "output_names": [
        "density_matrix"
    ],
    "type": ["graph"],
    ...
}
```

### Delta learning total density
For delta learning the total density with the promolecular density matrix as a baseline, set `config.json` as follows:

```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0],
    "output_dim": [4008004],
    "output_names": [
        "density_matrix_delta"
    ],
    "type": ["graph"],
    ...
}
```

The preprocessing path requires `--artifacts_dir` and saves unpadded promolecular density matrices as float32 NumPy files under `artifacts_dir/promolecular_density_matrices/<chembl_id>/conf_<conformer_id>.npy`. Training targets are padded total-density deltas, while inference and figures add the stored promolecular baseline back before computing metrics or downstream physical quantities.


### Learning only the upper triangle
To learn only the upper triangle of the total density, set `config.json` as follows:

```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0],
    "output_dim": [2005003],
    "output_names": [
        "density_matrix_uptr"
    ],
    "type": ["graph"],
    ...
}
```

To delta learn only the upper triangle of the total density, set `config.json` as follows:

```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0],
    "output_dim": [2005003],
    "output_names": [
        "density_matrix_delta_uptr"
    ],
    "type": ["graph"],
    ...
}
```


### Learning alpha and beta separately

For learning alpha and beta density matrices separately, set `config.json` as follows:
```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0, 1],
    "output_dim": [4008004, 4008004],
    "output_names": [
        "alpha_density_matrix",
        "beta_density_matrix"
    ],
    "type": ["graph", "graph"],
    ...
}
```


### Masking
NOTE: the padded region is by default masked out from the train, validation, and test losses. The `mask_output_loss` configuration only determines how the train and validation losses are further masked, and does NOT affect the test loss. 

The `data.y_mask` attribute stores this further train and validation mask. The `data.y_test_mask` attribute stores the default mask for the train and validation losses and the only mask for the test loss. 


### Specifying output constraints

The density matrix learning pipeline supports 2 forms of output constraints. By "output constaints",
we mean constraints applied on the output of the last output head of the network before loss computation, such that the network architecture remains identical. 

For learning the total density matrix (full or upper triangle), the `trace_norm` constraint is implemented, which normalizes the trace of the output to equal the number of electrons in the molecule. 

```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0],
    "output_dim": [4008004],
    "output_names": [
        "density_matrix"
    ],
    "output_constraints": [
        ["trace_norm"]
    ],
    "type": ["graph"],
}
```

For delta learning the total density matrix (full or upper triangle), the `trace_zero` constraint is
implemented, which zeros out the trace of the delta matrix, as the promolecular and total density
matrices are expected to have equal trace.

```
"Variables_of_interest": {
    "input_node_features": [0],
    "output_index": [0],
    "output_dim": [4008004],
    "output_names": [
        "density_matrix_delta"
    ],
    "output_constraints": [
        ["trace_zero"]
    ],
    "type": ["graph"],
    ...
}
```

## Schur Decoder

The Schur decoder replaces the dense output head with a sequential construction that builds the
density matrix one atomic orbital at a time. At step `k` the network predicts two latents,
`z_k` and `alpha_k`, and the matrix grows by bordering its leading principal submatrix:

```
c_k = D_{k-1} z_k,    d_k = z_k^T c_k + delta_k,    delta_k = f(alpha_k) >= 0
```

By Theorem 2.1 of `references/schur_decoder.tex` the result is positive semidefinite for *any*
`z_k` and any nonnegative `delta_k`, so constraints (1) and (3) of Section 3.1 — symmetry and
PSD — hold structurally rather than by penalty, and no pseudoinverse is ever formed. Constraint
(2), trace normalization against the AO overlap, is **not** implemented for the decoder.

Three heads, all shared across every step and every molecule, supply the latents: the `d_1` head
of Section 5.2, the `z_{k,i}` head of Section 5.3, and the `alpha_k` head of Section 5.4.

### Using the Schur decoder

Use `qmugs_schur.json`, or add a `SchurDecoder` block to your own config as a sibling of
`Architecture`, `Variables_of_interest` and `Training`:

```
"SchurDecoder": {
  "enable": true,

  "conditioning_type": "concat",
  "use_sequence_position": true,
  "condition_z_on_graph": false,

  "orbital_dim": 64,
  "orbital_hidden_dims": [128],
  "global_dim": 32,
  "global_hidden_dims": [64],
  "z_hidden_dims": [128, 128],
  "alpha_hidden_dims": [128, 128],
  "d1_hidden_dims": [64],
  "num_radial_basis": 16,
  "radial_cutoff": 10.0,
  "f": "softplus",
  "alpha_bias_init": -2.0,

  "target": "delta",
  "routing": "teacher_forcing",
  "supervision": "joint",
  "chunk_width": null,
  "epochs_per_chunk": 1,

  "w_schedule": "per_entry",
  "step_normalization": "entry_count",
  "z_regularization_weight": 0.0,
  "delta_regularization_weight": 0.0,

  "z_row_chunk": 128,
  "checkpoint_z_head": true,
  "max_basis_dim": 800,
  "report_stride": 50
}
```

Setting `"enable": false`, or leaving the block out entirely, leaves the run behaving exactly as
it does today. Starting from `qmugs_densmat.json`, four things change:

1. add the `SchurDecoder` block;
2. set `"output_dim": [1]` and `"dim_headlayers": [32]`. The Base output head is never called
   once the decoder is attached, and this is what makes it free. It also removes the
   `Linear(10000, 2005003)` the dense configuration carries, which alone is 2e10 parameters;
3. keep `"output_names"` as it is. All four variants work: the decoder unpads each molecule to
   its own `n_basis`, accepts either the square or the upper-triangle layout, and rebuilds the
   total from `y + D_pro` when the name is a delta variant. `qmugs_schur.json` ships with
   `"density_matrix_delta_uptr"` so that it can share a dataset with the dense-head route;
4. set `"batch_size": 8` and `"continue": 0`.

### Configuration keys

| key | reference | meaning |
|---|---|---|
| `enable` | — | false leaves the run on the existing dense head |
| `conditioning_type` | 5.2-5.4 | how metadata and the graph enter the heads; only `concat` is implemented |
| `use_sequence_position` | — | append `k / n_basis` to the per-orbital vector |
| `condition_z_on_graph` | 5.3 | concatenate the graph vector into the `z_{k,i}` head; off by default |
| `orbital_dim`, `orbital_hidden_dims` | — | the per-orbital projection shared by all three heads |
| `global_dim`, `global_hidden_dims` | 5.1 | the per-molecule projection of the pooled embedding and the molecular scalars |
| `z_hidden_dims` | 5.3, Eq. 12 | hidden widths of the `z_{k,i}` head |
| `alpha_hidden_dims` | 5.4, Eq. 13 | hidden widths of the `alpha_k` head |
| `d1_hidden_dims` | 5.2, Eq. 11 | hidden widths of the `d_1` head |
| `num_radial_basis`, `radial_cutoff` | 5.3 | fixed Gaussian basis on the interatomic distance |
| `f` | 3.4 | the nonnegative map giving `delta_k = f(alpha_k)`; `softplus` or `exp` |
| `alpha_bias_init` | — | initial bias of the final layer, so `delta_k` starts near the median target diagonal |
| `target` | 3.5, Eq. 17 | `total` compares against `D`, `delta` against `D - D_pro` |
| `routing` | 6.4 | `teacher_forcing` or `free_running` |
| `supervision` | 6.1-6.3 | `joint`, `semi_sequential` or `per_step` |
| `chunk_width`, `epochs_per_chunk` | 6.3 | the window of steps supervised, and how many epochs it is held |
| `w_schedule` | 4.2 | the step weights `w_k`; `uniform` weights each step equally, `per_entry` each matrix entry |
| `step_normalization` | 4.2 | the per-step normalization `N_k`; `entry_count` is `1/(2k+1)` |
| `z_regularization_weight`, `delta_regularization_weight` | 4.3 | the two penalty weights, both 0 by default |
| `z_row_chunk`, `checkpoint_z_head` | — | memory controls for the pair head |
| `max_basis_dim` | — | conformers larger than this are skipped and logged |
| `report_stride` | — | one diagnostic line every this many calls to `loss()`; `0` silences it |

**How each head is conditioned.** The reference leaves "metadata" and the graph open in
Equations 11-13. This implementation resolves them as concatenation followed by an MLP, in three
stages. First a per-orbital vector, built once per molecule:

```
v_orbital = MLP_orb( [ v[j(k)], onehot(l), onehot(index within shell),
                       onehot(shell within atom), onehot(species), k / n_basis ] )
```

where `j(k)` is the atom carrying orbital `k` and `v[j(k)]` is that atom's node embedding out of
the message-passing trunk. The `k / n_basis` channel is not in the reference; it is there because
Section 10.3 of `logs/qmugs-schur-diagnostics/schur_decomp_test.md` found the difficulty of a step
to be a strong function of `k/K` — the median `||z_k||` runs from 0.70 in the first tenth of the
sequence to 0.028 in the last twentieth. Drop it with `"use_sequence_position": false`.

Second a per-molecule vector, which is how the graph and the a priori metadata enter:

```
g = MLP_global( [ G, n_atoms / 100, n_basis / 1000, charge, multiplicity ] )
```

Third the heads themselves:

```
d_1     = f( MLP_d1(    [ v_orbital[0], g ] ) )
alpha_k =    MLP_alpha( [ v_orbital[k], g ] )
z_{k,i} =    MLP_z(     [ v_orbital[k], v_orbital[i], rbf(|r_{j(k)} - r_{j(i)}|) ] )
```

The graph vector `g` is left out of the `z_{k,i}` head by default. Section 5.3 says the pair of
node embeddings are the primary inputs, "with the rest of `G` present only as auxiliary
information if needed... the hope is that the encoder already injected this information into
`v^{j(k)}` and `v^{j(i)}`". It is also the one input that would be broadcast across all `K^2`
pairs, adding about 82 MB to the concatenated pair tensor at `K = 800` for no new information.
Set `"condition_z_on_graph": true` to include it.

To change the parametric form of the conditioning, edit the `conditioning()` and
`conditioning_dim()` methods on the relevant head in
`hydragnn/decoders/schur/heads.py`. Each head's first linear layer is sized from
`conditioning_dim()`, so the two cannot drift apart without an immediate shape error, and
`schur_decoder_test.py` checks that they agree across the width options. The natural second form
to try is FiLM, `h <- (1 + scale(g)) * h + shift(g)`, which conditions the `z_{k,i}` head on `g`
without widening the `K^2`-row tensor at all; `Base._ensure_graph_conditioner` in
`hydragnn/models/Base.py` is an existing implementation to copy the shape of.

**Choosing `target`.** Theorem 2.2 runs the recursion on the total matrix whichever target is
chosen, because Equation 17's `(D_pro_{k-1} + Delta D_{k-1})` is just `D_{k-1}`. The same latents
therefore produce both, and the residual is literally the same matrix:

```
Delta D_hat - Delta D = (D_hat - D_pro) - (D - D_pro) = D_hat - D
```

So under MAE with `entry_count` normalization the two settings agree to the last bit. The choice
starts to matter as soon as the loss or the metric depends on the target's own magnitude: a
relative error, where `||Delta D||` is much smaller than `||D||`, or a target-dependent `N_k` or
weight matrix `W`. Those settings are the natural follow-up and are not implemented yet. The
delta-relative error is logged either way, since it is the more demanding number.

`"target": "delta"` requires a dataset preprocessed with `--density_matrix_delta`, which is what
stores the promolecular baseline in the batch.

**Choosing `routing`.** Teacher forcing feeds the ground-truth `D_{k-1}` into `c_k = D_{k-1} z_k`;
free running feeds the model's own reconstruction. Teacher forcing is far better conditioned,
being linear in `z_k` given the true block, but the trained model then has an exposure gap when
it has to run autoregressively at inference. Free running has no gap, since training and
inference compute the same thing, but it propagates every step's error into the next through
`dc_k = dD_{k-1} z_k`, which is what Section 10.5 of the diagnostics measured. Section 6.4 of the
reference recommends the obvious curriculum: teacher forcing first, then free-running fine
tuning. Note also that only the free-running reconstruction carries the PSD guarantee; each
teacher-forced step block is PSD, but the assembled teacher-forced matrix is not the output of
any single valid recursion.

**Choosing `supervision`.** All three schemes read off the same forward pass and differ in which
steps carry nonzero `w_k` in a given epoch:

| mode | reference | active window per epoch |
|---|---|---|
| `joint` | 6.1 | all `k`, every epoch |
| `semi_sequential` | 6.3 | a contiguous window of `chunk_width` steps, advancing every `epochs_per_chunk` epochs |
| `per_step` | 6.2 | the same with `chunk_width` fixed at 1 |

This is the curriculum reading of Section 6.2 — run the decoder, take the loss against `D_k`,
backpropagate until convergence, then increment `k` — with "until convergence" realized as
`epochs_per_chunk`. All three take one optimizer step per batch. The two non-joint modes detach
the recursion at window boundaries, which makes them *cheaper* than `joint`, not more expensive.
The literal reading of Section 4.2(3), several optimizer steps inside one batch, would need
`retain_graph=True` once per window and is not implemented.

### Running it on DCC

The Schur task has its own SLURM scripts, with their own config, checkpoint directory and shell
variables. They **share a dataset** with the dense-head route, which is what makes the two
comparable:

```
sbatch scripts/dcc/qmugs_schur_pre_dcc.sh     # preprocessing, writes QMugs001_max800_delta_uptr_decodercmp
sbatch scripts/dcc/qmugs_schur_dcc.sh         # training
```

`scripts/dcc/qmugs_pre_dcc.sh` writes that same dataset and the same artifacts directory, so
**run one of the two preprocessing scripts, not both**. They are separate entry points only
because each route has its own config file, and they are interchangeable because preprocessing
reads exactly four things from the config, on which `qmugs_densmat.json` and `qmugs_schur.json`
agree: `Architecture.radius`, `Architecture.max_neighbours`,
`Variables_of_interest.output_names` and `Variables_of_interest.mask_output_loss`. Change any of
those in one config and the two routes stop sharing a dataset.

One dataset serves both because `_mol_to_graph` attaches the per-orbital metadata to every
conformer regardless of task, and the delta output name additionally stores the promolecular
upper triangle. The dense head reads `y` (which is `Delta D`); the decoder reads `y` plus the
promolecular triangle and rebuilds the total before its recursion.

**Preprocessing must be re-run** before the first training job. The decoder reads four
per-orbital fields that older pickled datasets do not carry, and the one-hot widths and species
list are derived from the dataset rather than assumed. Training stops with an explicit error
naming the missing metadata keys if the dataset predates this change. `SCHUR_MODELNAME` must
agree between the two scripts, since it is the on-disk dataset name, and
`SCHUR_MAX_DENSITY_MATRIX_SIZE` should be at or below `max_basis_dim` in the config so that
conformers are filtered at load time rather than skipped inside the decoder.

**Comparing the two decoders.** With the shared dataset the remaining difference is the loss
itself: the dense head takes a plain MAE over the unpadded upper triangle, while the decoder
takes the per-step form of Section 4.2. Setting `"w_schedule": "per_entry"` collapses the
per-step form exactly to the plain full-matrix MAE, which differs from the dense head's
upper-triangle MAE only by the constant factor `K^2 / T(K)` of about 2. `qmugs_schur.json`
ships with `per_entry` and `"target": "delta"` for that reason. Leave `"w_schedule": "uniform"`
only when the point is the decoder's own objective rather than a head-to-head comparison.

The three settings most worth sweeping are exposed as command line overrides, so a sweep needs
no config edits and the saved config still records what actually ran:

```
SCHUR_ROUTING=free_running SCHUR_SUPERVISION=per_step sbatch scripts/dcc/qmugs_schur_dcc.sh
```

The run name records all three, so sweep points do not overwrite each other's logs.

Note that the training script sets `HYDRAGNN_USE_FSDP=0`, against 1 in the dense-head script.
The decoder is attached by composition around the backbone and its unused Base heads are frozen
so that DDP with `find_unused_parameters=False` accepts them; sharding the wrapper under FSDP is
untested.

### What to expect

- *Memory.* The recursion grows the matrix out of place, so the retained autograd graph is about
  `K^3 / 3` floats per molecule: 288 MB at `K = 600` and 683 MB at `K = 800` in fp32. `batch_size`
  is the lever; 4 to 8 is the working range. Typical QMugs conformers sit at `K` of 300 to 600,
  four to eight times cheaper than the worst case.
- *Throughput.* The recursion is `K` sequential steps of a few small operations each, so it is
  latency bound rather than compute bound: roughly 0.6 s per batch at `K = 800` with
  `batch_size` 8. If that becomes the binding cost, the fix is to run one masked loop over the
  whole batch instead of one loop per molecule.
- *The pair head is the other cost.* It evaluates an MLP on all `K^2` ordered orbital pairs.
  With `checkpoint_z_head` on, only the `(K, K)` output survives into the backward pass, at the
  price of one recomputed forward. Turning it off is roughly 660 MB per molecule at `K = 800`.
- *Molecules above `max_basis_dim`* are skipped with a log line, not silently dropped. If every
  conformer in a batch is skipped the run stops with an error.
- *Checkpoints are not interchangeable.* Attaching the decoder prefixes every state-dict key
  with `backbone.`, so a checkpoint from a dense-head run will not load into a Schur model, or
  the reverse. Use a fresh `--modelname` and keep `"continue": 0`.
- *Inference and figures are not wired up yet.* `qmugs_densmat_inference.py` and `figs.py` still
  expect a padded upper-triangle vector from a dense head.

### Verifying the implementation

`schur_decoder_test.py` checks the decoder in float64 on CPU: PSD and symmetry of the
reconstruction, an oracle round trip in which exact latents from an independent unpivoted
`LDL^T` factorization must drive the recursion back to the matrix they came from, teacher forcing
and the PSD of every step block, that detaching changes gradients and not values, that the
per-step loss collapses to the plain full-matrix loss at the matching weights, `gradcheck` in
both routings, the delta-learning identities, that every head's declared conditioning width
matches what it assembles, and an end-to-end pass over a two-molecule batch of different sizes.
Run it with

```
sbatch scripts/dcc/qmugs_schur_decoder_test_dcc.sh
```

Set `DECODER_WFN_PATH` to a Psi4 wavefunction file to run the oracle checks on a real conformer
as well as on synthetic matrices. The script exits nonzero if any check fails.

## Diagnosing NaNs during training

Several runs, with the dense head and with the Schur decoder alike, produced NaN after roughly a
thousand optimizer steps. Under DDP or FSDP the gradients are all-reduced before the step, so one
non-finite gradient on one rank writes NaN into the parameters of every rank, and the run then
fails somewhere unrelated many batches later. The tools below exist to find the first offender
rather than the eventual symptom.

**Guarding the optimizer step.** Two environment variables, both inert when unset, so leaving
them out reproduces the old behaviour exactly:

| variable | effect |
| --- | --- |
| `HYDRAGNN_CLIP_GRAD_NORM` | clip the total gradient norm to this value before every step |
| `HYDRAGNN_SKIP_NONFINITE_STEP` | `1` skips a step whose gradient norm is NaN or Inf, logging the epoch, batch and the ChEMBL ids in that batch, instead of writing it into the parameters |
| `HYDRAGNN_GRAD_NORM_STRIDE` | log the gradient norm every this many batches even when it is finite |

Turning on the skip alone costs one extra reduction per batch and answers the question "which
conformers were in the batch that broke it". The stride answers the other half: a norm that creeps
up over epochs is a slow blow-up, one that is flat and then jumps in a single step is one bad
batch, and the loss shows neither. Turning on clipping as well is the remedy once the answer is in.

Both DCC training scripts export all three with `:-0` defaults, so they are set from the submitting
shell and no file needs editing:

```
HYDRAGNN_SKIP_NONFINITE_STEP=1 HYDRAGNN_GRAD_NORM_STRIDE=10 \
    sbatch scripts/dcc/qmugs_schur_dcc.sh
```

**Where the NaN entered the decoder.** When the Schur decoder sees a non-finite `z` or `delta` it
walks the stages in order — backbone `v` and `G`, positions, orbital embedding, global embedding,
then each head — and logs which one is already non-finite, separating an encoder failure from a
decoder failure. Independently, `report_stride` in the `SchurDecoder` block prints one diagnostic
line every that many calls to `loss()`, carrying the reconstruction and regularization terms, both
relative Frobenius errors, and `max|z|`, `max delta`, `max|D_hat|`. The latent magnitudes are the
drift to watch: the ideal `z_k` solves `c_k = D_{k-1} z_k` with a rank-deficient `D_{k-1}`, so
`max|z|` can grow steadily while the loss still falls. Set `report_stride` to `0` to silence it.

**Screening the dataset.** `scan_dataset_health.py` opens the pickled conformers and reports the
minimum interatomic distance, atoms with no neighbour inside the cutoff, self-loops, non-finite
positions or targets, and the distribution of `max |target entry|`, naming the worst conformers in
each case. PAINN divides by the interatomic distance twice, so a conformer with two atoms nearly on
top of each other yields enormous but finite features: a trigger that only becomes fatal once the
weights have drifted. It is read-only, single process, and uses no GPU. Torch is not installed on
the DCC login nodes, so it runs as a batch job:

```
sbatch scripts/dcc/qmugs_scan_dcc.sh
```

It lands on `youlab-gpu` only because the `youlab` CPU nodes are saturated; it requests no
`--gres`, so the node's GPUs stay free. The defaults scan the three draws that matter for the NaN
investigation. Override with `SCAN_DATASETS`, `SCAN_DATASET_DIR`, `SCAN_LABELS`, `SCAN_RADIUS`,
`SCAN_N_REPORT`, and `SCAN_LIMIT`, for instance

```
SCAN_DATASETS="QMugs001_max800_delta_uptr_decodercmp" SCAN_LIMIT=2000 \
    sbatch scripts/dcc/qmugs_scan_dcc.sh
```

The cost is dominated by reading the pickles, roughly 2.5 MB per conformer: about 18 GB for the
0.01 draw and 180 GB for the 0.1 draw, which is what `SCAN_LIMIT` is for. Keep `SCAN_RADIUS` equal
to the `radius` the graphs were built with, or the isolated-atom count means nothing.

Note that `qmugs.py` draws molecules with `random.sample`, so the `perc_load` subsets are
independent draws rather than nested ones, and a pathological conformer can be present in one and
absent from another.

**Reading a checkpoint.** `inspect_checkpoints.py` memory maps a saved `.pk` and reports, per
parameter and rolled up per module, the number of entries, the number of non-finite entries and
the largest finite magnitude, then does the same for the AdamW moments. It answers two questions
that no amount of log reading can: which modules hold the NaN in a poisoned checkpoint, and
whether the weights of a run that later died were already anomalously large compared with a
sibling run that survived longer. It is read-only, single process, and uses no GPU:

```
sbatch scripts/dcc/qmugs_inspect_ckpt_dcc.sh
```

Override `CKPT_PATHS` with a space-separated list to point it elsewhere, and `CKPT_TOP` to change
how many extreme tensors are named. The dense checkpoints are 6.3 GB because the output head is
`1600 x 320400` and AdamW keeps two moments of it, which is why the job asks for 48 GB and the
script scans in 16M-element chunks rather than calling `.abs().max()` on whole tensors.

**What the runs so far have established.** The dataset scan found no degenerate geometry, no
isolated atoms, no self-loops and no non-finite inputs or targets in any draw, so the inputs are
not at fault. Single-variable controls then exonerated the loss function (MSE failed at epoch 7
against MAE's 10), FSDP (plain DDP failed at epoch 19) and the Schur decoder itself, whose `max|z|`
plateaus around 1.4 rather than running away. The decoder's stage trace localized the failure: on
the batch that killed run 53692784 the positions were finite, the backbone `v` was not, and the
orbital embedding — which reads only discrete metadata and nothing continuous from the conformer —
was also non-finite, which only NaN weights can do. So the NaN is created in a backward pass and
written into the shared PAINN backbone by an optimizer step, and the giant dense output head is
not involved. PAINN carries no normalization: `node_scalar += a_sv * inner_prod` with `inner_prod`
quadratic in the vector features, iterated over six layers (`hydragnn/models/PAINNStack.py:298`).

## Other Architecture Configs (added for this task)

### Concat-style graph pooling

The default pooling modes (`"sum"`, `"mean"`, `"max"`) are permutation invariant; they collapse the
node embeddings into a single `hidden_dim`-wide vector in which the atom ordering is lost. But the
target density matrix is indexed by atomic-orbital blocks ordered by the atom ordering in the
SDF/wfn files, which is the same ordering as the graph nodes (specifically, the ordering used by Psi4 
for the def2-SVP basis). Concatenation pooling zero-pads the node embeddings to a fixed number 
of atoms and flattens them, so the head receives a per-atom resolved representation in which row/column 
position is still recoverable.

To use it, set the `Architecture` block of `config.json` as follows:

```
"Architecture": {
    "mpnn_type": "PAINN",
    "hidden_dim": 200,
    "graph_pooling": "concat",
    "output_heads": {
        "graph": {
            "num_sharedlayers": 0,
            "dim_sharedlayers": 400,
            "num_headlayers": 2,
            "dim_headlayers": [1000, 4000]
        }
    },
    ...
}
```

The pooled embedding has width `pooled_dim = max_nodes * hidden_dim`, where `max_nodes` is the
padding width in atoms.

**Specifying `max_nodes`.** You may set `"max_nodes"` explicitly in the `Architecture` block to pin
the architecture. If you leave it out, `qmugs_densmat_train.py` fills it from the `max_natoms` value
recorded in the pickle metadata during preprocessing, which is the largest atom count over the
train, validation, and test splits. Datasets preprocessed before this feature was added do not
carry `max_natoms`; training will stop with an error asking you to either set `max_nodes` by hand or
re-run preprocessing.

**Automatic widening of the first layer.** A single `Linear(pooled_dim, dim_sharedlayers)` would
contract the concatenated embedding by two orders of magnitude in one step. The training script
therefore widens whichever layer consumes the pooled embedding to `pooled_dim`, and logs a warning
naming the old value, the new value, and the contraction ratio it avoided:

- with `"num_sharedlayers": 0` the shared trunk is skipped entirely and the head consumes the
  pooled embedding directly, so `dim_headlayers[0]` is widened;
- with `"num_sharedlayers"` at least 1, `dim_sharedlayers` is widened instead.

Setting `"num_sharedlayers": 0` is the recommended configuration, since it leaves exactly one
`pooled_dim`-wide layer in the network and lets `dim_headlayers` control everything downstream of
it. Note that `dim_sharedlayers` must still be present in the config even when it is ignored.

Both the resolved `max_nodes` and the widened layer dimension are written into the saved
`logs/<run>/config.json`, so inference reconstructs the identical model with no extra flags.

**What to expect.**

- *Parameter count.* At `hidden_dim: 200` and roughly 160 atoms, `pooled_dim` is about 32,000 and
  the widened layer alone holds about 1.0e9 parameters, on top of whatever the final
  `Linear(dim_headlayers[-1], output_dim)` costs. Check the warning in the log for the exact figures
  for your dataset, and size `hidden_dim` and `dim_headlayers` accordingly.
- *Loss of permutation invariance.* This is the intended trade, but now the
  model is tied to the atom ordering in the source files. That ordering must be identical between
  preprocessing and inference, and the model will not generalize to a reordering of the same
  molecule. The QMugs SDF/wfn files provide a fixed ordering, which is the same ordering that
  defines the basis ordering of the target, so this holds for the pipeline as written.
- *Molecules larger than `max_nodes`.* Training or inference stops with an error naming the
  offending atom count. It does not silently drop atoms.
- *Supported models.* Only `"PAINN"` and `"EGNN"` accept `"concat"`. Other `mpnn_type` values 
  raise an error at model construction.

Existing checkpoints trained with `"sum"` or `"mean"` pooling are unaffected and still run under
`qmugs_densmat_inference.py` with no config edits, since each checkpoint is rebuilt from its own
saved `config.json`.
